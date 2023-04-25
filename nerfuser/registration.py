import argparse
import json
import os
import re
import shutil
from collections import defaultdict
from datetime import datetime
from itertools import cycle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from nerfstudio.process_data.colmap_utils import (
    CameraModel,
    read_images_binary,
    run_colmap,
)
from nerfstudio.process_data.hloc_utils import run_hloc
from nerfuser.utils.utils import (
    avg_trans,
    complete_transform,
    compute_trans_diff,
    decompose_sim3,
    extract_colmap_pose,
    gen_hemispherical_poses,
)
from nerfuser.view_renderer import ViewRenderer

# from nerfstudio.utils.fuser_utils.visualizer import Visualizer


class Registration:
    def __init__(
        self,
        name,
        model_method,
        model_names,
        model_gt_trans,
        model_dirs,
        step,
        cam_info,
        downscale_factor,
        chunk_size,
        training_poses,
        n_hemi_poses,
        render_hemi_views,
        fps,
        sfm_tool,
        sfm_wo_training_views,
        sfm_w_hemi_views,
        output_dir,
        render_views,
        run_sfm,
        compute_trans,
        vis,
    ) -> None:
        self.name = name
        self.model_method = model_method
        self.model_names = model_names
        self.model_gt_trans = model_gt_trans
        self.model_dirs = model_dirs
        self.step = step
        self.cam_info = cam_info
        self.downscale_factor = downscale_factor
        self.chunk_size = chunk_size
        self.training_poses = training_poses
        self.n_hemi_poses = n_hemi_poses
        self.render_hemi_views = render_hemi_views
        self.fps = fps
        self.sfm_tool = sfm_tool
        self.sfm_wo_training_views = sfm_wo_training_views
        self.sfm_w_hemi_views = sfm_w_hemi_views
        self.output_dir = output_dir
        self.render_views = render_views
        self.run_sfm = run_sfm
        self.compute_trans = compute_trans
        self.vis = vis

    def run(self):
        if self.name:
            name = self.name
        else:
            name = datetime.now().strftime('%m.%d_%H:%M:%S')
        output_dir = Path(self.output_dir) / name
        os.makedirs(output_dir, exist_ok=True)
        model_dirs = [Path(model_dir) for model_dir in self.model_dirs]
        n_models = len(model_dirs)
        model_names = self.model_names if self.model_names else [f'nerf{i}' for i in range(n_models)]
        if self.render_hemi_views:
            cfg = f'hemi{self.sfm_w_hemi_views:.2f}'
            if self.training_poses:
                cfg += f'_train{int(not (self.render_hemi_views and self.sfm_wo_training_views))}'
        else:
            cfg = 'train' if self.training_poses else 'hemi'
        sfm_dir = output_dir / f'sfm_{cfg}'
        log_dict = {k: getattr(self, k) for k in ['model_method', 'model_gt_trans', 'step', 'downscale_factor', 'training_poses', 'n_hemi_poses', 'sfm_tool']}
        log_dict['cam_info'] = [(str(cam_info) if isinstance(cam_info, Path) else cam_info) for cam_info in self.cam_info]
        log_dict['model_dirs'] = {model_name: str(model_dir) for model_name, model_dir in zip(model_names, model_dirs)}
        print('log_dict', log_dict)
        with open(output_dir / f'{cfg}.json', 'w') as f:
            json.dump(log_dict, f, indent=2)

        # nerf-to-normalized transforms
        Ts_nerf_norm = []
        Ss_nerf_norm = []
        for model_dir in model_dirs:
            with open(model_dir.parent / 'dataparser_transforms.json') as f:
                transforms = json.load(f)
            s = transforms['scale']
            Ss_nerf_norm.append(np.diag((s, s, s, 1)).astype(np.float32))
            Ts_nerf_norm.append(Ss_nerf_norm[-1] @ complete_transform(np.array(transforms['transform'], dtype=np.float32)))
        Ts_nerf_norm = np.stack(Ts_nerf_norm)
        Ss_nerf_norm = np.stack(Ss_nerf_norm)
        S_invs_nerf_norm = np.linalg.inv(Ss_nerf_norm)
        if self.model_gt_trans:
            # gt world-to-nerf transforms
            Ts_gt_world_nerf = np.broadcast_to(np.identity(4, dtype=np.float32)[None], (n_models, 4, 4)) if self.model_gt_trans.lower() in {'i', 'identity'} else np.load(self.model_gt_trans)
            # gt world-to-normalized transforms
            Ts_gt_world_norm = Ts_nerf_norm @ Ts_gt_world_nerf
            T_invs_gt_world_norm = np.linalg.inv(Ts_gt_world_norm)
            _, s = decompose_sim3(Ts_gt_world_norm)
            Ss_gt_world_norm = np.zeros((n_models, 4, 4))
            for i in range(3):
                Ss_gt_world_norm[:, i, i] = s
            Ss_gt_world_norm[:, 3, 3] = 1
        if self.training_poses:
            frames = []
            ls = []
            for training_pose in self.training_poses:
                with open(training_pose) as f:
                    transforms = json.load(f)
                frames.append(transforms['frames'])
                ls.append(len(frames[-1]))
            ls = np.array(ls)
            ks = np.floor(ls * 1.3).astype(int)
        else:
            ls = np.full(n_models, self.n_hemi_poses)
            ks = np.floor(ls * 1.3).astype(int) if self.render_hemi_views else ls
        m = (1 + np.sqrt(1 + ks / 3)) / 2
        ms = np.ceil(m).astype(int)
        ns = np.ceil(12 * (m - 1)).astype(int)
        rng = np.random.default_rng(0)

        if self.render_views:
            # if isinstance(self.cam_info, Path):
            if len(self.cam_info) == 1:
                with open(self.cam_info[0]) as f:
                    transforms = json.load(f)
                cam_info = {'fx': transforms['fl_x'], 'fy': transforms['fl_y'], 'cx': transforms['cx'], 'cy': transforms['cy'], 'width': transforms['w'], 'height': transforms['h'], 'distortion_params': np.array([transforms['k1'], transforms['k2'], 0, 0, transforms['p1'], transforms['p2']], dtype=np.float32)}
            else:
                cam_info = dict(zip(['fx', 'fy', 'cx', 'cy', 'width', 'height'], [float(v) for v in self.cam_info]))
            if self.downscale_factor:
                for pname in cam_info:
                    if pname != 'distortion_params':
                        cam_info[pname] /= self.downscale_factor
            cam_info['height'] = int(cam_info['height'])
            cam_info['width'] = int(cam_info['width'])
            for i, model_dir in enumerate(model_dirs):
                poses_norm = complete_transform(np.array(gen_hemispherical_poses(1, np.pi / 6, m=ms[i], n=ns[i])))[np.sort(rng.permutation(ms[i] * ns[i])[:ks[i]])] if not self.training_poses or self.render_hemi_views else np.empty((0, 4, 4), dtype=np.float32)
                if self.training_poses:
                    pose_dict = {frame['file_path']: np.array(frame['transform_matrix'], dtype=np.float32) for frame in frames[i]}
                    poses_norm = np.concatenate((poses_norm, Ts_nerf_norm[i] @ np.stack([pose_dict[k] for k in sorted(pose_dict.keys())]) @ S_invs_nerf_norm[i]))
                with torch.no_grad():
                    ViewRenderer(self.model_method, model_names[i], model_dir, load_step=self.step, chunk_size=self.chunk_size).render_views(poses_norm, cam_info, output_dir, animate=self.fps)
                np.save(output_dir / f'poses-{model_names[i]}_norm.npy', poses_norm)

        if self.run_sfm:
            shutil.rmtree(sfm_dir, ignore_errors=True)
            os.makedirs(sfm_dir)
            # uses hemi views, maybe training views
            c1 = not self.training_poses or self.render_hemi_views
            # uses hemi + training views
            c2 = self.training_poses and self.render_hemi_views and not self.sfm_wo_training_views
            # uses training views only
            c3 = self.training_poses and not self.render_hemi_views
            for i, model_name in enumerate(model_names):
                files = os.listdir(output_dir / model_name)
                if c1:
                    ids = set(rng.permutation(ks[i])[:np.floor(ls[i] * self.sfm_w_hemi_views).astype(int)])
                for f in files:
                    id = int(f.split('.')[0])
                    if c1 and id in ids or c2 and id >= ks[i] or c3:
                        os.symlink((output_dir / model_name / f).absolute(), sfm_dir / f'{model_name}_{f}')
            run_func = run_colmap if self.sfm_tool == 'colmap' else run_hloc
            run_func(sfm_dir, sfm_dir, CameraModel.OPENCV)

        if self.compute_trans or self.vis:
            images = read_images_binary(sfm_dir / 'sparse/0/images.bin')
            poses_sfm = defaultdict(list)
            for im_data in images.values():
                fname = im_data.name
                r = re.fullmatch(r'(.+)_(\d+).png', fname)
                model_name = r[1]
                id = int(r[2])
                poses_sfm[model_name].append((id, extract_colmap_pose(im_data)))

        if self.compute_trans:
            # sfm-to-norm transforms
            Ts_sfm_norm = {}
            for model_name in model_names:
                n = len(poses_sfm[model_name])
                print(f'Got {n} poses for {model_name} from SfM.')
                if n < 2:
                    continue
                poses_norm = np.load(output_dir / f'poses-{model_name}_norm.npy')
                s_lst = []
                for i in range(n - 1):
                    for j in range(i + 1, n):
                        tAi_norm = poses_norm[poses_sfm[model_name][i][0]][:3, 3]
                        tAj_norm = poses_norm[poses_sfm[model_name][j][0]][:3, 3]
                        tAi_sfm = poses_sfm[model_name][i][1][:3, 3]
                        tAj_sfm = poses_sfm[model_name][j][1][:3, 3]
                        s_lst.append(np.linalg.norm(tAi_norm - tAj_norm) / np.linalg.norm(tAi_sfm - tAj_sfm))
                s = np.median(s_lst)
                T = avg_trans([poses_norm[id] @ np.diag((s, s, s, 1)).astype(np.float32) @ np.linalg.inv(pose_sfm) for id, pose_sfm in poses_sfm[model_name]], s=s, avg_func=np.median)
                Ts_sfm_norm[model_name] = T
                np.save(output_dir / f'T~{cfg}-{model_name}_norm.npy', T)
            if not Ts_sfm_norm:
                print(f'failed to recover any transform')
                exit()
            T_invs_sfm_norm = {model_name: np.linalg.inv(T_sfm_norm) for model_name, T_sfm_norm in Ts_sfm_norm.items()}
            if self.model_gt_trans:
                # mean world-to-sfm transform
                T_world_sfm = avg_trans([T_invs_sfm_norm[model_name] @ Ts_gt_world_norm[i] for i, model_name in enumerate(Ts_sfm_norm)])
                T_inv_world_sfm = np.linalg.inv(T_world_sfm)
                np.save(output_dir / f'T~{cfg}.npy', T_world_sfm)
            for i, model_name in enumerate(model_names):
                if model_name not in Ts_sfm_norm:
                    print(f'failed to recover {model_name}_norm-to-world transform')
                    continue
                T_pred = T_inv_world_sfm @ T_invs_sfm_norm[model_name]
                print(f'\n{model_name}_norm-to-world:')
                print('pred transform\n', T_pred)
                if self.model_gt_trans:
                    T_gt = T_invs_gt_world_norm[i]
                    print('gt transform\n', T_gt)
                    r, t, s = compute_trans_diff(T_gt, T_pred)
                    print(f'rotation error {r:.3g}')
                    print(f'translation error {t:.3g}')
                    print(f'scale error {s:.3g}')

        # if self.vis:
        #     T_world_sfm = np.load(output_dir / f'T~{cfg}.npy')
        #     T_inv_world_sfm = np.linalg.inv(T_world_sfm)
        #     _, s = decompose_sim3(T_world_sfm)
        #     S_world_sfm = np.diag((s, s, s, 1)).astype(np.float32)
        #     Ts_norm_sfm = {}
        #     for model_name in model_names:
        #         T_sfm_norm_path = output_dir / f'T~{cfg}-{model_name}_norm.npy'
        #         if T_sfm_norm_path.exists():
        #             Ts_norm_sfm[model_name] = np.linalg.inv(np.load(T_sfm_norm_path))
        #     colors = cycle(plt.cm.tab20.colors)
        #     vis = Visualizer(show_frame=True)
        #     vis.add_trajectory([T_inv_world_sfm @ Ts_norm_sfm[model_name] for model_name in Ts_norm_sfm], pose_spec=0, cam_size=0.3, color=next(colors))
        #     if self.model_gt_trans:
        #         vis.add_trajectory(T_invs_gt_world_norm, pose_spec=0, cam_size=0.28, color=next(colors))
        #     for i, model_name in enumerate(model_names):
        #         vis.add_trajectory(T_inv_world_sfm @ np.stack([pose_sfm[1] for pose_sfm in poses_sfm[model_name]]) @ S_world_sfm, cam_size=0.3, color=next(colors))
        #         if self.model_gt_trans:
        #             poses_norm = np.load(output_dir / f'poses-{model_name}_norm.npy')
        #             vis.add_trajectory(T_invs_gt_world_norm[i] @ poses_norm @ Ss_gt_world_norm[i], cam_size=0.28, color=next(colors))
        #     vis.show()
