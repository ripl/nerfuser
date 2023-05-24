import json
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from itertools import cycle
from pathlib import Path
from typing import Literal, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import tyro
from nerfstudio.process_data.colmap_utils import CameraModel, read_images_binary, run_colmap
from nerfstudio.process_data.hloc_utils import run_hloc

from nerfuser.utils.utils import avg_trans, complete_trans, compute_trans_diff, decompose_sim3, extract_colmap_pose, gen_hemispheric_poses
from nerfuser.utils.visualizer import Visualizer
from nerfuser.view_renderer import ViewRenderer


@dataclass
class Registration:
    """Register multiple NeRF models to a common coordinate system."""

    model_dirs: list[Path]
    """model checkpoint directories"""
    output_dir: Path = Path('outputs/registration')
    """output directory"""
    name: Optional[str] = None
    """if present, will continue with the existing named experiment"""
    model_method: Literal['nerfacto'] = 'nerfacto'
    """model method"""
    model_names: Optional[list[str]] = None
    """names of models to register"""
    model_gt_trans: Optional[Path] = None
    """path to npy containing ground-truth transforms from the common world coordinate system to each model's local one; can be "identity" """
    step: Optional[int] = None
    """model step to load"""
    cam_info: Union[Path, list[float]] = field(default_factory=lambda: [400.0, 400.0, 400.0, 300.0, 800, 600])
    """either path to json or cam params (fx fy cx cy w h)"""
    downscale_factor: Optional[float] = None
    """downscale factor for NeRF rendering"""
    training_poses: Optional[list[Path]] = None
    """paths to json containing training poses; if present, will be used to render training views and to determine the number of hemispheric poses"""
    n_hemi_poses: int = 30
    """number of hemispheric poses; only applicable when training-poses is not present"""
    render_hemi_views: bool = False
    """use 1.3x hemispheric poses for rendering"""
    chunk_size: Optional[int] = None
    """number of rays to process at a time"""
    fps: Optional[int] = None
    """if present, will use this frame rate for video output"""
    sfm_tool: Literal['hloc', 'colmap'] = 'hloc'
    """SfM tool to use"""
    sfm_w_training_views: bool = True
    """when render-hemi-views, set this to False to only use hemispheric views for SfM"""
    sfm_w_hemi_views: float = 1
    """ratio of #hemi-views vs. #training-views or n-hemi-poses for SfM, within range [0, 1.3]"""
    device: str = 'cuda:0'
    """device to use"""
    render_views: bool = False
    """whether to render views"""
    run_sfm: bool = False
    """whether to run SfM"""
    compute_trans: bool = False
    """whether to compute transforms"""
    vis: bool = False
    """whether to visualize the registration"""

    def main(self):
        if not self.name:
            self.name = datetime.now().strftime('%m.%d_%H:%M:%S')
        output_dir = self.output_dir / self.name
        output_dir.mkdir(parents=True, exist_ok=True)
        n_models = len(self.model_dirs)
        if not self.model_names:
            self.model_names = [f'nerf{i}' for i in range(n_models)]
        if self.render_hemi_views:
            cfg = f'hemi{self.sfm_w_hemi_views:.2f}'
            if self.training_poses:
                cfg += f'_train{int(self.sfm_w_training_views)}'
        else:
            cfg = 'train' if self.training_poses else 'hemi'
        sfm_dir = output_dir / f'{self.sfm_tool}~{cfg}'
        log_dict = {}
        for attr in ('model_dirs', 'model_method', 'model_gt_trans', 'step', 'cam_info', 'downscale_factor', 'training_poses', 'n_hemi_poses', 'sfm_tool'):
            val = getattr(self, attr)
            if attr in {'model_dirs', 'training_poses'}:
                if val:
                    val = dict(zip(self.model_names, [str(item) for item in val]))
            elif isinstance(val, Path):
                val = str(val)
            log_dict[attr] = val
        with (output_dir / f'{cfg}.json').open(mode='w') as f:
            json.dump(log_dict, f, indent=2)

        # nerf-to-nerf_norm transforms
        Ts_nerf_norm = []
        Ss_norm_nerf = []
        for model_dir in self.model_dirs:
            with (model_dir.parent / 'dataparser_transforms.json').open() as f:
                transforms = json.load(f)
            s = transforms['scale']
            S_nerf_norm = np.diag((s, s, s, 1)).astype(np.float32)
            Ss_norm_nerf.append(np.linalg.inv(S_nerf_norm))
            Ts_nerf_norm.append(S_nerf_norm @ complete_trans(np.array(transforms['transform'], dtype=np.float32)))
        Ts_nerf_norm = np.array(Ts_nerf_norm)
        Ss_norm_nerf = np.array(Ss_norm_nerf)
        if self.model_gt_trans:
            # gt world-to-nerf transforms
            Ts_gt_world_nerf = np.broadcast_to(np.identity(4, dtype=np.float32), (n_models, 4, 4)) if str(self.model_gt_trans).lower() in {'i', 'identity'} else np.load(self.model_gt_trans)
            # gt world-to-nerf_norm transforms
            Ts_gt_world_norm = Ts_nerf_norm @ Ts_gt_world_nerf
            Ts_gt_norm_world = np.linalg.inv(Ts_gt_world_norm)
            _, s = decompose_sim3(Ts_gt_world_norm)
            Ss_gt_world_norm = np.zeros((n_models, 4, 4))
            for i in range(3):
                Ss_gt_world_norm[:, i, i] = s
            Ss_gt_world_norm[:, 3, 3] = 1
        if self.training_poses:
            frames = []
            ls = []
            for training_pose in self.training_poses:
                with training_pose.open() as f:
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
            if isinstance(self.cam_info, Path):
                with self.cam_info.open() as f:
                    transforms = json.load(f)
                cam_info = {'fx': transforms['fl_x'], 'fy': transforms['fl_y'], 'cx': transforms['cx'], 'cy': transforms['cy'], 'width': transforms['w'], 'height': transforms['h'], 'distortion_params': np.array((transforms['k1'], transforms['k2'], 0, 0, transforms['p1'], transforms['p2']), dtype=np.float32)}
            else:
                cam_info = dict(zip(('fx', 'fy', 'cx', 'cy', 'width', 'height'), self.cam_info))
            if self.downscale_factor:
                for pname in cam_info:
                    if pname != 'distortion_params':
                        cam_info[pname] /= self.downscale_factor
            cam_info['height'] = int(cam_info['height'])
            cam_info['width'] = int(cam_info['width'])
            for i in range(n_models):
                poses_norm = complete_trans(np.array(gen_hemispheric_poses(1, np.pi / 6, m=ms[i], n=ns[i])))[np.sort(rng.permutation(ms[i] * ns[i])[:ks[i]])] if not self.training_poses or self.render_hemi_views else np.empty((0, 4, 4), dtype=np.float32)
                if self.training_poses:
                    pose_dict = {frame['file_path']: np.array(frame['transform_matrix'], dtype=np.float32) for frame in frames[i]}
                    poses_norm = np.concatenate((poses_norm, Ts_nerf_norm[i] @ np.array([pose_dict[k] for k in sorted(pose_dict.keys())]) @ Ss_norm_nerf[i]))
                with torch.no_grad():
                    ViewRenderer(self.model_method, self.model_names[i], self.model_dirs[i], load_step=self.step, chunk_size=self.chunk_size, device=self.device).render_views(poses_norm, cam_info, output_dir, animate=self.fps)
                np.save(output_dir / f'poses~{self.model_names[i]}_norm.npy', poses_norm)

        if self.run_sfm:
            shutil.rmtree(sfm_dir, ignore_errors=True)
            sfm_dir.mkdir(parents=True)
            # uses hemi views, maybe training views
            c1 = not self.training_poses or self.render_hemi_views
            # uses hemi + training views
            c2 = self.training_poses and self.render_hemi_views and self.sfm_w_training_views
            # uses training views only
            c3 = self.training_poses and not self.render_hemi_views
            for i, model_name in enumerate(self.model_names):
                if c1:
                    ids = set(rng.permutation(ks[i])[:np.floor(ls[i] * self.sfm_w_hemi_views).astype(int)])
                for f in (output_dir / model_name).iterdir():
                    id = int(f.stem)
                    if c1 and id in ids or c2 and id >= ks[i] or c3:
                        (sfm_dir / f'{model_name}_{f.name}').symlink_to(f.absolute())
            run_func = run_hloc if self.sfm_tool == 'hloc' else run_colmap
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
            # sfm-to-nerf_norm transforms
            Ts_sfm_norm = {}
            for model_name in self.model_names:
                n = len(poses_sfm[model_name])
                print(f'Got {n} poses for {model_name} from SfM.')
                T_sfm_norm_path = output_dir / f'T~{cfg}~{model_name}_norm.npy'
                if n < 2:
                    T_sfm_norm_path.unlink(missing_ok=True)
                    continue
                poses_norm = np.load(output_dir / f'poses~{model_name}_norm.npy')
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
                np.save(T_sfm_norm_path, T)
            if not Ts_sfm_norm:
                print(f'failed to recover any transform')
                exit()
            Ts_norm_sfm = {model_name: np.linalg.inv(T_sfm_norm) for model_name, T_sfm_norm in Ts_sfm_norm.items()}
            if self.model_gt_trans:
                # mean world-to-sfm transform
                T_world_sfm = avg_trans([Ts_norm_sfm[model_name] @ Ts_gt_world_norm[i] for i, model_name in enumerate(Ts_sfm_norm)])
                T_sfm_world = np.linalg.inv(T_world_sfm)
                np.save(output_dir / f'T~{cfg}.npy', T_world_sfm)
            else:
                T_sfm_world = np.identity(4, dtype=np.float32)
            for i, model_name in enumerate(self.model_names):
                if model_name not in Ts_sfm_norm:
                    print(f'failed to recover {model_name}_norm-to-world transform')
                    continue
                T_pred = T_sfm_world @ Ts_norm_sfm[model_name]
                print(f'\n{model_name}_norm-to-world:')
                print('pred transform\n', T_pred)
                if self.model_gt_trans:
                    T_gt = Ts_gt_norm_world[i]
                    print('gt transform\n', T_gt)
                    r, t, s = compute_trans_diff(T_gt, T_pred)
                    print(f'rotation error {r:.3g}')
                    print(f'translation error {t:.3g}')
                    print(f'scale error {s:.3g}')

        if self.vis:
            T_world_sfm = np.load(output_dir / f'T~{cfg}.npy') if self.model_gt_trans else np.identity(4, dtype=np.float32)
            T_sfm_world = np.linalg.inv(T_world_sfm)
            _, s = decompose_sim3(T_world_sfm)
            S_world_sfm = np.diag((s, s, s, 1)).astype(np.float32)
            Ts_norm_sfm = {}
            for model_name in self.model_names:
                T_sfm_norm_path = output_dir / f'T~{cfg}~{model_name}_norm.npy'
                if T_sfm_norm_path.exists():
                    Ts_norm_sfm[model_name] = np.linalg.inv(np.load(T_sfm_norm_path))
            colors = cycle(plt.cm.tab20.colors if self.model_gt_trans else plt.cm.tab10.colors)
            vis = Visualizer(show_frame=True)
            vis.add_trajectory([T_sfm_world @ Ts_norm_sfm[model_name] for model_name in Ts_norm_sfm], pose_spec=0, cam_size=0.3, color=next(colors))
            color = next(colors)
            if self.model_gt_trans:
                vis.add_trajectory(Ts_gt_norm_world, pose_spec=0, cam_size=0.28, color=color)
            for i, model_name in enumerate(self.model_names):
                color = next(colors)
                if len(poses_sfm[model_name]):
                    vis.add_trajectory(T_sfm_world @ np.array([pose_sfm[1] for pose_sfm in poses_sfm[model_name]]) @ S_world_sfm, cam_size=0.3, color=color)
                color = next(colors)
                if self.model_gt_trans:
                    poses_norm = np.load(output_dir / f'poses~{model_name}_norm.npy')
                    vis.add_trajectory(Ts_gt_norm_world[i] @ poses_norm @ Ss_gt_world_norm[i], cam_size=0.28, color=color)
            vis.show()


if __name__ == '__main__':
    tyro.cli(Registration).main()
