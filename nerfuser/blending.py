import argparse
import json
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import imageio
import numpy as np
import torch
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from tqdm import tqdm

from nerfuser.utils.utils import complete_transform, gen_circular_poses
from nerfuser.view_blender import ViewBlender

class Blending:
    def __init__(
        self,
        name,
        dataset_dir,
        model_method,
        model_names,
        model_gt_trans,
        model_dirs,
        step,
        cam_info,
        downscale_factor,
        test_poses,
        test_frame,
        reg_dir,
        reg_name,
        trans_src,
        blend_methods,
        tau,
        gammas,
        fps,
        save_extras,
        output_dir,
        device,
        blend_views,
        evaluate,
    ):
        self.name = name
        self.dataset_dir = dataset_dir
        self.model_method = model_method
        self.model_names = model_names
        self.model_gt_trans = model_gt_trans
        self.model_dirs = model_dirs
        self.step = step
        self.cam_info = cam_info
        self.downscale_factor = downscale_factor
        self.test_poses = test_poses
        self.test_frame = test_frame
        self.reg_dir = reg_dir
        self.reg_name = reg_name
        self.trans_src = trans_src
        self.blend_methods = blend_methods
        self.tau = tau
        self.gammas = gammas
        self.fps = fps
        self.save_extras = save_extras
        self.output_dir = output_dir
        self.device = device
        self.blend_views = blend_views
        self.evaluate = evaluate

    def run(self):
        if self.name:
            name = self.name
        else:
            name = datetime.now().strftime('%m.%d_%H:%M:%S')
        output_dir = Path(self.output_dir) / name / self.trans_src
        os.makedirs(output_dir, exist_ok=True)
        n_models = len(self.model_dirs)
        model_names = self.model_names if self.model_names else [f'nerf{i}' for i in range(n_models)]
        # log_dict = {k: getattr(self, k) for k in ['dataset_dir', 'model_method', 'step', 'downscale_factor', 'test_poses', 'blend_methods', 'tau', 'gammas', 'fps']}
        log_dict = {}
        for k in ['dataset_dir', 'model_method', 'step', 'downscale_factor', 'test_poses', 'blend_methods', 'tau', 'gammas', 'fps']:
            attr = getattr(self, k)
            if isinstance(attr, Path):
                log_dict[k] = str(attr)
        log_dict['cam_info'] = [(str(cam_info) if isinstance(cam_info, Path) else cam_info) for cam_info in self.cam_info]
        log_dict['model_dirs'] = {model_name: str(model_dir) for model_name, model_dir in zip(model_names, self.model_dirs)}
        print('log_dict', log_dict)
        with open(output_dir / 'config.json', 'w') as f:
            json.dump(log_dict, f, indent=2)

        dataset_dir = Path(self.dataset_dir) if self.dataset_dir else None
        multi_cam = False
        if len(self.cam_info) == 1:
            cam_info_path = self.cam_info[0]
            # if dataset_dir:
            #     cam_info_path = dataset_dir / cam_info_path
            with open(cam_info_path) as f:
                transforms = json.load(f)
            if 'fl_x' in transforms:
                cam_info = {'fx': transforms['fl_x'], 'fy': transforms['fl_y'], 'cx': transforms['cx'], 'cy': transforms['cy'], 'width': transforms['w'], 'height': transforms['h'], 'distortion_params': np.array([transforms['k1'], transforms['k2'], 0, 0, transforms['p1'], transforms['p2']], dtype=np.float32)}
            else:
                intrinsics_dict = {frame['file_path']: {'fx': frame['fl_x'], 'fy': frame['fl_y'], 'cx': frame['cx'], 'cy': frame['cy'], 'width': frame['w'], 'height': frame['h'], 'distortion_params': [frame['k1'], frame['k2'], 0, 0, frame['p1'], frame['p2']]} for frame in transforms['frames']}
                keys = sorted(intrinsics_dict.keys())
                cam_info = {param: np.array([intrinsics_dict[k][param] for k in keys], dtype=np.float32) for param in ['fx', 'fy', 'cx', 'cy', 'width', 'height', 'distortion_params']}
                multi_cam = True
        else:
            cam_info = dict(zip(['fx', 'fy', 'cx', 'cy', 'width', 'height'], [float(v) for v in self.cam_info]))
        if self.downscale_factor:
            for pname in cam_info:
                if pname != 'distortion_params':
                    cam_info[pname] /= self.downscale_factor
        if multi_cam:
            cam_info['height'] = cam_info['height'].astype(int)
            cam_info['width'] = cam_info['width'].astype(int)
            self.fps = 0
        else:
            cam_info['height'] = int(cam_info['height'])
            cam_info['width'] = int(cam_info['width'])
        gs = [float(g) for g in self.gammas] if self.gammas else [3]
        blend_methods = {}
        for blend_method in self.blend_methods:
            if blend_method == 'nearest':
                blend_methods[blend_method] = (blend_method, None)
            else:
                for g in gs:
                    blend_methods[f'{blend_method}_g{g:.2g}'] = (blend_method, g)

        if self.blend_views:
            model_dirs = [Path(model_dir) for model_dir in self.model_dirs]
            if self.test_poses:
                # test_poses = dataset_dir / self.test_poses if dataset_dir else Path(self.test_poses)
                test_poses = Path(self.test_poses)
                with open(test_poses) as f:
                    transforms = json.load(f)
                pose_dict = {frame['file_path']: np.array(frame['transform_matrix'], dtype=np.float32) for frame in transforms['frames']}
                poses = np.array([pose_dict[k] for k in sorted(pose_dict.keys())])
            else:
                poses = np.array(gen_circular_poses(1, 0, n=60))
            Ts = []
            if self.trans_src == 'gt':
                assert self.test_frame == 'world'
                Ts_world_nerf = np.broadcast_to(np.identity(4, dtype=np.float32)[None], (n_models, 4, 4)) if self.model_gt_trans.lower() in {'i', 'identity'} else np.load(self.model_gt_trans)
                for i, model_dir in enumerate(model_dirs):
                    with open(model_dir.parent / 'dataparser_transforms.json') as f:
                        transforms = json.load(f)
                    s = transforms['scale']
                    Ts.append(np.diag((s, s, s, 1)).astype(np.float32) @ complete_transform(np.array(transforms['transform'], dtype=np.float32)) @ Ts_world_nerf[i])
            else:
                reg_name = self.reg_name if self.reg_name else name
                reg_dir = Path(self.reg_dir) / reg_name
                valid_ids = []
                if self.test_frame == 'world':
                    T_sfm_path = reg_dir / f'T~{self.trans_src}.npy'
                    if not T_sfm_path.exists():
                        raise FileNotFoundError(f'{T_sfm_path}. Please run registration first.')
                    T_sfm = np.load(T_sfm_path)
                else:
                    T_sfm = np.identity(4, dtype=np.float32)
                for i, model_name in enumerate(model_names):
                    T_path = reg_dir / f'T~{self.trans_src}-{model_name}_norm.npy'
                    if T_path.exists():
                        Ts.append(np.load(T_path) @ T_sfm)
                        valid_ids.append(i)
                    else:
                        print(f'sfm-to-{model_name}_norm transform not found. Skipping.')
                model_names = [model_names[i] for i in valid_ids]
                model_dirs = [model_dirs[i] for i in valid_ids]
            with torch.no_grad():
                print('blend_methods: ', blend_methods)
                ViewBlender(
                    self.model_method, 
                    model_names, 
                    model_dirs, 
                    np.stack(Ts), 
                    self.tau, 
                    load_step=self.step, 
                    device=self.device
                ).blend_views(
                    poses, 
                    cam_info, 
                    output_dir, 
                    blend_methods, 
                    multi_cam=multi_cam, 
                    save_extras=self.save_extras, 
                    animate=self.fps
                )

        if self.evaluate:
            assert self.test_poses, 'must provide test_poses json to evaluate'
            # test_poses = dataset_dir / self.test_poses if dataset_dir else Path(self.test_poses)
            test_poses = Path(self.test_poses)
            with open(test_poses) as f:
                transforms = json.load(f)
            filepaths = sorted(frame['file_path'] for frame in transforms['frames'])
            methods = [model_name for model_name in model_names if self.save_extras and (output_dir / model_name).exists()] + list(blend_methods.keys())
            metrics = {
                'psnr': PeakSignalNoiseRatio(data_range=1).to(self.device),
                'ssim': StructuralSimilarityIndexMeasure(data_range=1).to(self.device),
                'lpips': LearnedPerceptualImagePatchSimilarity(normalize=True).to(self.device),
            }
            if not multi_cam:
                h, w = cam_info['height'], cam_info['width']
            results = defaultdict(lambda: defaultdict(list))
            for i, filepath in enumerate(tqdm(filepaths)):
                if multi_cam:
                    h, w = cam_info['height'][i], cam_info['width'][i]
                gt = imageio.v3.imread(test_poses.parent / filepath) / np.float32(255)
                if self.downscale_factor:
                    gt = cv2.resize(gt, (w, h), interpolation=cv2.INTER_AREA)
                gt = torch.tensor(gt, device=self.device).permute(2, 0, 1)[None]
                for method in methods:
                    pred = (torch.tensor(imageio.v3.imread(output_dir / method / f'{i:04d}.png'), device=self.device) / 255).permute(2, 0, 1)[None]
                    for j, metric in enumerate(metrics):
                        results[method][metric].append(metrics[metric](pred, gt).item())
            for metric in metrics:
                for method in methods:
                    print(f'{metric} {method}: {np.mean(results[method][metric]):.3f}')
            with open(output_dir / 'eval.json', 'w') as f:
                json.dump(results, f, indent=2)