import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Literal, Optional, Union

import cv2
import imageio
import numpy as np
import torch
import tyro
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from tqdm import tqdm

from nerfuser.utils.utils import complete_transform, gen_circular_poses
from nerfuser.view_blender import ViewBlender


@dataclass
class Blending:
    """Blend multiple registered NeRF models to render novel views."""

    model_dirs: List[Path]
    """model checkpoint directories"""
    output_dir: Path = Path('outputs/blending')
    """output directory"""
    name: Optional[str] = None
    """if present, will continue with the existing named experiment"""
    model_method: Literal['nerfacto'] = 'nerfacto'
    """model method"""
    model_names: Optional[List[str]] = None
    """names of models to blend"""
    model_gt_trans: Optional[str] = None
    """path to npy containing ground-truth transforms from the common world coordinate system to each model's local one; can be "identity"; only applicable when trans-src is "gt" """
    step: Optional[int] = None
    """model step to load"""
    cam_info: Union[str, List[float]] = field(default_factory=lambda: [400, 400, 400, 300, 800, 600])
    """either path to json or cam params (fx fy cx cy w h)"""
    downscale_factor: Optional[float] = None
    """downscale factor for NeRF rendering"""
    test_poses: Optional[str] = None
    """path to json containing test poses; will use circular poses if not specified"""
    test_frame: Literal['sfm', 'world'] = 'sfm'
    """the coordinate system in which test-poses are defined"""
    chunk_size: Optional[int] = None
    """number of rays to process at a time"""
    reg_dir: Path = Path('outputs/registration')
    """directory containing registration results"""
    reg_name: Optional[str] = None
    """will load transforms from the named registration; defaults to the current name"""
    trans_src: str = 'hemi'
    """source of sfm to normalized nerf transforms; if "gt", will use "model-gt-trans" and test-frame must be "world" """
    blend_methods: List[Literal['nearest', 'idw2', 'idw3', 'idw4']] = field(default_factory=lambda: ['idw4'])
    """blending methods"""
    tau: float = 3
    """maximum blending distance ratio; must be larger than 1"""
    gammas: List[float] = field(default_factory=lambda: [3])
    """blending rates for all applicable methods"""
    fps: int = 8
    """frame rate for video output"""
    save_extras: bool = False
    """whether to save extra outputs (raw renderings, weight maps)"""
    device: str = 'cuda:0'
    """device to use"""
    blend_views: bool = False
    """whether to blend views"""
    evaluate: bool = False
    """whether to evaluate the blending results"""

    def main(self):
        if not self.name:
            self.name = datetime.now().strftime('%m.%d_%H:%M:%S')
        output_dir = Path(self.output_dir) / self.name / self.trans_src
        os.makedirs(output_dir, exist_ok=True)
        n_models = len(self.model_dirs)
        if not self.model_names:
            self.model_names = [f'nerf{i}' for i in range(n_models)]
        log_dict = {attr: dict(zip(self.model_names, [str(model_dir) for model_dir in self.model_dirs])) if attr == 'model_dirs' else getattr(self, attr) for attr in ['model_dirs', 'model_method', 'model_gt_trans', 'step', 'cam_info', 'downscale_factor', 'test_poses', 'blend_methods', 'tau', 'gammas', 'fps']}
        with open(output_dir / 'config.json', 'w') as f:
            json.dump(log_dict, f, indent=2)

        multi_cam = False
        if isinstance(self.cam_info, str):
            with open(self.cam_info) as f:
                transforms = json.load(f)
            if 'fl_x' in transforms:
                cam_info = {'fx': transforms['fl_x'], 'fy': transforms['fl_y'], 'cx': transforms['cx'], 'cy': transforms['cy'], 'width': transforms['w'], 'height': transforms['h'], 'distortion_params': np.array([transforms['k1'], transforms['k2'], 0, 0, transforms['p1'], transforms['p2']], dtype=np.float32)}
            else:
                intrinsics_dict = {frame['file_path']: {'fx': frame['fl_x'], 'fy': frame['fl_y'], 'cx': frame['cx'], 'cy': frame['cy'], 'width': frame['w'], 'height': frame['h'], 'distortion_params': [frame['k1'], frame['k2'], 0, 0, frame['p1'], frame['p2']]} for frame in transforms['frames']}
                keys = sorted(intrinsics_dict.keys())
                cam_info = {param: np.array([intrinsics_dict[k][param] for k in keys], dtype=np.float32) for param in ['fx', 'fy', 'cx', 'cy', 'width', 'height', 'distortion_params']}
                multi_cam = True
        else:
            cam_info = dict(zip(['fx', 'fy', 'cx', 'cy', 'width', 'height'], self.cam_info))
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
        blend_methods = {}
        for blend_method in self.blend_methods:
            if blend_method == 'nearest':
                blend_methods[blend_method] = (blend_method, None)
            else:
                for g in self.gammas:
                    blend_methods[f'{blend_method}_g{g:.2g}'] = (blend_method, g)

        if self.blend_views:
            if self.test_poses:
                with open(self.test_poses) as f:
                    transforms = json.load(f)
                pose_dict = {frame['file_path']: np.array(frame['transform_matrix'], dtype=np.float32) for frame in transforms['frames']}
                poses = np.array([pose_dict[k] for k in sorted(pose_dict.keys())])
            else:
                poses = np.array(gen_circular_poses(1, 0, n=60))
            Ts = []
            if self.trans_src == 'gt':
                assert self.test_frame == 'world', 'test poses must be specified in world coordinates to utilize ground-truth world-to-nerf transforms'
                assert self.model_gt_trans, 'ground-truth world-to-nerf transforms must be specified'
                Ts_world_nerf = np.broadcast_to(np.identity(4, dtype=np.float32)[None], (n_models, 4, 4)) if self.model_gt_trans.lower() in {'i', 'identity'} else np.load(self.model_gt_trans)
                for i, model_dir in enumerate(self.model_dirs):
                    with open(model_dir.parent / 'dataparser_transforms.json') as f:
                        transforms = json.load(f)
                    s = transforms['scale']
                    Ts.append(np.diag((s, s, s, 1)).astype(np.float32) @ complete_transform(np.array(transforms['transform'], dtype=np.float32)) @ Ts_world_nerf[i])
            else:
                if not self.reg_name:
                    self.reg_name = self.name
                reg_dir = Path(self.reg_dir) / self.reg_name
                valid_ids = []
                if self.test_frame == 'world':
                    T_sfm_path = reg_dir / f'T~{self.trans_src}.npy'
                    if not T_sfm_path.exists():
                        raise FileNotFoundError(f'{T_sfm_path}. Please run registration first.')
                    T_sfm = np.load(T_sfm_path)
                else:
                    T_sfm = np.identity(4, dtype=np.float32)
                for i, model_name in enumerate(self.model_names):
                    T_path = reg_dir / f'T~{self.trans_src}-{model_name}_norm.npy'
                    if T_path.exists():
                        Ts.append(np.load(T_path) @ T_sfm)
                        valid_ids.append(i)
                    else:
                        print(f'sfm-to-{model_name}_norm transform not found. Skipping.')
                self.model_names = [self.model_names[i] for i in valid_ids]
                self.model_dirs = [self.model_dirs[i] for i in valid_ids]
            with torch.no_grad():
                ViewBlender(self.model_method, self.model_names, self.model_dirs, np.stack(Ts), self.tau, load_step=self.step, chunk_size=self.chunk_size, device=self.device).blend_views(poses, cam_info, output_dir, blend_methods, multi_cam=multi_cam, save_extras=self.save_extras, animate=self.fps)

        if self.evaluate:
            assert self.test_poses, 'must provide test_poses json to evaluate'
            with open(self.test_poses) as f:
                transforms = json.load(f)
            filepaths = sorted(frame['file_path'] for frame in transforms['frames'])
            methods = [model_name for model_name in self.model_names if self.save_extras and (output_dir / model_name).exists()] + list(blend_methods.keys())
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
                gt = imageio.v3.imread(Path(self.test_poses).parent / filepath) / np.float32(255)
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


if __name__ == '__main__':
    tyro.cli(Blending).main()
