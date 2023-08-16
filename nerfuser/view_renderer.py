import shutil
from types import MethodType

import imageio
import numpy as np
import torch
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.models.nerfacto import NerfactoModelConfig
from tqdm import trange

from nerfuser.components import get_nerfacto_outputs
from nerfuser.utils.utils import complete_trans, decompose_sim3


class ViewRenderer:
    def __init__(self, model_method, model_name, load_dir, transform=None, load_step=None, chunk_size=None, device='cuda') -> None:
        """
        Args:
            transforms: an array of transforms, each of which transforms pts from the common coordinate system to a local normalized NeRF one
            use_global_metric: whether to use global metric for measuring distances
        """
        self.model_method = model_method
        self.model_name = model_name
        if load_step is None or not (load_path := load_dir / f'step-{load_step:09d}.ckpt').exists():
            # load the latest checkpoint
            load_path = max(load_dir.iterdir())
            load_step = int(load_path.stem[5:])
        state = torch.load(load_path, map_location=device)
        state = {key[7:]: val for key, val in state['pipeline'].items() if key.startswith('_model.')}
        if model_method == 'nerfacto':
            self.model = NerfactoModelConfig().setup(scene_box=SceneBox(aabb=state['field.aabb']), num_train_data=len(state['field.embedding_appearance.embedding.weight'])).to(device)
            self.model.get_outputs = MethodType(get_nerfacto_outputs, self.model)
            if not chunk_size:
                chunk_size = 1 << 16
            self.model.update_to_step(load_step)
            self.model.load_state_dict(state)
            self.model.eval()
            print(f'loaded checkpoint from {load_path}')
        if transform is None:
            transform = torch.eye(4)
        self.transform = complete_trans(torch.as_tensor(transform, device=device))
        _, s = decompose_sim3(self.transform)
        S = torch.diag(torch.tensor((s, s, s, 1), device=device))
        self.S_inv = torch.linalg.inv(S)
        self.chunk_size = chunk_size
        self.device = device

    def render_views(self, c2ws, cam_info, output_dir, filter_poses_acc_dist=None, multi_cam=False, save_extras=False, animate=0):
        """
        Args:
            c2ws: an array of camera-to-world transforms, where world is the common coordinate system as in self.transforms
        """
        shutil.rmtree(output_dir / self.model_name / 'imgs', ignore_errors=True)
        (output_dir / self.model_name / 'imgs').mkdir(parents=True)
        # as of now, save_extras only affects saving dist_accs
        save_extras = save_extras and filter_poses_acc_dist is not None
        if save_extras:
            shutil.rmtree(output_dir / self.model_name / f'dist_accs_{filter_poses_acc_dist:.2f}', ignore_errors=True)
            (output_dir / self.model_name / f'dist_accs_{filter_poses_acc_dist:.2f}').mkdir(parents=True)
        c2ws = complete_trans(torch.as_tensor(c2ws, device=self.device))
        for p in cam_info:
            if isinstance(cam_info[p], np.ndarray):
                cam_info[p] = torch.from_numpy(cam_info[p]).to(self.device)
        # poses in nerf_norm
        poses = self.transform @ c2ws @ self.S_inv
        if multi_cam:
            hs, ws = cam_info['height'], cam_info['width']
            n_rays_lst = hs * ws
        else:
            h, w = cam_info['height'], cam_info['width']
            n_rays = h * w
        if animate:
            imgs = []
        if filter_poses_acc_dist is not None:
            dist_accs = []
        for i in trange(len(poses)):
            if multi_cam:
                h, w = hs[i], ws[i]
                n_rays = n_rays_lst[i]
                cam = Cameras(poses[i, :3].cpu(), **{p: cam_info[p][i] for p in cam_info}).to(self.device)
            else:
                cam = Cameras(poses[i, :3].cpu(), **cam_info).to(self.device)
            rgb_chunks = []
            if filter_poses_acc_dist is not None:
                dist_acc_chunks = []
            cam_ray_bundle = cam.generate_rays(0)
            for k in trange(0, n_rays, self.chunk_size, leave=False):
                start_idx = k
                end_idx = k + self.chunk_size
                ray_bundle = cam_ray_bundle.get_row_major_sliced_ray_bundle(start_idx, end_idx)
                output = self.model(ray_bundle)
                rgb_chunks.append(output['rgb'].cpu())
                if filter_poses_acc_dist is not None:
                    dist_acc_chunks.append(self.compute_distant_accumulation(output['weights'], output['deltas'], d=filter_poses_acc_dist))
            img = (torch.cat(rgb_chunks).view(h, w, -1) * 255).to(torch.uint8).numpy()
            if filter_poses_acc_dist is not None:
                dist_acc = torch.cat(dist_acc_chunks)
                dist_accs.append(dist_acc.mean().item())
            imageio.v3.imwrite(output_dir / self.model_name / f'imgs/{i:04d}.png', img)
            if save_extras:
                dist_acc = (dist_acc.view(h, w, 1) * 255).to(torch.uint8).broadcast_to(-1, -1, 3).numpy()
                imageio.v3.imwrite(output_dir / self.model_name / f'dist_accs_{filter_poses_acc_dist:.2f}/{i:04d}_{dist_accs[-1]:.3f}.png', dist_acc)
            if animate:
                imgs.append(img)
        if filter_poses_acc_dist is not None:
            np.save(output_dir / self.model_name / f'dist_accs_{filter_poses_acc_dist:.2f}.npy', np.array(dist_accs))
        if animate:
            imageio.v3.imwrite(output_dir / f'{self.model_name}.mp4', imgs, fps=animate, quality=10)

    @staticmethod
    def compute_distant_accumulation(weights, deltas, d=0):
        # computes the accumulation of weights starting at distance d
        # weights: (n_rays, n_samples, 1)
        # deltas: (n_rays, n_samples, 1)
        # d: the distance threshold to the camera
        # return: (n_rays, 1)
        accs = torch.cumsum(weights, dim=1)  # (n_rays, n_samples, 1)
        dists = torch.cumsum(deltas, dim=1)  # (n_rays, n_samples, 1)
        indices = torch.searchsorted(dists[..., 0], torch.full((len(dists), 1), d, device=dists.device), right=True)  # (n_rays, 1)
        dist_acc = torch.gather(weights[..., 0], 1, indices) * (torch.gather(dists[..., 0], 1, indices) - d) / torch.gather(deltas[..., 0], 1, indices) + accs[:, -1] - torch.gather(accs[..., 0], 1, indices)
        return dist_acc.cpu()
