import shutil
from collections import defaultdict
from types import MethodType

import imageio
import numpy as np
import torch
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.models.nerfacto import NerfactoModelConfig
from tqdm import trange

from nerfuser.components import WeightedRGBRenderer, get_nerfacto_outputs
from nerfuser.utils.utils import complete_trans, decompose_sim3, img_cat


class ViewBlender:
    def __init__(self, model_method, model_names, load_dirs, transforms, tau, load_step=None, use_global_metric=False, chunk_size=None, device='cuda') -> None:
        """
        Args:
            transforms: an array of transforms, each of which transforms pts from the common coordinate system to a local normalized NeRF one
            use_global_metric: whether to use global metric for measuring distances
        """
        self.model_method = model_method
        self.model_names = model_names
        self.models = []
        for load_dir in load_dirs:
            if load_step is None or not (load_path := load_dir / f'step-{load_step:09d}.ckpt').exists():
                # load the latest checkpoint
                load_path = max(load_dir.iterdir())
                load_step = int(load_path.stem[5:])
            state = torch.load(load_path, map_location=device)
            state = {key[7:]: val for key, val in state['pipeline'].items() if key.startswith('_model.')}
            if model_method == 'nerfacto':
                model = NerfactoModelConfig().setup(scene_box=SceneBox(aabb=state['field.aabb']), num_train_data=len(state['field.embedding_appearance.embedding.weight'])).to(device)
                model.get_outputs = MethodType(get_nerfacto_outputs, model)
                if not chunk_size:
                    chunk_size = 1 << 16
            model.update_to_step(load_step)
            model.load_state_dict(state)
            model.eval()
            self.models.append(model)
            print(f'loaded checkpoint from {load_path}')
        self.transforms = complete_trans(torch.as_tensor(transforms, device=device))
        _, s = decompose_sim3(self.transforms)
        S = torch.zeros(len(s), 4, 4, device=device)
        for i in range(3):
            S[:, i, i] = s
        S[:, 3, 3] = 1
        self.S_invs = torch.linalg.inv(S)
        self.use_global_metric = use_global_metric
        self.tau = torch.inf if tau is None else tau
        self.chunk_size = chunk_size
        self.device = device

    def blend_views(self, c2ws, cam_info, output_dir, methods, multi_cam=False, save_extras=False, animate=0):
        """
        Args:
            c2ws: an array of camera-to-world transforms, where world is the common coordinate system as in self.transforms
        """
        for method in methods:
            shutil.rmtree(output_dir / method, ignore_errors=True)
            (output_dir / method).mkdir(parents=True)
        if save_extras:
            for model_name in self.model_names:
                shutil.rmtree(output_dir / model_name, ignore_errors=True)
                (output_dir / model_name).mkdir(parents=True)
        c2ws = complete_trans(torch.as_tensor(c2ws, device=self.device))
        for p in cam_info:
            if isinstance(cam_info[p], np.ndarray):
                cam_info[p] = torch.from_numpy(cam_info[p]).to(self.device)
        # poses in nerf_norm
        poses = self.transforms @ c2ws.unsqueeze(1) @ self.S_invs
        m, n = poses.shape[:2]
        if multi_cam:
            hs, ws = cam_info['height'], cam_info['width']
            n_rays_lst = hs * ws
        else:
            h, w = cam_info['height'], cam_info['width']
            n_rays = h * w
        if animate:
            imgs = defaultdict(list)
        for i in trange(m):
            if multi_cam:
                h, w = hs[i], ws[i]
                n_rays = n_rays_lst[i]
                cams = Cameras(poses[i, :, :3].cpu(), **{p: cam_info[p][i] for p in cam_info}).to(self.device)
            else:
                cams = Cameras(poses[i, :, :3].cpu(), **cam_info).to(self.device)
            dists = torch.linalg.norm(poses[i, :, :3, 3], dim=-1)
            if self.use_global_metric:
                dists *= self.S_invs[:, 0, 0]
            keep_flags = dists / dists.min() < self.tau
            rgb_chunks = defaultdict(list)
            if save_extras:
                models = self.models
                cam_ray_bundles = [cams.generate_rays(j) for j in range(n)]
                ws_chunks = defaultdict(list)
            else:
                models = [self.models[j] for j in range(n) if keep_flags[j]]
                cam_ray_bundles = [cams.generate_rays(j) for j in range(n) if keep_flags[j]]
            for k in trange(0, n_rays, self.chunk_size, leave=False):
                start_idx = k
                end_idx = k + self.chunk_size
                outputs = defaultdict(list)
                for j, model in enumerate(models):
                    cam_ray_bundle = cam_ray_bundles[j]
                    ray_bundle = cam_ray_bundle.get_row_major_sliced_ray_bundle(start_idx, end_idx)
                    output = model(ray_bundle)
                    if not save_extras or keep_flags[j]:
                        for key in output:
                            outputs[key].append(output[key])
                    if save_extras:
                        rgb_chunks[self.model_names[j]].append(output['rgb'].cpu())
                for key in outputs:
                    outputs[key] = torch.stack(outputs[key])
                for method in methods:
                    val = self.blend(*methods[method], outputs, poses[i, keep_flags, :3, 3], self.S_invs[keep_flags, 0, 0], save_extras, keep_flags)
                    rgb_chunks[method].append(val[0].cpu())
                    if save_extras:
                        ws_chunks[method].append(val[1].cpu())
            if save_extras:
                for model_name in self.model_names:
                    img = (torch.cat(rgb_chunks[model_name]).view(h, w, -1) * 255).to(torch.uint8).numpy()
                    imageio.v3.imwrite(output_dir / model_name / f'{i:04d}.png', img)
                    if animate:
                        imgs[model_name].append(img)
            for method in methods:
                img = (torch.cat(rgb_chunks[method]).view(h, w, -1) * 255).to(torch.uint8).numpy()
                imageio.v3.imwrite(output_dir / method / f'{i:04d}.png', img)
                if save_extras:
                    ws = (torch.cat(ws_chunks[method], dim=1).view(-1, h, w, 1) * 255).to(torch.uint8).broadcast_to(-1, -1, -1, 3).numpy()
                    imageio.v3.imwrite(output_dir / method / f'w{i:04d}.png', img_cat(ws, 1))
                if animate:
                    imgs[method].append(img)
        if animate:
            if save_extras:
                for model_name in self.model_names:
                    imageio.v3.imwrite(output_dir / f'{model_name}.mp4', imgs[model_name], fps=animate, quality=10)
            for method in methods:
                imageio.v3.imwrite(output_dir / f'{method}.mp4', imgs[method], fps=animate, quality=10)

    def blend(self, method, g, data, c2w_ts, scales, save_extras, keep_flags):
        n_models = len(keep_flags)
        n_keeps, n_rays = data['weights'].shape[:2]
        if method in {'nearest', 'idw2'}:
            if method == 'nearest':
                g = torch.inf
            dists = torch.linalg.norm(c2w_ts, dim=-1)
            if self.use_global_metric:
                dists *= scales
            ws = self.idw(dists, g)[:, None, None].broadcast_to(-1, n_rays, -1)
            val = (data['rgb'] * ws).sum(dim=0)
            if save_extras:
                ws_full = torch.zeros(n_models, n_rays, 1, device=ws.device)
                ws_full[keep_flags] = ws
                return val, ws_full
            return val,
        if method == 'idw3':
            dists = torch.linalg.norm(c2w_ts[:, None, :] + data['direction'] * data['depth'], dim=-1)
            if self.use_global_metric:
                dists *= scales[:, None]
            ws = self.idw(dists, g)[..., None]
            val = (data['rgb'] * ws).sum(dim=0)
            if save_extras:
                ws_full = torch.zeros(n_models, n_rays, 1, device=ws.device)
                ws_full[keep_flags] = ws
                return val, ws_full
            return val,
        if method == 'idw4':
            bg = 'last_sample'
            merged_weights, merged_rgbs, merged_mids = self.merge_ray_samples(data['weights'], data['rgbs'], data['deltas'], scales)
            dists = torch.linalg.norm(c2w_ts[:, None, None, :] + data['direction'][..., None, :] * merged_mids, dim=-1)
            if self.use_global_metric:
                dists *= scales[:, None, None]
            ws = self.idw(dists, g)
            w_bg = ws[..., [-1]] if bg == 'last_sample' else torch.full((n_keeps, n_rays, 1), 1 / n_keeps, device=ws.device)
            cs = merged_weights * ws[..., None]
            c_bg = (1 - data['accumulation']) * w_bg
            s = torch.cat([cs, c_bg[..., None, :]], dim=-2).sum(dim=(0, -2), keepdim=True)
            c = 1 / (cs.shape[0] * (cs.shape[-2] + 1))
            cs = torch.nan_to_num(cs / s, nan=c)
            c_bg = torch.nan_to_num(c_bg / s.squeeze(-2), nan=c)
            val = WeightedRGBRenderer(background_color=bg)(merged_rgbs, cs, c_bg).sum(dim=0).clamp(min=0, max=1)
            if save_extras:
                ws_full = torch.zeros(n_models, n_rays, 1, device=ws.device)
                ws_full[keep_flags] = torch.cat([cs, c_bg[..., None, :]], dim=-2).sum(dim=-2)
                return val, ws_full
            return val,

    @staticmethod
    def idw(dists, g):
        """ dists: (n_dists, ...) """
        t = (dists.unsqueeze(1) / dists).nan_to_num(nan=1)
        return 1 / (t**g).sum(dim=1)

    @staticmethod
    def merge_ray_samples(weights, rgbs, deltas, scales):
        # weights (n_models, n_rays, n_samples, 1)
        # rgbs (n_models, n_rays, n_samples, 3)
        # deltas (n_models, n_rays, n_samples, 1)
        # scales (n_models)

        # merged_weights (n_models, n_rays, n_samples * n_models, 1)
        # merged_rgbs (n_models, n_rays, n_samples * n_models, 3)
        # merged_mids (n_rays, n_samples * n_models, 1)

        device = weights.device
        n_models, n_rays, n_samples = weights.shape[:3]
        merged_weights = torch.empty(n_models, n_rays, n_samples * n_models, 1, device=device)
        merged_rgbs = torch.empty(n_models, n_rays, n_samples * n_models, 3, device=device)
        merged_ends = torch.empty(n_rays, n_samples * n_models, 1, device=device)

        weights = torch.cat((weights, torch.zeros(n_models, n_rays, 1, 1, device=device)), dim=-2)
        rgbs = torch.cat((rgbs, torch.zeros(n_models, n_rays, 1, 3, device=device)), dim=-2)
        deltas = torch.cat((deltas, torch.full((n_models, n_rays, 1, 1), torch.inf, device=device)), dim=-2)
        scales = scales[:, None, None, None]
        deltas *= scales
        ends = torch.cumsum(deltas, dim=-2)
        ps = torch.zeros(n_models, n_rays, 1, 1, dtype=int, device=device)
        for i in range(n_samples * n_models):
            end = ends.gather(-2, ps)
            end_min, model_id = torch.min(end, 0, keepdim=True)
            delta = end_min - (merged_ends[:, [i - 1]] if i else 0)
            merged_weights[:, :, [i]] = delta / deltas.gather(-2, ps) * weights.gather(-2, ps)
            merged_rgbs[:, :, [i]] = rgbs.gather(-2, ps.broadcast_to(n_models, n_rays, 1, 3))
            merged_ends[:, [i]] = end_min[0]
            ps.scatter_(0, model_id, ps.gather(0, model_id) + 1)
        merged_ends = torch.cat((torch.zeros((n_rays, 1, 1), device=device), merged_ends), dim=-2)
        return merged_weights, merged_rgbs, (merged_ends[:, :-1] + merged_ends[:, 1:]) / 2 / scales
