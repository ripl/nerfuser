import torch

from nerfuser.view_blender import ViewBlender


class ViewRenderer(ViewBlender):
    def __init__(self, model_method, model_name, load_dir, transform=None, load_step=None, chunk_size=None, device='cuda') -> None:
        if transform is None:
            transform = torch.eye(4)
        super().__init__(model_method, [model_name], [load_dir], transform[None], None, load_step, chunk_size, device)

    def render_views(self, c2ws, cam_info, output_dir, multi_cam=False, animate=0):
        return super().blend_views(c2ws, cam_info, output_dir, [], multi_cam, True, animate)

    def query_density(self, pts):
        """ pts: points in local NeRF coordinates before scene contraction """
        pts = torch.as_tensor(pts, device=self.device)
        if pts.shape[-1] != 1:
            pts.unsqueeze_(-1)
        if pts.shape[-2] == 3:
            pts = torch.cat((pts, torch.ones((*pts.shape[:-2], 1, 1), device=self.device)), dim=-2)
        pts = self.transforms[0] @ pts
        return self.models[0].field.density_fn(pts[..., :3, 0])
