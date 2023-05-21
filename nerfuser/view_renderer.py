import torch

from nerfuser.view_blender import ViewBlender


class ViewRenderer(ViewBlender):
    def __init__(self, model_method, model_name, load_dir, transform=None, load_step=None, chunk_size=None, device='cuda') -> None:
        if transform is None:
            transform = torch.eye(4)
        super().__init__(model_method, (model_name,), (load_dir,), transform[None], None, load_step=load_step, chunk_size=chunk_size, device=device)

    def render_views(self, c2ws, cam_info, output_dir, multi_cam=False, animate=0):
        return super().blend_views(c2ws, cam_info, output_dir, (), multi_cam=multi_cam, save_extras=True, animate=animate)
