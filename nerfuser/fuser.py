from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional, Union

import tyro

from nerfuser.blending import Blending
from nerfuser.registration import Registration


@dataclass
class Fuser:
    """
    Fuse (i.e. register and blend) an arbitrary number of NeRFs to render novel views.
    """

    model_dirs: List[Path]
    """model checkpoint directories"""
    enable_reg: bool = True
    """whether to enable registration"""
    enable_blend: bool = True
    """whether to enable blending"""
    reg_output_dir: Path = Path('outputs/registration')
    """output directory for registration results"""
    name: Optional[str] = None
    """if present, will continue with the existing named experiment"""
    model_method: Literal['nerfacto'] = 'nerfacto'
    """model method"""
    model_names: Optional[List[str]] = None
    """names of models to fuse"""
    model_gt_trans: Optional[str] = None
    """path to npy containing ground-truth transforms from the common world coordinate system to each model's local one; can be "identity" """
    step: Optional[int] = None
    """model step to load"""
    cam_info: Union[str, List[float]] = field(default_factory=lambda: [400.0, 400.0, 400.0, 300.0, 800, 600])
    """either path to json or cam params (fx fy cx cy w h)"""
    downscale_factor: Optional[float] = None
    """downscale factor for NeRF rendering"""
    n_hemi_poses: int = 30
    """number of hemispheric poses to render for registration"""
    chunk_size: Optional[int] = None
    """number of rays to process at a time"""
    sfm_tool: Literal['hloc', 'colmap'] = 'hloc'
    """SfM tool to use for registration"""
    device: str = 'cuda:0'
    """device to use"""
    render_views: bool = False
    """whether to render views for registration"""
    run_sfm: bool = False
    """whether to run SfM for registration"""
    compute_trans: bool = False
    """whether to compute transforms for registration"""
    vis: bool = False
    """whether to visualize the registration"""
    blend_output_dir: Path = Path('outputs/blending')
    """output directory for blending results"""
    test_poses: Optional[str] = None
    """path to json containing test poses for blending; will use circular poses if not specified"""
    test_frame: Literal['sfm', 'world'] = 'sfm'
    """the coordinate system in which test-poses are defined"""
    trans_src: Literal['hemi', 'gt'] = 'hemi'
    """source of sfm to normalized nerf transforms; if "gt", will use "model-gt-trans" and test-frame must be "world" """
    blend_methods: List[Literal['nearest', 'idw2', 'idw3', 'idw4']] = field(default_factory=lambda: ['idw4'])
    """blending methods"""
    tau: float = 2.5
    """maximum blending distance ratio; must be larger than 1"""
    gammas: List[float] = field(default_factory=lambda: [4])
    """blending rates for all applicable methods"""
    fps: int = 8
    """frame rate for video output"""
    save_extras: bool = False
    """whether to save extra outputs (raw renderings, blending weight maps)"""
    blend_views: bool = False
    """whether to blend views"""
    eval_blend: bool = False
    """whether to evaluate the blending results"""

    def main(self) -> None:
        """main method"""
        if self.enable_reg:
            Registration(
                model_dirs=self.model_dirs,
                output_dir=self.reg_output_dir,
                name=self.name,
                model_method=self.model_method,
                model_names=self.model_names,
                model_gt_trans=self.model_gt_trans,
                step=self.step,
                cam_info=self.cam_info,
                downscale_factor=self.downscale_factor,
                n_hemi_poses=self.n_hemi_poses,
                chunk_size=self.chunk_size,
                sfm_tool=self.sfm_tool,
                device=self.device,
                render_views=self.render_views,
                run_sfm=self.run_sfm,
                compute_trans=self.compute_trans,
                vis=self.vis,
            ).main()
        if self.enable_blend:
            Blending(
                model_dirs=self.model_dirs,
                output_dir=self.blend_output_dir,
                name=self.name,
                model_method=self.model_method,
                model_names=self.model_names,
                model_gt_trans=self.model_gt_trans,
                step=self.step,
                cam_info=self.cam_info,
                downscale_factor=self.downscale_factor,
                test_poses=self.test_poses,
                test_frame=self.test_frame,
                chunk_size=self.chunk_size,
                reg_dir=self.reg_output_dir,
                reg_name=self.name,
                trans_src=self.trans_src,
                blend_methods=self.blend_methods,
                tau=self.tau,
                gammas=self.gammas,
                fps=self.fps,
                save_extras=self.save_extras,
                device=self.device,
                blend_views=self.blend_views,
                evaluate=self.eval_blend,
            ).main()


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color('bright_yellow')
    tyro.cli(Fuser).main()


if __name__ == '__main__':
    entrypoint()
