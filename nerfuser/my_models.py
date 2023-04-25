from dataclasses import dataclass, field
from typing import Type, Union

import torch
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.model_components.renderers import RGBRenderer

# from nerfstudio.models.bayes_nerf import BayesNeRFModel, BayesNeRFModelConfig
from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig

# from nerfstudio.models.nerfacto_wo_app import NerfactoWoAppModel, NerfactoWoAppModelConfig
from torchtyping import TensorType
from typing_extensions import Literal


@dataclass
class MyNerfactoModelConfig(NerfactoModelConfig):
    """My Nerfacto Model Config"""

    _target: Type = field(default_factory=lambda: MyNerfactoModel)


class MyNerfactoModel(NerfactoModel):
    def get_outputs(self, ray_bundle: RayBundle):
        ray_samples = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)[0]
        field_outputs = self.field(ray_samples)

        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        rgbs = field_outputs[FieldHeadNames.RGB]
        rgb = self.renderer_rgb(rgb=rgbs, weights=weights)
        depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        accumulation = self.renderer_accumulation(weights=weights)

        weights = torch.cat((torch.zeros_like(weights[..., [0], :]), weights), dim=-2)
        rgbs = torch.cat((torch.zeros_like(rgbs[..., [0], :]), rgbs), dim=-2)
        deltas = torch.cat((ray_samples.frustums.starts[..., [0], :], ray_samples.deltas), dim=-2)

        outputs = {
            'weights': weights,  # (n_rays, n_samples, 1)
            'rgbs': rgbs,  # (n_rays, n_samples, 3)
            'rgb': rgb,  # (n_rays, 3)
            'accumulation': accumulation,  # (n_rays, 1)
            'depth': depth,  # (n_rays, 1)
            'deltas': deltas,  # (n_rays, n_samples, 1)
            'direction': ray_samples.frustums.directions[:, 0]  # (n_rays, 3)
        }
        return outputs


# @dataclass
# class MyInstantNGPModelConfig(InstantNGPModelConfig):
#     """My Instant NGP Model Config"""

#     _target: Type = field(default_factory=lambda: MyNGPModel)


# class MyNGPModel(NGPModel):
#     def get_outputs(self, ray_bundle: RayBundle):
#         num_rays = len(ray_bundle)
#         with torch.no_grad():
#             ray_samples, ray_indices = self.sampler(
#                 ray_bundle=ray_bundle,
#                 near_plane=self.config.near_plane,
#                 far_plane=self.config.far_plane,
#                 render_step_size=self.config.render_step_size,
#                 cone_angle=self.config.cone_angle,
#             )
#         field_outputs = self.field(ray_samples)
#         packed_info = nerfacc.pack_info(ray_indices, num_rays)

#         weights = nerfacc.render_weight_from_density(
#             packed_info=packed_info,
#             sigmas=field_outputs[FieldHeadNames.DENSITY],
#             t_starts=ray_samples.frustums.starts,
#             t_ends=ray_samples.frustums.ends,
#         )
#         rgbs = field_outputs[FieldHeadNames.RGB]
#         rgb = self.renderer_rgb(
#             rgb=rgbs,
#             weights=weights,
#             ray_indices=ray_indices,
#             num_rays=num_rays,
#         )
#         depth = self.renderer_depth(weights=weights, ray_samples=ray_samples, ray_indices=ray_indices, num_rays=num_rays)
#         accumulation = self.renderer_accumulation(weights=weights, ray_indices=ray_indices, num_rays=num_rays)

#         outputs = {
#             'rgbs': rgbs,  # (n_samples, 3)
#             'rgb': rgb,  # (n_rays, 3)
#             'accumulation': accumulation,  # (n_rays, 1)
#             'depth': depth,  # (n_rays, 1)
#             'weights': weights,  # (n_samples, 1)
#             'starts': ray_samples.frustums.starts,  # (n_samples, 1)
#             'ends': ray_samples.frustums.ends,  # (n_samples, 1)
#             # 'num_samples_per_ray': packed_info[:, 1],  # (n_rays)
#             'directions': ray_samples.frustums.directions[packed_info[:, 0].to(int)]  # (n_rays, 3)
#         }
#         return outputs


class MyRGBRenderer(RGBRenderer):
    """Weighted volumetic rendering.

    Args:
        background_color: Background color as RGB. Uses random colors if None.
    """

    @classmethod
    def combine_rgb(
        cls,
        rgb: TensorType["bs":..., "num_samples", 3],
        ws: TensorType["bs":..., "num_samples", 1],
        bg_w: TensorType["bs":..., 1],
        background_color: Union[Literal["random", "last_sample"], TensorType[3]] = "random"
    ) -> TensorType["bs":..., 3]:
        """Composite samples along ray and render color image

        Args:
            rgb: RGB for each sample.
            weights: Termination probability mass for each sample.
            ws: Weights for each sample. E.g. from IDW.
            background_color: Background color as RGB.
            ray_indices: Ray index for each sample, used when samples are packed.
            num_rays: Number of rays, used when samples are packed.

        Returns:
            Outputs rgb values.
        """
        comp_rgb = torch.sum(rgb * ws, dim=-2)

        if background_color == "last_sample":
            background_color = rgb[..., -1, :]
        elif background_color == "random":
            background_color = torch.rand_like(comp_rgb)

        comp_rgb += background_color * bg_w

        return comp_rgb

    def forward(
        self,
        rgb: TensorType["bs":..., "num_samples", 3],
        ws: TensorType["bs":..., "num_samples", 1],
        bg_w: TensorType["bs":..., 1],
    ) -> TensorType["bs":..., 3]:
        """Composite samples along ray and render color image

        Args:
            rgb: RGB for each sample.
            ws: weighted termination probability mass for each sample.

        Returns:
            Outputs of rgb values.
        """

        rgb = self.combine_rgb(rgb, ws, bg_w, background_color=self.background_color)
        return rgb
