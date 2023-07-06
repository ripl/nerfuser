import torch
from jaxtyping import Float
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.model_components.renderers import RGBRenderer
from torch import Tensor


def get_nerfacto_outputs(self, ray_bundle: RayBundle):
    ray_samples = self.proposal_sampler(ray_bundle=ray_bundle, density_fns=self.density_fns)[0]
    field_outputs = self.field(ray_samples, compute_normals=self.config.predict_normals)

    weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
    rgbs = field_outputs[FieldHeadNames.RGB]
    rgb = self.renderer_rgb(rgbs, weights)
    depth = self.renderer_depth(weights, ray_samples)
    accumulation = self.renderer_accumulation(weights)

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
        'direction': ray_samples.frustums.directions[..., 0, :]  # (n_rays, 3)
    }
    if self.config.predict_normals:
        outputs.update({
            'normals': self.normals_shader(self.renderer_normals(field_outputs[FieldHeadNames.NORMALS], weights)),  # computed normals from density gradients
            'pred_normals': self.normals_shader(self.renderer_normals(field_outputs[FieldHeadNames.PRED_NORMALS], weights))  # predicted normals from MLP
        })
    return outputs


class WeightedRGBRenderer(RGBRenderer):
    """Weighted volumetic rendering."""

    def forward(
        self,
        rgb: Float[Tensor, '*bs n_samples 3'],
        ws: Float[Tensor, '*bs n_samples 1'],
        bg_w: Float[Tensor, '*bs 1'],
    ) -> Float[Tensor, '*bs 3']:
        """Composite samples along ray and render color image

        Args:
            rgb: RGB for each sample.
            ws: Weighted termination probability mass for each sample.
            bg_w: Weighted termination probability mass for the background.

        Returns:
            Rendered RGB values.
        """
        comp_rgb = torch.sum(rgb * ws, dim=-2)
        if self.background_color == 'last_sample':
            background_color = rgb[..., -1, :]
        elif self.background_color == 'random':
            background_color = torch.rand_like(comp_rgb)
        else:
            background_color = self.background_color
        comp_rgb += background_color * bg_w
        return comp_rgb
