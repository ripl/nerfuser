# nerfuser-release
Official Code Release for NeRFuser

# Install
1. Install [nerfstudio](https://github.com/nerfstudio-project/nerfstudio)
2. Install [hloc](https://github.com/cvg/Hierarchical-Localization)
3. Navigate to this repo and install by
```code
pip install -e .
```

# Run
Use command `ns-fuse`. Example usage:
```
ns-fuse --name ttic_common_large-reg-and-blend --dataset-dir data/ttic/common_large/ --model-method nerfacto --model-names A B C --model-gt-trans I --model-dirs models/common_large/A/nerfacto/2023-04-16_185306/nerfstudio_models/ models/common_large/B/nerfacto/2023-04-16_185245/nerfstudio_models/ models/common_large/C/nerfacto/2023-04-16_185251/nerfstudio_models/ --cam-info data/ttic/common_large/test/transforms.json --render-views --run-sfm  --compute-trans  --test-poses data/ttic/common_large/test/transforms.json  --blend-output-dir outputs/blending/sfm_blend  --gammas 5 --tau 2  --test-frame world  --blend_views --evaluate_blend
```