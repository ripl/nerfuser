# NeRFuser

## Installation

### 0. Create a conda environment and activate it

```bash
conda create -n nerfuser -y python=3.9 && conda activate nerfuser
```

### 1. Install dependencies

* [nerfstudio](https://github.com/nerfstudio-project/nerfstudio)

    ```bash
    pip install torch==1.13.1 torchvision functorch
    pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
    pip install nerfstudio
    ```

* [hloc](https://github.com/cvg/Hierarchical-Localization)

    ```bash
    git clone --recursive git@github.com:cvg/Hierarchical-Localization.git && pip install -e Hierarchical-Localization
    ```

* [imageio-ffmpeg](https://pypi.org/project/imageio-ffmpeg/)

    ```bash
    pip install imageio-ffmpeg
    ```

### 2. Install NeRFuser

```bash
pip install git+https://github.com/ripl/nerfuser
```

## Run

Run `python -m nerfuser.fuser -h` for details. Usage examples:

```bash
python -m nerfuser.fuser \
    --model-dirs models/ttic/common_large/A/nerfacto/2023-04-16_185306/nerfstudio_models/ models/ttic/common_large/B/nerfacto/2023-04-16_185245/nerfstudio_models/ models/ttic/common_large/C/nerfacto/2023-04-16_185251/nerfstudio_models/ \
    --name ttic/common_large \
    --model-gt-trans I \
    --cam-info data/ttic/common_large/test/transforms.json \
    --render-views \
    --run-sfm \
    --compute-trans \
    --test-poses data/ttic/common_large/test/transforms.json \
    --test-frame world \
    --blend-views \
    --eval-blend

python -m nerfuser.fuser \
    --model-dirs models/ttic/common_large/A/nerfacto/2023-04-16_185306/nerfstudio_models/ models/ttic/common_large/B/nerfacto/2023-04-16_185245/nerfstudio_models/ models/ttic/common_large/C/nerfacto/2023-04-16_185251/nerfstudio_models/ \
    --name ttic/common_large \
    --render-views \
    --run-sfm \
    --compute-trans \
    --blend-views
```
