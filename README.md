# NeRFuser

Official code release for NeRFuser.

https://github.com/ripl/nerfuser/assets/7736732/c622b940-4e20-4a34-915a-33c2b23e1b0f

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
git clone git@github.com:ripl/nerfuser.git && cd nerfuser/
pip install .
```

## Data Preparation

The data preparation assumes that you have several `MOV` videos focusing on different yet overlapping portions of the same scene. One of the videos should be named `test.MOV`, from which we will extract images for blending evaluation. Others can be named whatever you like. Assume you collected 3 more videos besides `test.MOV`, whose names w/o the `MOV` extension are stored as `A`, `B` and `C`. First put all the video files in the directory `DATASET_DIR`, then run the following command to prepare data from the videos:

```bash
python prep_data.py \
    --dataset-dir $DATASET_DIR \
    --vid-ids test $A $B $C \
    --downsample 8 \
    --extract-images \
    --run-sfm \
    --write-json \
    --vis
```

Please look further into `prep_data.py` for more options. A sample dataset is provided [here](https://huggingface.co/datasets/RIPL/TTIC-common/tree/main).

## Training NeRFs

Let `MODELS_DIR` be the directory where you want to save the trained NeRF models. Run the following command to train a NeRF model for each video other than `test`:

```bash
for VID in $A $B $C; do
    ns-train nerfacto \
        --output-dir $MODELS_DIR \
        --data $DATASET_DIR/$VID \
        --viewer.quit-on-train-completion True \
        --pipeline.datamanager.camera-optimizer.mode off
done
```

## NeRF Registration

Let `TS_A`, `TS_B` and `TS_C` be the timestamps of the trained NeRF models for videos `A`, `B` and `C` respectively. Run the following command to register the NeRF models:

```bash
python -m nerfuser.registration \
    --model-dirs $MODELS_DIR/$A/nerfacto/$TS_A/nerfstudio_models $MODELS_DIR/$B/nerfacto/$TS_B/nerfstudio_models $MODELS_DIR/$C/nerfacto/$TS_C/nerfstudio_models \
    --name my_scene \
    --model-names $A $B $C \
    --model-gt-trans I \
    --cam-info $DATASET_DIR/test/transforms.json \
    --render-views \
    --run-sfm \
    --compute-trans \
    --vis
```

Registration results are saved in `outputs/registration` by default. Please run `python -m nerfuser.registration -h` for more details.

## NeRF Blending

Run the following command to query the NeRFs with test poses as in `test.MOV` and generate the blending results:

```bash
python -m nerfuser.blending \
    --model-dirs $MODELS_DIR/$A/nerfacto/$TS_A/nerfstudio_models $MODELS_DIR/$B/nerfacto/$TS_B/nerfstudio_models $MODELS_DIR/$C/nerfacto/$TS_C/nerfstudio_models \
    --name my_scene \
    --model-names $A $B $C \
    --cam-info $DATASET_DIR/test/transforms.json \
    --test-poses $DATASET_DIR/test/transforms.json \
    --test-frame world \
    --blend-views \
    --evaluate
```

Blending results are saved in `outputs/blending` by default. Please run `python -m nerfuser.blending -h` for more details.

## NeRF Fusion

Alternative to the above two steps, you can run the following command to perform NeRF registration and blending in one go:

```bash
python -m nerfuser.fuser \
    --model-dirs $MODELS_DIR/$A/nerfacto/$TS_A/nerfstudio_models $MODELS_DIR/$B/nerfacto/$TS_B/nerfstudio_models $MODELS_DIR/$C/nerfacto/$TS_C/nerfstudio_models \
    --name my_scene \
    --model-names $A $B $C \
    --model-gt-trans I \
    --cam-info $DATASET_DIR/test/transforms.json \
    --render-views \
    --run-sfm \
    --compute-trans \
    --test-poses data/ttic/common_large/test/transforms.json \
    --test-frame world \
    --blend-views \
    --eval-blend
```

Please run `python -m nerfuser.fuser -h` for more details.
