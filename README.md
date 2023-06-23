# NeRFuser

Official code release for "NeRFuser: Large-Scale Scene Representation by NeRF Fusion" [[paper](https://arxiv.org/abs/2305.13307)].

<image src="assets/teaser.svg">

### Original Videos

<p align="center">
    <image width="32%" src="assets/A.gif"> <image width="32%" src="assets/B.gif"> <image width="32%" src="assets/C.gif">
</p>

### Raw NeRFs

<p align="center">
    <image width="32%" src="assets/nerfA.gif"> <image width="32%" src="assets/nerfB.gif"> <image width="32%" src="assets/nerfC.gif">
</p>

### *NeRFuser Result*

<p align="center">
    <image width="60%" src="assets/nerfuser.gif">
</p>

## Installation

### 0. Create a conda environment and activate it

```bash
conda create -n nerfuser -y python=3.10 && conda activate nerfuser
```

### 1. Install dependencies

* [nerfstudio](https://github.com/nerfstudio-project/nerfstudio)

    ```bash
    pip install torch torchvision
    pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
    pip install nerfstudio
    ```

* [hloc](https://github.com/cvg/Hierarchical-Localization)

    ```bash
    git clone --recursive git@github.com:cvg/Hierarchical-Localization.git && pip install -e Hierarchical-Localization
    ```

* [imageio-ffmpeg](https://github.com/imageio/imageio-ffmpeg)

    ```bash
    pip install imageio-ffmpeg
    ```

### 2. Install NeRFuser

```bash
git clone git@github.com:ripl/nerfuser.git && cd nerfuser/
pip install .
```

## Data Preparation

The data preparation assumes that you have several videos focusing on different yet overlapping portions of the same scene. Let `ext` denote the video file extension (e.g. `mp4`, `mov`, etc.), then one of the videos should be named `test.ext`, from which images will be extracted for blending evaluation. Others can be named whatever you like. Assume you have collected 3 more videos besides `test.ext`, whose names w/o the `ext` extension are stored as `A`, `B` and `C`. First put all the video files (including `test.ext`) in the directory `DATASET_DIR`, then run the following command to prepare data for training NeRFs:

```bash
python -m nerfuser.prep_data \
    --dataset-dir $DATASET_DIR \
    --vid-ids test $A $B $C \
    --downsample 8 \
    --extract-images \
    --run-sfm \
    --write-json \
    --vis
```

Please run `python -m nerfuser.prep_data -h` for more details. A sample dataset containing both videos and prepared data is provided [here](https://huggingface.co/datasets/RIPL/TTIC-common/tree/main).

## Training NeRFs

Let `MODELS_DIR` be the directory where you want to save the trained NeRF models. Run the following command to train a NeRF model corresponding to each video other than `test`:

```bash
for VID in $A $B $C; do
    ns-train nerfacto \
        --output-dir $MODELS_DIR \
        --data $DATASET_DIR/$VID \
        --viewer.quit-on-train-completion True \
        --pipeline.datamanager.camera-optimizer.mode off
done
```

Please run `ns-train nerfacto -h` for more details. Trained NeRF models on the [sample dataset](https://huggingface.co/datasets/RIPL/TTIC-common/tree/main) are provided [here](https://huggingface.co/RIPL/TTIC-common/tree/main).

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

Run the following command to query the NeRFs with test poses as in `test` and generate the blending results:

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
    
## Citing NeRFuser

If you find our work useful in your research, please consider citing the paper as follows:

``` bibtex
@article{fang23,
    Author  = {Jiading Fang and Shengjie Lin and Igor Vasiljevic and Vitor Guizilini and Rares Ambrus and Adrien Gaidon and Gregory Shakhnarovich and Matthew R. Walter},
    Title   = {{NeRFuser}: {L}arge-Scale Scene Representation by {NeRF} Fusion},
    Journal = {arXiv:2305.13307},
    Year    = {2023},
    Arxiv   = {2305.13307}
}
```
