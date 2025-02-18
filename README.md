# SSDFusion

Rui Ming, Yixian Xiao, Xinyu Liu, Guolong Zheng, Guobao Xiao*,
SSDFusion: A scene-semantic decomposition approach for visible and infrared image fusion,
Pattern Recognition, 2025.

- [DOI](https://doi.org/10.1016/j.patcog.2025.111457)

# Abstract

Visible and infrared image fusion aims to generate fused images with comprehensive scene understanding and detailed contextual information. However, existing methods often struggle to adequately handle relationships between different modalities and optimize for downstream applications. To address these challenges, we propose a novel scene-semantic decomposition-based approach for visible and infrared image fusion, termed _SSDFusion_. Our method employs a multi-level encoder-fusion network with fusion modules implementing the proposed scene-semantic decomposition and fusion strategy to extract and fuse scene-related and semantic-related components, respectively, and inject the fused semantics into scene features, enriching the contextual information in fused features while sustaining fidelity of fused images. Moreover, we further incorporate meta-feature embedding to connect the encoder-fusion network with the downstream application network during the training process, enhancing our method's ability to extract semantics, optimize the fusion effect, and serve tasks such as semantic segmentation. Extensive experiments demonstrate that SSDFusion achieves state-of-the-art image fusion performance while enhancing results on semantic segmentation tasks. Our approach bridges the gap between feature decomposition-based image fusion and high-level vision applications, providing a more effective paradigm for multi-modal image fusion.

# Quick Start

## Setup

To setup evaluation environment for our method, run the following command:

```bash
pip install -r requirements.txt
```

Additionally, full environment requirements are provided in `environment.yml`.

## Data preparation

Format of datasets:

```
DATASET
├── dataset.toml
├── train
│   ├── ir
│   ├── Segmentation_labels
│   └── vi
├── test
│   ├── ir
│   ├── Segmentation_labels
│   └── vi
```

All datasets should follow the above format for training and testing.

For image reconstruction and fusion learning, images are needed to be preprocessed.
To preprocess the dataset, adjust parameters in `utils/datapreprocess.py`,
then run the script to get preprocessed dataset in HDF5 format.

## Inference

To inference with our proposed method. Download the weights from [Releases](https://github.com/YiXian-Xiao/SSDFusion/releases).
Then run the following command:

```bash
python -m utils.test --model <path to model weights> \
                     --dataset-path <path to dataset>/test \
                     --output <path to output dir>
```

## Training

The training process is divided into two phases: Image Reconstruction and Fusion Learning, and Meta-feature Embedding.

A set of variables used in the commands are defined as follows:

| Name         | Description                                    |
|--------------|------------------------------------------------|
| SESSION_NAME | The name of a training session defined by user |
| DATASET      | The name of the dataset used                   |

### Phase I: Image reconstruction and fusion learning

For image reconstruction learning, run the following command:

```bash
python -m utils.train --session-name $SESSION_NAME \
                      --config config/$DATASET/stage-1.toml
```

Checkpoints can be found in `work/$SESSION/checkpoint`. The interval of saving checkpoints can be adjusted inside configs.

Statistics such as loss values are logged using TensorBoard, which can be found inside `work/$SESSION/tensorboard`.

For image fusion learning, run the following command:

```bash
python -m utils.train --session-name $SESSION_NAME \
                      --config config/$DATASET/stage-2.toml \
                      --resume <Checkpoint from previous stage> \
                      --resume-params-only
```

### Phase II: Meta-feature embedding

To generate images for training downstream model at this stage, run the following command for the `train` dataset:

```bash
python -m utils.evaluate --session-name <Your session name> \
                         --config config/msrs/stage-3/round-0/config.toml \
                         --model work/$SESSION_NAME/checkpoint/Stage2_MSRS-latest.pth \
                         --dataset train \
                         --session-name $SESSION_NAME fusion --rgb --output "work/$SESSION_NAME/output/train/fused" --no-metric
```

and the following command for `test` dataset:

```bash
python -m utils.evaluate --session-name <Your session name> \
                         --config config/msrs/stage-3/round-0/config.toml \
                         --model work/$SESSION_NAME/checkpoint/Stage2_MSRS-latest.pth \
                         --dataset test \
                         --session-name $SESSION_NAME fusion --rgb --output "work/$SESSION_NAME/output/train/fused" --no-metric
```

Then use the generated fused results to train a downstream application model.

To begin the meta-feature embedding phase, first convert a checkpoint from previous stage using the following command:

```bash
python -m utils.convert_checkpoint <path of checkpoint to convert> <path of converted checkpoint> II III
```
