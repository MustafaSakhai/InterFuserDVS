# InterfuserDVS: DVS-Based Sensor Fusion Transformer for Safe RL-Based Decision Making

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)

## Overview

This repository contains the official implementation of **InterfuserDVS**, an initial DVS-enhanced InterFuser architecture that integrates Dynamic Vision Sensors (DVS) with traditional multi-modal perception for autonomous driving. Our approach extends the state-of-the-art InterFuser transformer with event-based sensing capabilities, achieving robust performance across challenging environmental conditions.

### Key Features

- 🚗 **Multi-Modal Fusion**: Integrates RGB cameras, LiDAR, semantic segmentation, depth maps, and DVS event cameras
- 🧠 **Transformer Architecture**: 77.4M parameter model with 6-layer encoder-decoder structure
- ⚡ **Event-Based Sensing**: DVS integration for high temporal resolution and dynamic range
- 🌦️ **Weather Robustness**: Trained across 20 diverse weather conditions in CARLA
- 📊 **Comprehensive Evaluation**: Validated on 250+ episodes in CARLA Town 05
- 🔬 **Synthetic DVS Pipeline**: v2e-based conversion from LMDrive dataset

## Architecture

The DVS-enhanced InterFuser employs a sophisticated multi-modal fusion mechanism:

- **DVS Backbone**: ResNet50d with 3 input channels for event processing
- **RGB Backbone**: ResNet50d with ImageNet pre-training
- **LiDAR Backbone**: ResNet18d with 2 input channels
- **Transformer Encoder**: 6 layers with 8-head attention (256 embedding dim)
- **Transformer Decoder**: 6 layers with cross-attention
- **Total Parameters**: 77,382,351

## Installation

### Prerequisites

- Python 3.8+
- CUDA 11.0+
- 8 GPUs (recommended for training)

### Environment Setup

1. **Clone the repository**:
```bash
git clone https://github.com/MustafaSakhai/InterFuserDVS.git
cd InterFuserDVS
```

2. **Create conda environment**:
```bash
conda create -n interfuser python=3.8
conda activate interfuser
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Install CARLA** (for evaluation):
```bash
# Download CARLA 0.9.10.1
wget https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/CARLA_0.9.10.1.tar.gz
tar -xzf CARLA_0.9.10.1.tar.gz
export CARLA_ROOT=/path/to/CARLA_0.9.10.1
```

## Dataset Preparation

### LMDrive Dataset

1. **Download LMDrive dataset**:
```bash
# Follow instructions at: https://github.com/opendilab/LMDrive
# Download the dataset to your preferred location
```

2. **DVS Conversion with v2e**:

For DVS event conversion, we use the official [v2e repository](https://github.com/SensorsINI/v2e) <mcreference link="https://github.com/SensorsINI/v2e" index="0"></mcreference>:

```bash
# Clone and install v2e
git clone https://github.com/SensorsINI/v2e.git
cd v2e
conda create -n v2e python=3.10
conda activate v2e
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch
python -m pip install -e .

# Download SuperSloMo model (required for v2e)
# Download SuperSloMo39.ckpt from Google Drive and place in input folder

# Convert RGB frames to DVS events according to instructions
```

For detailed installation and usage instructions, please refer to the [official v2e documentation](https://github.com/SensorsINI/v2e) <mcreference link="https://github.com/SensorsINI/v2e" index="0"></mcreference>.

3. **Dataset Structure**:
```
dataset/
├── route_00/
│   ├── rgb_front/
│   │   ├── 0000.png
│   │   ├── 0001.png
│   │   └── ...
│   ├── rgb_left/
│   ├── rgb_right/
│   ├── dvs_full/          # DVS front view
│   │   ├── 0000.png
│   │   ├── 0001.png
│   │   └── ...
│   ├── dvs_left/          # DVS left view
│   ├── dvs_right/         # DVS right view
│   ├── lidar/
│   ├── measurements/
│   └── waypoints/
├── route_01/
└── ...
```

## Training

### Quick Start

```bash
# Single GPU training
python interfuser/train.py \
    --model interfuser_baseline_dvs \
    --dataset carla \
    --train-towns 5 \
    --val-towns 5 \
    --with-dvs \
    --multi-view \
    --with-lidar \
    --epochs 25 \
    --batch-size 16
```

### Multi-GPU Training (Recommended)

```bash
# 8 GPU distributed training
./interfuser/distributed_train.sh 8 /path/to/dataset \
    --dataset carla \
    --train-towns 5 \
    --val-towns 5 \
    --train-weathers 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 \
    --val-weathers 10 11 12 13 \
    --model interfuser_baseline_dvs \
    --sched cosine \
    --epochs 25 \
    --warmup-epochs 5 \
    --lr 0.0005 \
    --batch-size 16 \
    --opt adamw \
    --weight-decay 0.05 \
    --with-backbone-lr \
    --backbone-lr 0.0002 \
    --multi-view \
    --with-lidar \
    --with-dvs \
    --multi-view-input-size 3 128 128 \
    --log-wandb
```

## Model Configuration

### Key Parameters

- **Model**: `interfuser_baseline_dvs`
- **Input Modalities**: RGB (multi-view), LiDAR, DVS (multi-view), Segmentation, Depth
- **Learning Rate**: 0.0005 (main), 0.0002 (backbone)
- **Optimizer**: AdamW with cosine scheduling
- **Batch Size**: 16 per GPU
- **Training Towns**: CARLA Town 05
- **Weather Conditions**: 1-20 (training), 10-13 (validation)


## Contributing

We welcome contributions! 
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Authors and Affiliations**:
    - **Mustafa Sakhai**: Faculty of Computer Science, Electronics and Telecommunications, AGH University of Krakow, Krakow, 30-059, Poland
    - **Kaung Sithu**: Faculty of Computer Science, Electronics and Telecommunications, AGH University of Krakow, Krakow, 30-059, Poland
    - **Maciej Wielgosz**: 
        - Faculty of Computer Science, Electronics and Telecommunications, AGH University of Krakow, Krakow, 30-059, Poland
        - Academic Computer Centre AGH, AGH University of Krakow, Krakow, 30-950, Poland
- **Base Architecture**: [InterFuser](https://github.com/opendilab/InterFuser) by OpenDILab
- **Dataset**: [LMDrive](https://github.com/opendilab/LMDrive)
- **DVS Conversion**: [v2e](https://github.com/SensorsINI/v2e)

## Contact

- **Mustafa Sakhai**: msakhai@agh.edu.pl
- **Kaung Sithu**: sithu@student.agh.edu.pl
- **Maciej Wielgosz**: wielgosz@agh.edu.pl

## Related Work

- [InterFuser: Safety-Enhanced Autonomous Driving Using Interpretable Sensor Fusion Transformer](https://github.com/opendilab/InterFuser)
- [LMDrive: Closed-Loop End-to-End Driving with Large Language Models](https://github.com/opendilab/LMDrive)
- [v2e: From Video Frames to Realistic DVS Events](https://github.com/SensorsINI/v2e)

---

**Note**: This is a research implementation. For production use, additional safety measures and extensive testing are required.