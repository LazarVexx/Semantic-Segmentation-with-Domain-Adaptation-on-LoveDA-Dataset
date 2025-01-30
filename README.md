# Semantic Segmentation with Domain Adaptation on LoveDA Dataset

This repository implements and benchmarks advanced semantic segmentation architectures with domain adaptation strategies on the **LoveDA** remote sensing dataset. Focused on urban-to-rural domain shift challenges, we provide a unified framework for comparing:

- **PIDNet** (Real-time segmentation)
- **DeepLabV2** (Atrous spatial pyramid pooling)
- **CycleGAN** (Domain translation)
- **PEM** (Prototype-based segmentation)

In ResultTable.md and in "Experiments and Results" folders you will find the result of our experiments. 

### For additional information:
In the .pdf file in s327313_s324807_s326811_project4.pdf 



## Installation

### Basic Setup
```
git clone https://github.com/LucaIanniello/AML2024
cd AML2024
```

**System Requirements**:
- Python 3.8+ with pip
- NVIDIA GPU (16GB+ VRAM recommended)
- CUDA 11.3+ and cuDNN 8.2+

For manual installation: [Download ZIP](https://github.com/LucaIanniello/AML2024/archive/refs/heads/main.zip)

## Implemented Architectures
All the model are on Google Colab, ready-to-run or in locally. 

### 🎯 DeepLabV2 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ywc1VuXIAH3tmSfRn8ev3yvSJDGAvSxF?usp=sharing)
- Atrous convolutions for dense feature extraction
- ASPP module for multi-scale context

### ⚡ PIDNet 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/126h9tjDoQ4w1jrmareDz9scs8UbT5VMe?usp=sharing)

Locally: 
```
cd PIDNet
pip install -r requirements.txt
python tools/importDataset.py
```
In run.sh  you will find all the commands to start the trainings. 

- Triple-branch architecture (P/I/D)
- Boundary-aware loss function
- Real-time inference capabilities

### 🔄 CycleGAN 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1coAv3KDPPzsrPN3k-t6qIDQ5AKYw-kEP?usp=sharing)

Dataset CycleGAN LoveDa-Urban images: https://zenodo.org/records/14739456

- Unpaired image-to-image translation
- Cycle-consistency loss
- Semantic-guided style transfer

### 🎭 PEM 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1KbzvDoGrSK90cJrAZCP5IxDz6apq-0DR?usp=sharing)
- Deformable transformer architecture
- Prototype-based cross-attention
- Panoptic segmentation support

