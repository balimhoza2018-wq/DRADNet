# DRADNet: Dual-Gated Reverse Attention and Directional Structure Enhancement for Precise Polyp Segmentation in Colonoscopic Images.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.7+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Installation

### Setup

```bash
git clone https://github.com/balimhoza2018-wq/DRADNet.git
cd DRADNet
python -m venv DRADNet
source DRADNet/bin/activate
pip install -r requirements.txt
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html
```

## Data Preparation
Download and put your polyp dataset in 
```bash
polyp_seg/data
```
with this structure:
```
polyp_seg/data/
├── TrainDataset/
│   ├── images/
│   └── masks/
├── TestDatase/
    ...

```
## Backbone Preparation
Download pre-trained weights for shunted_s and place it in:
```
polyp_seg/pretrained_weight/
```
## Training, Testing and Evaluation
### Training
```
cd polyp_seg
python -W ignore Train.py
```
After training completes, checkpoints and best model will be saved in:
```
polyp_seg/snapshots/DRADNet_res/
```
### Testing
```
cd polyp_seg
python -W ignore Test.py
```
After testing completes, predicted masks will be saved in:
```
polyp_seg/results/DRADNet/
```
### Evaluation
```
cd polyp_seg
python -W ignore eval.py
```
Evaluation results will be saved in:
```
polyp_seg/eval_results/
