# DRADNet: Dual-Gated Reverse Attention and Directional Structure Enhancement for Precise Polyp Segmentation in Colonoscopic Images.


[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.7+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🔧 Installation

### Prerequisites
- Python 3.8+
- CUDA 11.0+ (for GPU training)
- 16GB+ RAM recommended

### Setup

```bash
# Clone the repository
git clone https://github.com/balimhoza2018-wq/DRADNet.git
cd DRADNet

# Create virtual environment
python -m venv DRADNet
source DRADNet/bin/activate  # Linux/Mac
# or
DRADNet\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install PyTorch with CUDA 11.0
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html
```
this should be seperate from setup
## Data preparation
Download and put your polyp dataset in polyp_seg/data

##Backbones Preparation
It is required to download pre-trained weights for shunted_s and place it in DRADnet/polyp_seg/pretrained_weight


##Training, Testing and Evaluation
cd polyp_seg
python -W ignore Train.py
After training has completed, the checkpoints and best model will be saved DRADNet/polyp_seg/snapshots/DRADNet_res

#Testing
cd polyp_seg
python -W ignore Test.py
After the command, the predicted mask will be saved in DRADNet/polyp_seg/results/DRADNet

#Evaluation
cd polyp_seg
python -W ignore eval.py
The evaluation results will be saved in DRADNet/polyp_seg/eval_results
