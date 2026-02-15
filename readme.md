# Duality AI Offroad Semantic Segmentation - Hackathon Submission

## 📋 Project Overview

This repository contains our complete submission for the Duality AI Offroad Autonomy Segmentation Challenge. We developed a semantic segmentation model using DINOv2 backbone with a ConvNeXt-style segmentation head, achieving test IoU of **0.232** through systematic iterative optimization.

## 🏆 Final Results

- **Validation IoU**: 0.355
- **Test IoU**: 0.232
- **Best Val Dice Score**: 0.509
- **Pixel Accuracy**: 72.92%

## 📁 Repository Structure

```
project/
├── code/                           # All source code
│   ├── train_segmentation.py      # Training script with optimizations
│   ├── test_segmentation.py       # Testing/inference script
│   ├── inference.py                # Standalone inference on single images
│   └── requirements.txt            # Python dependencies
├── report/
│   └── report.docx                 # Complete hackathon report (8 pages)
└── results/                        # Training outputs and metrics
    ├── evaluation_metrics.txt      # Final training metrics
    ├── segmentation_head.pth       # Trained model weights
    ├── loss and pixel accuracy/    # Training curves
    ├── Overall result/             # Combined metrics visualization
    ├── train an vlidation IoU vs Epoch/
    └── Train Dice and Validation Dice vs Epoch/
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- Anaconda or Miniconda

### Installation

1. **Create environment**:
```bash
conda create -n segmentation python=3.8
conda activate segmentation
```

2. **Install dependencies**:
```bash
cd code
pip install -r requirements.txt
```

### Dataset Setup

Download the dataset from the [Duality AI Hackathon page](https://falconcloud.dualityai.com/) and organize as:

```
Offroad_Segmentation_Training_Dataset/
├── train/
│   ├── Color_Images/
│   └── Segmentation/
├── val/
│   ├── Color_Images/
│   └── Segmentation/
└── testImages/
    └── Color_Images/
```

Place the dataset folder in the parent directory of this project.

## 🎯 Usage

### Training

To train the model with our optimized configuration:

```bash
python train_segmentation.py
```

**Training Configuration:**
- Optimizer: AdamW (lr=5e-4, weight_decay=1e-4)
- Scheduler: CosineAnnealingLR (T_max=25)
- Epochs: 25
- Batch size: 4
- Backbone: DINOv2-small (frozen)
- Augmentation: Horizontal flip, ColorJitter(0.3)

**Outputs:**
- Model weights: `segmentation_head.pth`
- Training curves: `train_stats/` directory
- Metrics log: `evaluation_metrics.txt`

### Testing

To evaluate on test set:

```bash
python test_segmentation.py
```

**Outputs:**
- Per-class IoU scores
- Mean IoU
- Predicted segmentation masks (optional)

### Inference on Single Images

```bash
python inference.py --image path/to/image.png --model segmentation_head.pth
```

## 🧪 Model Architecture

**Backbone**: DINOv2-small
- Pre-trained Vision Transformer
- 384-dimensional embeddings
- Frozen during training

**Segmentation Head**: ConvNeXt-inspired
- Stem: 7×7 conv (384 → 128 channels)
- Processing block: Depthwise 7×7 conv + 1×1 conv
- Classifier: 1×1 conv to 10 classes
- Bilinear upsampling to original resolution

## 📊 Optimization Journey

Our systematic approach improved validation IoU from **0.294 → 0.355**:

### Iteration 1: Extended Training
- Change: 10 → 25 epochs
- Gain: +0.017 IoU

### Iteration 2: Optimizer Switch ⭐
- Change: SGD → AdamW
- Gain: +0.043 IoU (largest improvement!)

### Iteration 3: LR Scheduling
- Change: Added CosineAnnealingLR
- Result: Smoother convergence, final val IoU 0.355

## 🔬 Augmentation Experiments

We tested aggressive augmentation to address the validation-test gap (0.355 → 0.232):

| Strategy | Augmentation | Test IoU | Change |
|----------|-------------|----------|--------|
| **Baseline** | Standard | **0.232** | Baseline |
| Aggressive | ColorJitter(0.5), Blur, Grayscale, Rotation(20°) | 0.163 | -30% ❌ |
| Moderate | ColorJitter(0.3), Rotation(10°) | 0.176 | -24% ❌ |

**Key Finding**: Aggressive augmentation degraded performance by destroying domain-specific features (desert color palette). We submit the baseline model.

## 📈 Per-Class Performance

| Class | Training % | Test IoU |
|-------|-----------|----------|
| Sky | 36.85% | 0.9463 ✅ |
| Landscape | 16.65% | 0.5606 ✅ |
| Dry Grass | 11.96% | 0.1818 |
| Rocks | 6.17% | 0.1050 |
| Dry Bushes | 6.58% | 0.0951 |
| Trees | 3.35% | 0.0851 ⚠️ |
| Logs | 5.17% | 0.0653 ⚠️ |
| Ground Clutter | 6.23% | 0.0599 ⚠️ |
| Lush Bushes | 5.30% | 0.0156 ❌ |

**Pattern**: Strong on large, color-consistent classes (Sky, Landscape). Struggles with small, texture-heavy objects (Trees, Bushes, Logs).

## 🧠 Key Learnings

1. **Optimizer matters**: AdamW gave +0.043 IoU over SGD for transformer backbones
2. **Domain shift is real**: 34% val-test gap despite both being desert environments
3. **Augmentation can harm**: Destroyed domain-specific features when too aggressive
4. **Systematic experimentation**: Understanding failures is as valuable as successes

## 📖 Report

Complete documentation available in `report/report.docx`:
- Iterative optimization process
- Augmentation experiments
- Failure analysis
- Training curves and visualizations
- Lessons learned and future work

## 🔧 Troubleshooting

**Out of memory during training**:
- Reduce batch size in `train_segmentation.py`
- Use smaller input resolution

**DINOv2 download issues**:
- Model downloads automatically from torch.hub
- Requires internet connection on first run
- Cached locally after first download

**Missing dataset error**:
- Verify dataset path in scripts
- Check folder structure matches expected layout

## 💡 Future Improvements

1. **Domain-adaptive augmentation**: Preserve color palette, vary lighting
2. **Multi-scale features**: Better small object detection
3. **Larger backbone**: DINOv2-base or -large
4. **Class-specific strategies**: Different augmentation per class
5. **Progressive training**: Start minimal, gradually increase augmentation

## 🙏 Acknowledgments

- **Duality AI** for organizing the hackathon and providing synthetic data
- **Facebook Research** for DINOv2 backbone
- **Challenge dataset** generated using FalconCloud digital twin platform

## 📝 Citation

If using this code, please cite:

```
Duality AI Offroad Autonomy Segmentation Challenge 2026
Model: DINOv2-small + ConvNeXt Segmentation Head
Final Test IoU: 0.232
```

## 📞 Contact

For questions about this submission, please reach out via the Duality AI Discord channel.

---

**Competition**: Duality AI Offroad Autonomy Segmentation Challenge  
**Date**: February 2026  
**Final Submission**: Optimized Baseline Model (Test IoU: 0.232)
