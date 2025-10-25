# EMTSF: Extraordinary Mixture of SOTA Models for Time Series Forecasting

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Paper: [EMTSF — Full paper (PDF)](https://ebooks.iospress.nl/Download/Pdf)


## 📝 Abstract

The immense success of the Transformer architecture in Natural Language Processing has led to its adoption in Time Series Forecasting (TSF), where superior performance has been shown. However, a recent important paper questioned their effectiveness by demonstrating that a simple single layer linear model outperforms Transformer-based models. This was soon shown to be not as valid, by a better transformer-based model termed PatchTST. More recently, TimeLLM demonstrated even better results by reprogramming i.e., repurposing a Large Language Model (LLM) for the TSF domain. Again, a follow up paper challenged this by demonstrating that removing the LLM component or replacing it with a basic attention layer in fact yields better performance.

One of the challenges in forecasting is the fact that TSF data favors the more recent past, and is sometimes subject to unpredictable events. Based upon these recent insights in TSF, we propose a **Mixture of Experts (MoE) framework**. Our method combines state-of-the-art (SOTA) models including **xLSTM**, **enhanced Linear models**, **PatchTST**, and **minGRU** among others. This set of complimentary and diverse models for TSF are integrated in a Transformer-based MoE architecture. Our results on standard TSF benchmarks demonstrate better results surpassing all current TSF models, including those based on recent MoE frameworks.

## 🎯 Key Features

- **Mixture of Experts Architecture**: Combines multiple SOTA models (xLSTM, minGRU, PatchTST, Enhanced Linear) for superior forecasting
- **Advanced Gating Mechanism**: Transformer-based attention layer for intelligent expert selection
- **Flexible Configuration**: Support for multiple forecasting horizons (96, 192, 336, 720)
- **Comprehensive Dataset Support**: Works with 14+ standard benchmarks (ETT, Weather, Electricity, Traffic, etc.)
- **Reversible Instance Normalization (RevIN)**: Built-in support for improved generalization
- **Distributed Training**: Multi-GPU support for efficient training

## 📊 Supported Datasets

The framework supports the following standard TSF benchmarks:

- **ETT (Electricity Transformer Temperature)**: `ettm1`, `ettm2`, `etth1`, `etth2`
- **Weather**: Weather forecasting data
- **Electricity**: Electricity consumption data
- **Traffic**: Road occupancy rates
- **Solar**: Solar power production
- **Exchange**: Exchange rate data
- **Illness**: Illness cases data
- **PEMS**: Traffic datasets (`PEMS03`, `PEMS04`, `PEMS07`, `PEMS08`)

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- PyTorch 2.0 or higher
- CUDA (optional, for GPU support)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/muslehal/EMTSF.git
cd EMTSF
```

2. Install dependencies:
```bash
pip install torch torchvision torchaudio
pip install numpy pandas scikit-learn matplotlib
pip install einops timm
```

3. Download datasets:
   - Place your datasets in the appropriate directories as configured in `datautils.py`
   - Update the `root_path` in `datautils.py` to match your local paths

## 💻 Usage

### Training Individual Expert Models

Before training the MoE model, you need to train individual expert models:

```bash
# Train model_a (Linear model)
python main.py --dset ettm1 --model_type model_a --context_points 512 --target_points 96 --n_epochs 100

# Train model_b (xLSTM model)
python main.py --dset ettm1 --model_type model_b --context_points 512 --target_points 96 --n_epochs 100

# Train model_c (minGRU model)
python main.py --dset ettm1 --model_type model_c --context_points 512 --target_points 96 --n_epochs 100

# Train model_d (PatchTST model)
python main.py --dset ettm1 --model_type model_d --context_points 512 --target_points 96 --n_epochs 100
```

### Training the MoE Model

After training all expert models:

```bash
python main.py --dset ettm1 --model_type EMTSF --context_points 512 --target_points 96 --n_epochs 50
```

### Using the Training Script

For automated training across multiple forecasting horizons:

```bash
# Run training for multiple target points (192, 336, 720)
bash script.sh -d ettm1 -e 100

# With testing
bash script.sh -d ettm1 -e 100 --test
```

### Key Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--dset` | str | `ettm1` | Dataset name |
| `--context_points` | int | `512` | Input sequence length |
| `--target_points` | int | `96` | Forecasting horizon |
| `--batch_size` | int | `64` | Batch size |
| `--n_epochs` | int | `100` | Number of training epochs |
| `--lr` | float | `1e-3` | Learning rate |
| `--model_type` | str | `based_model` | Model type to train |
| `--patch_len` | int | `32` | Patch length for PatchTST |
| `--stride` | int | `16` | Stride between patches |
| `--n_layers` | int | `6` | Number of Transformer layers |
| `--d_model` | int | `128` | Model dimension |
| `--dropout` | float | `0.2` | Dropout rate |
| `--revin` | int | `1` | Use Reversible Instance Normalization |

## 🏗️ Model Architecture

### EMTSF (Mixture of Experts)

The EMTSF model architecture consists of:

1. **Expert Models**:
   - **Model A**: Enhanced Linear model with decomposition
   - **Model B**: xLSTM-based model for long-term dependencies
   - **Model C**: minGRU for efficient sequence modeling
   - **Model D**: PatchTST for patch-based attention

2. **Gating Network**: Transformer-based attention mechanism that learns to weight expert predictions

3. **Integration Layer**: Combines expert outputs using learned gating weights

```
Input → [Expert A, Expert B, Expert C, Expert D] → Transformer Attention → Gating Weights → Weighted Combination → Output
```

### Expert Models Details

#### Model A (Enhanced Linear)
- Series decomposition (trend + seasonal)
- Channel-independent processing
- RevIN normalization

#### Model B (xLSTM)
- mLSTM and sLSTM blocks
- Configurable depth and dimension
- Efficient long-sequence modeling

#### Model C (minGRU)
- Lightweight gating mechanism
- Fast inference
- Memory-efficient

#### Model D (PatchTST)
- Patch-based tokenization
- Self-attention over patches
- Channel-independent or channel-mixing options

## 📈 Results

![Results Overview](https://github.com/user-attachments/assets/41fcd172-1485-4998-a6ba-3e973c9edaa9)
![Performance Comparison](https://github.com/user-attachments/assets/b96c9dc6-0e12-4942-9c70-7c7ca3ecb71f)

Our EMTSF model achieves state-of-the-art performance across multiple benchmarks, outperforming:
- Traditional LSTM/GRU models
- Transformer-based models (Autoformer, FEDformer, etc.)
- Recent MoE frameworks
- LLM-based approaches (TimeLLM)

## 📁 Project Structure

```
EMTSF/
├── main.py                 # Main training script
├── models.py              # Model architectures (EMTSF, model_a-d)
├── datautils.py           # Dataset loading utilities
├── lr_scheduler.py        # Learning rate scheduling
├── StandardNorm.py        # Normalization utilities
├── script.sh              # Automated training script
├── src/
│   ├── learner.py         # Training loop implementation
│   ├── data/              # Data loading modules
│   ├── models/            # Additional model components
│   └── callback/          # Training callbacks
├── xlstm1/                # xLSTM implementation
├── minGRU_pytorch/        # minGRU implementation
└── layers/                # Custom layer implementations
```

## 🔧 Advanced Configuration

### Custom Dataset

To add a custom dataset, modify `datautils.py`:

```python
elif params.dset == 'your_dataset':
    root_path = '/path/to/your/dataset'
    size = [params.context_points, 0, params.target_points]
    dls = DataLoaders(
        datasetCls=Dataset_Custom,
        dataset_kwargs={
            'root_path': root_path,
            'data_path': 'your_data.csv',
            'features': params.features,
            'scale': True,
            'size': size,
            'use_time_features': params.use_time_features
        },
        batch_size=params.batch_size,
        workers=params.num_workers,
    )
```

### Hyperparameter Tuning

Key hyperparameters to tune:
- `d_model`: Model dimension (affects capacity)
- `n_layers`: Number of Transformer layers
- `patch_len` and `stride`: For PatchTST expert
- `lr`: Learning rate (start with 1e-3)
- Expert weights in MoE (adjustable via gating network)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@article{emtsf2024,
  title={EMTSF: Extraordinary Mixture of SOTA Models for Time Series Forecasting},
  author={[Your Name]},
  journal={arXiv preprint},
  year={2024}
}
```

## 🙏 Acknowledgments

This project builds upon several excellent works:
- [PatchTST](https://github.com/yuqinie98/PatchTST)
- [xLSTM](https://github.com/NX-AI/xlstm)
- [minGRU](https://github.com/lucidrains/minGRU-pytorch)
- [Autoformer](https://github.com/thuml/Autoformer)

## 📧 Contact

For questions and feedback, please open an issue on GitHub.

---

**Note**: Make sure to update dataset paths in `datautils.py` before running experiments.
