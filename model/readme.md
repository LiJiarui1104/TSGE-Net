# TSGE-Net

## Requirements

- Python >= 3.8
- PyTorch >= 2.0
- CUDA 12.7
- [timm](https://github.com/huggingface/pytorch-image-models)

Install dependencies:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install timm
```

### Configuration

| Hyperparameter     | Value               |
|--------------------|---------------------|
| Backbone           | VAN-B0              |
| Optimizer          | AdamW               |
| Learning Rate      | 1×10⁻⁴             |
| Weight Decay       | 0.05                |
| Batch Size         | 64                  |
| Epochs             | 200                 |
| LR Schedule        | Cosine Annealing    |
| Loss Function      | Cross-Entropy       |
| Input Size         | 224 × 224           |

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.
