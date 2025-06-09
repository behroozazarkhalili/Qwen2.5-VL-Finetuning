# Fine-Tuning Vision Language Model (Qwen2.5-VL-7B) with TRL

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/behroozazarkhalili/Qwen2.5-VL-Finetuning/blob/main/fine_tuning_vlm_trl_clean.ipynb)

## Overview

This repository contains a comprehensive tutorial for fine-tuning the Qwen2.5-VL-7B Vision Language Model using the Hugging Face TRL (Transformer Reinforcement Learning) library. The tutorial demonstrates how to adapt a pre-trained vision-language model for visual question-answering tasks using the ChartQA dataset.

## Features

- **Vision Language Model Fine-tuning**: Step-by-step guide to fine-tune Qwen2.5-VL-7B
- **Hugging Face Ecosystem**: Utilizes TRL library for efficient training
- **ChartQA Dataset**: Specialized for chart and visual data understanding
- **Google Colab Ready**: One-click execution in Google Colab environment
- **Resource Optimized**: Designed for A100 GPU environments

## Quick Start

### Option 1: Google Colab (Recommended)
Click the "Open in Colab" badge above to run the notebook directly in Google Colab with GPU support.

### Option 2: Local Setup
1. Clone the repository:
```bash
git clone https://github.com/behroozazarkhalili/Qwen2.5-VL-Finetuning.git
cd Qwen2.5-VL-Finetuning
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Open the Jupyter notebook:
```bash
jupyter notebook fine_tuning_vlm_trl_clean.ipynb
```

## Requirements

- **GPU**: A100 or equivalent (for optimal performance)
- **Memory**: Minimum 16GB GPU memory recommended
- **Python**: 3.8+
- **CUDA**: Compatible version for PyTorch

## Dataset

The tutorial uses the **ChartQA dataset**, which contains:
- Chart images with corresponding questions
- Visual reasoning tasks
- Question-answering pairs for training

## Model Architecture

- **Base Model**: Qwen2.5-VL-7B
- **Task**: Visual Question Answering
- **Training Method**: Fine-tuning with TRL
- **Output**: Specialized VLM for chart understanding

## Author

**Behrooz Azarkhalili**

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Related Resources

- [Hugging Face TRL Documentation](https://huggingface.co/docs/trl)
- [Qwen2.5-VL Model](https://huggingface.co/Qwen/Qwen2.5-VL-7B)
- [ChartQA Dataset](https://huggingface.co/datasets/HuggingFaceM4/ChartQA) 