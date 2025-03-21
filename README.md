# 🧠 Neural Network Image Classification Models

![AI Banner](https://img.shields.io/badge/AI-Image%20Classification-blue?style=for-the-badge&logo=pytorch&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-orange?style=for-the-badge&logo=pytorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.7%2B-yellow?style=for-the-badge&logo=python&logoColor=white)

A collection of neural network models for image classification tasks implemented using PyTorch.

## 📋 Project Overview

This repository contains implementations of various neural network architectures for image classification on popular datasets:

- **MNIST**: Handwritten digit recognition (0-9)
- **CIFAR-10**: Classification of 10 different object categories

The models range from simple fully-connected networks to more complex convolutional neural networks (CNNs).

## 🧰 Models

### MNIST Models

| Model | Description | Accuracy |
|-------|-------------|----------|
| [Simple NN](simplemodels/nmist.py) | Basic fully-connected neural network | ~98% |
| [CNN](simplemodels/cnn-nmist.py) | Convolutional neural network with batch normalization and dropout | ~99% |

### CIFAR-10 Model

| Model | Description | Accuracy |
|-------|-------------|----------|
| [CNN](simplemodels/cifar10.py) | Convolutional neural network with data augmentation | ~75% |

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- PyTorch
- torchvision
- matplotlib
- scikit-learn
- tqdm

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ai.git
cd ai

# Install required packages
pip install torch torchvision matplotlib scikit-learn tqdm
```

### Training and Evaluation

To train and evaluate a model:

```bash
# Train the MNIST simple model
python simplemodels/nmist.py

# Train the MNIST CNN model
python simplemodels/cnn-nmist.py

# Train the CIFAR-10 model
python simplemodels/cifar10.py
```

## 📊 Datasets

The datasets are automatically downloaded by the PyTorch/torchvision libraries when running the models.

- **MNIST**: 60,000 training images and 10,000 test images of handwritten digits.
- **CIFAR-10**: 50,000 training images and 10,000 test images across 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck).

## 🏗️ Project Structure

```
├── data/                   # Dataset directory
│   ├── cifar-10-batches-py/  # CIFAR-10 dataset
│   └── MNIST/              # MNIST dataset
└── simplemodels/           # Model implementations
    ├── cifar10.py          # CIFAR-10 CNN implementation
    ├── cnn-nmist.py        # MNIST CNN implementation
    └── nmist.py            # Simple MNIST model
```

## 🧩 Model Architectures

### Simple MNIST Model
- Input layer (784 neurons)
- Hidden layer (128 neurons with ReLU activation)
- Output layer (10 neurons)

### CNN MNIST Model
- Multiple convolutional layers with batch normalization
- Max pooling and dropout for regularization
- Fully connected layers with batch normalization
- Achieves high accuracy through advanced techniques

### CIFAR-10 CNN Model
- Three convolutional blocks with batch normalization
- Max pooling layers
- Fully connected layers with dropout
- Trained with data augmentation techniques

## 📈 Future Improvements

- Implement more advanced architectures (ResNet, DenseNet)
- Add visualization tools for model interpretation
- Add transfer learning examples
- Implement models for other datasets (ImageNet, CIFAR-100)

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

*Last Updated: March 21, 2025*