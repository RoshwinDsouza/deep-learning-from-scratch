# Deep Learning Models from Scratch

## Project Overview

This project implements three fundamental deep learning models from scratch using Python and NumPy:

- Multilayer Perceptron (MLP) for classification
- Sparse Autoencoder for representation learning and anomaly detection
- Restricted Boltzmann Machine (RBM) for generative feature learning

The Fashion-MNIST dataset was used for all experiments.

No high-level deep learning libraries (TensorFlow/PyTorch) were used for model implementation. All forward propagation, backpropagation, and weight updates were implemented manually.

---

## Dataset

Dataset Used: Fashion-MNIST  
- Training Samples: 60,000  
- Testing Samples: 10,000  
- Image Size: 28 × 28  
- Classes: 10  

Images are flattened to 784-dimensional vectors and normalized to [0,1].

---

## Project Structure


mlp_autoencoder_rbm.py

## Requirements

- Python 3.x
- NumPy
- Matplotlib
- TensorFlow (only for loading Fashion-MNIST dataset)

Install dependencies:

pip install numpy matplotlib tensorflow


---

## How to Run

### 1. Run MLP (Classification)

python mlp.py


Output:
- Training loss printed per epoch
- Accuracy evaluation
- Loss curve plot

---

### 2. Run Autoencoder

python autoencoder.py



Output:
- Reconstruction loss per epoch
- Outlier detection count
- Reconstruction error trend

---

### 3. Run RBM

python rbm.py



Output:
- Reconstruction error per epoch
- RBM reconstruction error plot

---

## Hyperparameters Used

MLP:
- Hidden Units: 128
- Learning Rate: 0.1
- Epochs: 25

Autoencoder:
- Bottleneck Size: 32
- Loss: MSE + L1 sparsity

RBM:
- Hidden Units: 100
- Training: Contrastive Divergence (CD-1)
- Epochs: 10

---

## Key Observations

- ReLU improved convergence speed in MLP.
- Sparse Autoencoder enabled meaningful feature compression.
- RBM successfully learned probabilistic representations.
- All models showed stable decreasing loss curves.

---

## GitHub Repository

https://github.com/RoshwinDsouza/deep-learning-from-scratch

---

## Author

Roshwin Dsouza  
USN: NNM23IS152  
NMAM Institute of Technology
