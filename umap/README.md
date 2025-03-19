# Overview

Example of how to use UMAP for visualization of latent space.
1. A neural network learns to classify MNIST digits
2. Visualize the latent space using UMAP during training
3. ```umap_example.py``` is an example of a linear encoder and ```umap_cnn_example.py``` is an example of a CNN encoder with dropout and batch normalization

## Installation
You need to install the required Python packages if you haven't already imported the conda environment ```env.yml``` .

```bash
pip install umap-learn
pip install torch torchvision
```
