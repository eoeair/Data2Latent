# Data2Latent
Data-to-Latent Exploration

# About

This is a demonstration of how neural networks use kernel functions to map data from the data space to the final latent space. 

What I will showcase is how deep learning models learn and represent data in multiple layers, which is the first step in pattern learning. 

To simplify, the MNIST dataset is used here.

# Download Mnist
[Training images](https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz),
[Training labels](https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz)

[Testing images ](https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz),
[Testing labels](https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz)

# Requirments

`flax mloader scikit-learn matplotlib`

# Tutorial

1. download original data in `data`
2. run`data-process.py`