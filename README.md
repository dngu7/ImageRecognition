# ML Algorithm performance on image datasets

# Techniques
- Deep Learning: Convolutional Neural Network
- k-Nearest Neighbours
- SVM
- Decision Trees
- Boosting

# Performance
- Convolutional 2-Layer Neural Network achieved 99% accuracy on MNIST database and 41% accuracy on CIFAR-100
- Decision Trees performed the worst with 84% accuracy on MNIST and 14% accuracy on CIFAR-100

# Setup
Ensure you have python3.6 and following packages installed.
tensorflow >=1.12.0
numpy >= 1.16.1
panda >= 0.3.1
sklearn >= 0.13.3
matplotlib >=3.0.2

# Data
Download the MNIST and CIFAR-100 datasets and place into their respective data folders
MNIST: http://yann.lecun.com/exdb/mnist/
CIFAR-100: https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz

# Run
To run the algorithms, use the following arguments while in the directory

- argument 1: python
- argument 2: runner.py
- argument 3: [algorithm] = {"tree", "nn", "ada","bag","svm", "knear"}
- argument 4: [data] = {"mnist", "cifar"}

Example 1: Neural Network on the MNIST database
- python runner.py nn mnist

Example 2: Support Vector Machine on the CIFAR database
- python runner.py svm cifar

