# 0x07. Convolutional Neural Networks

Author: Carlos Andres Polania

## Learning Objectives


### General

-   What is a convolutional layer?
-   What is a pooling layer?
-   Forward propagation over convolutional and pooling layers
-   Back propagation over convolutional and pooling layers
-   How to build a CNN using Tensorflow and Keras


## Tasks


### 0. Convolutional Forward Prop

Write a function `def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):` that performs forward propagation over a convolutional layer of a neural network:

### 1. Pooling Forward Prop

Write a function `def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):` that performs forward propagation over a pooling layer of a neural network:

### 2. Convolutional Back Prop

Write a function `def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):` that performs back propagation over a convolutional layer of a neural network:

### 3. Pooling Back Prop

Write a function `def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):` that performs back propagation over a pooling layer of a neural network

### 4. LeNet-5 (Tensorflow 1)

Write a function `def lenet5(x, y):` that builds a modified version of the `LeNet-5` architecture using `tensorflow`:


### 5. LeNet-5 (Keras)
Write a function `def lenet5(X):` that builds a modified version of the `LeNet-5` architecture using `keras`:

### 6. Summarize Like a Pro

A common practice in the machine learning industry is to read and review journal articles on a weekly basis. Read and write a summary of Krizhevsky et. al.‘s 2012 paper  [ImageNet Classification with Deep Convolutional Neural Networks](https://intranet.hbtn.io/rltoken/hj0CacwoEVC2GY1StNsPlA "ImageNet Classification with Deep Convolutional Neural Networks"). Your summary should include:

-   **Introduction:**  Give the necessary background to the study and state its purpose.
-   **Procedures:**  Describe the specifics of what this study involved.
-   **Results:**  In your own words, discuss the major findings and results.
-   **Conclusion:**  In your own words, summarize the researchers’ conclusions.
-   **Personal Notes:**  Give your reaction to the study.

Your posts should have examples and at least one picture, at the top. Publish your blog post on Medium or LinkedIn, and share it at least on Twitter and LinkedIn.

When done, please add all URLs below (blog post, tweet, etc.)

Please, remember that these blogs must be written in English to further your technical ability in a variety of settings.