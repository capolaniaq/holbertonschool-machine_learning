# 0x06. Keras

Author: Carlos Andres Polania (capolaniaq@correo.udistrital.edu.co)

## Learning Objectives

### General

-   What is Keras?
-   What is a model?
-   How to instantiate a model (2 ways)
-   How to build a layer
-   How to add regularization to a layer
-   How to add dropout to a layer
-   How to add batch normalization
-   How to compile a model
-   How to optimize a model
-   How to fit a model
-   How to use validation data
-   How to perform early stopping
-   How to measure accuracy
-   How to evaluate a model
-   How to make a prediction with a model
-   How to access the weights/outputs of a model
-   What is HDF5?
-   How to save and load a model’s weights, a model’s configuration, and the entire model

## Tasks

### 0. Sequential

Write a function `def build_model(nx, layers, activations, lambtha, keep_prob):` that builds a neural network with the Keras library:


### 1. Input

Write a function `def build_model(nx, layers, activations, lambtha, keep_prob):` that builds a neural network with the Keras library:

### 2. Optimize

Write a function `def optimize_model(network, alpha, beta1, beta2):` that sets up Adam optimization for a keras model with categorical crossentropy loss and accuracy metrics:

### 3. One Hot

Write a function  `def one_hot(labels, classes=None):`  that converts a label vector into a one-hot matrix:

-   The last dimension of the one-hot matrix must be the number of classes
-   Returns: the one-hot matrix

### 4. Train

Write a function `def train_model(network, data, labels, batch_size, epochs, verbose=True, shuffle=False):` that trains a model using mini-batch gradient descent:

### 5. Validate

Based on `4-train.py`, update the function `def train_model(network, data, labels, batch_size, epochs, validation_data=None, verbose=True, shuffle=False):` to also analyze validaiton data:

### 6. Early Stopping

Based on `5-train.py`, update the function `def train_model(network, data, labels, batch_size, epochs, validation_data=None, early_stopping=False, patience=0, verbose=True, shuffle=False):` to also train the model using early stopping:

### 7. Learning Rate Decay

Based on  `6-train.py`, update the function  `def train_model(network, data, labels, batch_size, epochs, validation_data=None, early_stopping=False, patience=0, learning_rate_decay=False, alpha=0.1, decay_rate=1, verbose=True, shuffle=False):`  to also train the model with learning rate decay:

### 8. Save Only the Best

Based on `7-train.py`, update the function `def train_model(network, data, labels, batch_size, epochs, validation_data=None, early_stopping=False, patience=0, learning_rate_decay=False, alpha=0.1, decay_rate=1, save_best=False, filepath=None, verbose=True, shuffle=False):` to also save the best iteration of the model:

### 9. Save and Load Model

Write the following functions:

-   `def save_model(network, filename):`  saves an entire model:
    -   `network`  is the model to save
    -   `filename`  is the path of the file that the model should be saved to
    -   Returns:  `None`
-   `def load_model(filename):`  loads an entire model:
    -   `filename`  is the path of the file that the model should be loaded from
    -   Returns: the loaded model
### 10. Save and Load Weights

Write the following functions:

-   `def save_weights(network, filename, save_format='h5'):`  saves a model’s weights:
    -   `network`  is the model whose weights should be saved
    -   `filename`  is the path of the file that the weights should be saved to
    -   `save_format`  is the format in which the weights should be saved
    -   Returns:  `None`
-   `def load_weights(network, filename):`  loads a model’s weights:
    -   `network`  is the model to which the weights should be loaded
    -   `filename`  is the path of the file that the weights should be loaded from
    -   Returns:  `None`



