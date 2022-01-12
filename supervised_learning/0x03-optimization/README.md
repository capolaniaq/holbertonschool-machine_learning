# 0x03. Optimization

Author Carlos Andres Polania

## Learning Objectives

-   What is a hyperparameter?
-   How and why do you normalize your input data?
-   What is a saddle point?
-   What is stochastic gradient descent?
-   What is mini-batch gradient descent?
-   What is a moving average? How do you implement it?
-   What is gradient descent with momentum? How do you implement it?
-   What is RMSProp? How do you implement it?
-   What is Adam optimization? How do you implement it?
-   What is learning rate decay? How do you implement it?
-   What is batch normalization? How do you implement it?

## Tasks

### 0. Normalization Constants

Write the function  `def normalization_constants(X):`  that calculates the normalization (standardization) constants of a matrix:

-   `X`  is the  `numpy.ndarray`  of shape  `(m, nx)`  to normalize
    -   `m`  is the number of data points
    -   `nx`  is the number of features
-   Returns: the mean and standard deviation of each feature, respectively
### 1. Normalize

Write the function  `def normalize(X, m, s):`  that normalizes (standardizes) a matrix:

-   `X`  is the  `numpy.ndarray`  of shape  `(d, nx)`  to normalize
    -   `d`  is the number of data points
    -   `nx`  is the number of features
-   `m`  is a  `numpy.ndarray`  of shape  `(nx,)`  that contains the mean of all features of  `X`
-   `s`  is a  `numpy.ndarray`  of shape  `(nx,)`  that contains the standard deviation of all features of  `X`
-   Returns: The normalized  `X`  matrix

### 2. Shuffle Data
Write the function `def shuffle_data(X, Y):` that shuffles the data points in two matrices the same way:

### 3. Mini-Batch

Write the function `def train_mini_batch(X_train, Y_train, X_valid, Y_valid, batch_size=32, epochs=5, load_path="/tmp/model.ckpt", save_path="/tmp/model.ckpt"):` that trains a loaded neural network model using mini-batch gradient descent:

### 4. Moving Average

Write the function `def moving_average(data, beta):` that calculates the weighted moving average of a data set:

### 5. Momentum

Write the function `def update_variables_momentum(alpha, beta1, var, grad, v):` that updates a variable using the gradient descent with momentum optimization algorithm:
### 6. Momentum Upgraded

Write the function `def create_momentum_op(loss, alpha, beta1):` that creates the training operation for a neural network in `tensorflow` using the gradient descent with momentum optimization algorithm:

### 7. RMSProp

Write the function `def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):` that updates a variable using the RMSProp optimization algorithm:
### 8. RMSProp Upgraded
Write the function `def create_RMSProp_op(loss, alpha, beta2, epsilon):` that creates the training operation for a neural network in `tensorflow` using the RMSProp optimization algorithm:

### 9. Adam
Write the function `def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):` that updates a variable in place using the Adam optimization algorithm:
### 10. Adam Upgraded

Write the function `def create_Adam_op(loss, alpha, beta1, beta2, epsilon):` that creates the training operation for a neural network in `tensorflow` using the Adam optimization algorithm:
### 11. Learning Rate Decay
Write the function `def learning_rate_decay(alpha, decay_rate, global_step, decay_step):` that updates the learning rate using inverse time decay in `numpy`:
### 12. Learning Rate Decay Upgraded
Write the function `def learning_rate_decay(alpha, decay_rate, global_step, decay_step):` that creates a learning rate decay operation in `tensorflow` using inverse time decay:
### 13. Batch Normalization
Write the function `def batch_norm(Z, gamma, beta, epsilon):` that normalizes an unactivated output of a neural network using batch normalization:
### 14. Batch Normalization Upgraded
Write the function `def create_batch_norm_layer(prev, n, activation):` that creates a batch normalization layer for a neural network in `tensorflow`:
### 15. Put it all together and what do you get?
Complete the script `15-model.py` to write the function `def model(Data_train, Data_valid, layers, activations, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, decay_rate=1, batch_size=32, epochs=5, save_path='/tmp/model.ckpt'):` that builds, trains, and saves a neural network model in `tensorflow` using Adam optimization, mini-batch gradient descent, learning rate decay, and batch normalization:
