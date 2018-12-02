This directory contains example files related to this project.

# File list under the examples directory

## K-means

| Summary| File name | Note | 
|---|---|---|
|K-means| kmeans_example | An example for k-means clustering with 5 clusters. ![sample](../assets/images/k-means-demo.png)|

## Using NeuralNetwork class

| Summary| File name | Note | 
|---|---|---|
| NeuralNetwork class demo with batch gradient descent against MNIST| neural_network_mnist_example | Neural network example to identify MNIST digit data using the **batch gradient descent**. Accuracy as of [this rev]( https://github.com/hideyukiinada/ml/commit/5b9e4dca610791d5d9f21dd1890e1a27c3002c2a) is 91.0% against the MNIST test data that contains 10,000 sample data. |
| NeuralNetwork class demo with SGD against MNIST|neural_network_mnist_sgd_example | Neural network example to identify MNIST digit data using **SGD**. Accuracy as of [this rev]( https://github.com/hideyukiinada/ml/commit/1cfd9bb688b364309c8dda9cabdc41e72c512b7a) is 91.7% with 5 epochs against the MNIST test data that contains 10,000 sample data. |
| NeuralNetwork class demo with Adam optimizer against MNIST| neural_network_mnist_adam_example | Neural network example to identify MNIST digit data using **Adam**. Accuracy should be around 96% with 2 epochs against the MNIST test data that contains 10,000 sample data. (Tests were executed 3 times on 11/29/2018 and the result was 96.35%, 96.02% and 96.2% respectively) |
| NeuralNetwork class demo for logical AND operation| neural_network_logistic_regression_example | Neural network logistic regression example code |
| NeuralNetwork class demo for logical XOR operation| neural_network_xor_example | Neural network logistic regression example code |
|Activation functions demo| activation_example | Activation function example code |
|Cost functions demo| cost_example | Cost function example code |

