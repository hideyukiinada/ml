# General code for machine learning

# File list under the project directory

| File name | Note | |
|---|---|---|
| neuralnetwork.py | Implementation of a plain-vanilla neural network built from scratch with numpy | |
| activationfunction.py | Implementation of activation functions for neural network | |
| costfunction.py | Implementation of cost functions for neural network including cross-entropy & MSE | |
| kmeans.py | Implementation of k-means clustering method. Notes: As the below Wikipedia article points out, there is no guarantee that this implementation finds the optimal clustering result. https://en.wikipedia.org/wiki/K-means_clustering | |

# File list under the examples directory

| File| File name | Note | |
|---|---|---|---|
|K-means| kmeans_example | An example for k-means clustering with 5 clusters | ![sample](assets/images/k-means-demo.png)|
| NeuralNetwork class demo with batch gradient descent against MNIST| neural_network_mnist_example | Neural network example to identify MNIST digit data using the **batch gradient descent**. Accuracy as of [this rev]( https://github.com/hideyukiinada/ml/commit/5b9e4dca610791d5d9f21dd1890e1a27c3002c2a) is 91.0% against the MNIST test data that contains 10,000 sample data. | |
| NeuralNetwork class demo with SGD against MNIST|| neural_network_mnist_sgd_example | Neural network example to identify MNIST digit data using **SGD**. Accuracy as of [this rev]( https://github.com/hideyukiinada/ml/commit/1cfd9bb688b364309c8dda9cabdc41e72c512b7a) is 91.7% with 5 epochs against the MNIST test data that contains 10,000 sample data. | |
| NeuralNetwork class demo with Adam optimizer against MNIST|| neural_network_mnist_adam_example | Neural network example to identify MNIST digit data using **Adam**. Accuracy as of [this rev]( https://github.com/hideyukiinada/ml/commit/62eba5630239a368fa1895569b669ed290e899e8) is 90.3% with 2 epochs against the MNIST test data that contains 10,000 sample data. | |
| NeuralNetwork class demo for logical AND operation|| neural_network_logistic_regression_example | Neural network logistic regression example code | |
|Activation functions demo| activation_example | Activation function example code | |
|Cost functions demo| cost_example | Cost function example code | |


