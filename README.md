# Neural Network Implementations
This repository contains source code for the neural network (_NeuralNetwork_ class) that I implemented from scratch using numpy.  [Examples](https://github.com/hideyukiinada/ml/blob/master/examples) directory contains scripts that use NeuralNetwork class to solve specific problems including MNIST digit recognition and GAN (generative adversarial network). In addition [web_examples/mnist](https://github.com/hideyukiinada/ml/tree/master/web_examples/mnist) directory contains a sample webapp that you can use to recognize your own handwriting using the model that you train.
![screenshot](https://github.com/hideyukiinada/ml/blob/master/assets/images/mnist-webapp-example.png)

# Getting started
Have a look at [read me in examples](https://github.com/hideyukiinada/ml/blob/master/examples/readme.md) to get started.

# Supported optimizers
* Batch gradient descent
* Mini-batch gradient descent/SGD
* Adam

# Intended use
* Understand how neural network works
  - Forward propagation
  - Back propagation and gradient descent 
* Use as a playground to test various technologies in AI (e.g. new optimizer, new cost function, GAN). 

# Latest features
* Convolutional Neural Network - 
  The CNN-related code checked-in in this repo is still experimental.  If you try it and does not work with your parameters,   please create an issue.  Accuracy against MNIST digit data was 97.95% with 4 epochs.
  Backprop calculation is documented [here]( https://hideyukiinada.github.io/cnn_backprop_strides2.html).
