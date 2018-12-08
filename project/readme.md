This directory contains implementation of neural network (NeuralNetwork class) and helper classes.

# File list under the project directory
# Neural Network

| Summary| File name | Note |
|---|---|---|
| Main neural network code| neuralnetwork.py | Implementation of a plain-vanilla neural network built from scratch with numpy | |
| Activation functions| activationfunction.py | Implementation of activation functions for neural network |
| Convolution (cross-correlation) | convolve.py | Implementation of convolution. This is called convolution in ML terminology, but it is actually cross-correlation. |
| Cost functions | costfunction.py | Implementation of cost functions for neural network including cross-entropy & MSE |
| Optimizer | optimizer.py | Optimizer meta-data to be used with NeuralNetwork. Actual implementation is in NeuralNetwork. |
| Weight parameter | weightparameter.py | Parameters to be used in initializing neural network weights and biases. |
| Weight loading and saving | weightpersistence.py | Load and save weight from/to a HDF5 file. |

# K-means

| Summary| File name | Note |
|---|---|---|
| K-means | kmeans.py | Implementation of k-means clustering method. Notes: As the below Wikipedia article points out, there is no guarantee that this implementation finds the optimal clustering result. https://en.wikipedia.org/wiki/K-means_clustering |

