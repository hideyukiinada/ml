#!/usr/bin/env python
"""
Neural Network implementation.

Credit
------
I primarily learned the neural network algorithm from below sources:
[ICL] Imperial College London, Mathematics for Machine Learning Specialization, https://www.coursera.org/specializations/mathematics-machine-learning
[DA] deeplearning.ai, Deep Learning Specialization, https://www.coursera.org/specializations/deep-learning
[IG] Ian Goodfellow, Yoshua Bengio and Aaron Courville, Deep Learning, MIT Press, 2016
bibtex entry for the [IG] above:
@book{Goodfellow-et-al-2016,
    title={Deep Learning},
    author={Ian Goodfellow and Yoshua Bengio and Aaron Courville},
    publisher={MIT Press},
    note={\\url={http://www.deeplearningbook.org}},
    year={2016}
}

Adam optimization code is based on the below paper:
[DK] Diederik P. Kingma and Jimmy Lei Ba, ADAM: A METHOD FOR STOCHASTIC OPTIMIZATION, https://arxiv.org/abs/1412.6980, 2015

I crafted the code from scratch based on the algorithm that I learned, so please email me if you see an error
in this code

* Model class was inspired by the Keras SequenceModel and Keras layer.

__author__ = "Hide Inada"
__copyright__ = "Copyright 2018, Hide Inada"
__license__ = "The MIT License"
__email__ = "hideyuki@gmail.com"
"""
import sys
import os
import logging

import numpy as np

from .activationfunction import ActivationFunction as af
from .costfunction import CostFunction as cf
from .optimizer import Optimizer as opt
from .weightpersistence import WeightPersistence as wp

log = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))  # Change the 2nd arg to INFO to suppress debug logging


class Layer():
    """
    Holds meta-information of a single layer of neural network
    """

    def __init__(self, num_units, activation=af.RELU):
        self._num_units = num_units
        self._activation = activation

    def num_units(self):
        """
        Provides access to number of units on the layer.

        Returns
        -------
        out: int
            Number of units on the layer
        """
        return self._num_units

    def activation(self):
        """
        Provides access to the type of the activation function for the layer.

        Returns
        -------
        out: ActivationFunction constant
            Type of the activation
        """
        return self._activation


class Model():
    """
    Container for holding information for multiple layers
    """

    def __init__(self, num_input):
        """
        Initialize the model.
        """

        self._layers = list()
        self._layers.append(Layer(num_units=num_input, activation=af.NONE))

    def add(self, layer):
        """
        Add a single layer to the model.

        Parameters
        ----------
        layer: Layer
            A single layer of the network
        """
        self._layers.append(layer)

    def layers(self):
        """
        Provides access to the layers list.

        Returns
        -------
        layers: list
            list of layers

        Notes:
        This includes the input as a layer at index 0.
        """
        return self._layers


class NeuralNetwork():
    """
    Neural Network

    Variables used in the class
    ---------------------------
    weight: matrix
        Weight stored in matrix associated with each layer.
        Size of the matrix is unit count of the previous layer multiplied by the unit count of the current layer
    bias: vector
        Bias stored in matrix associated with each layer.
        Size of the vector is unit unit count of the current layer
    z: matrix
        Affine transformation applied to previous layer's activation
        z = a.T w.  In this code, a is a matrix with each row holding all parameters for a single point.
        Therefore, z = a w is used.
    a: matrix
        Output of an activation function with z as an input.  Activation functions include sigmoid and ReLU.

    Notes
    -----
    Layer number starts with 0 with 0 being the input.  However, input is not counted as a layer following a convention.
    Weight and bias only exist for layers 1 and above.
    """

    def _init_weight_forward_prop_data_list(self):
        """
        Allocate list for weight, bias, z, a, gradient of weight, gradient of bias.
        Allocate matrix and vector as weight and bias for each layer.

        Notes
        -----
        With the exception of a[0] which is used to access input, all others have valid values only with indices
        greater than or equal to 1.
        """

        def list_with_n_elements(n):
            """
            Helper function to generate a list with n elements.
            The primary use for this is to instantiate a list with one item as our neural network uses
            1-based index for all except for accessing the input as layer 0 activation.


            Parameters
            ----------
            n: int
                Number of elements

            Returns
            -------
            out: list
                list with n elements.  All elements are set to None
            """
            return [None] * n

        self.weight = list_with_n_elements(1)
        self.gradient_weight = list_with_n_elements(1)
        self.bias = list_with_n_elements(1)
        self.gradient_bias = list_with_n_elements(1)
        self.z = list_with_n_elements(self.num_layers + 1)
        self.a = list_with_n_elements(self.num_layers + 1)

        # Create a list for holding references to moment vectors for ADAM
        if self.optimizer == opt.ADAM:
            self.mt_weight = list_with_n_elements(1)  # First moment vector for weight
            self.mt_bias = list_with_n_elements(1)  # First moment vector for bias
            self.vt_weight = list_with_n_elements(1)  # Second moment vector for weight
            self.vt_bias = list_with_n_elements(1)  # Second moment vector for bias

        # Allocate weight and bias for each layer
        for i in range(self.num_layers):
            num_units_this_layer = self.model.layers()[i + 1].num_units()
            num_units_prev_layer = self.model.layers()[i].num_units()

            # w initialization below is following the recommendation on http://cs231n.github.io/neural-networks-2/
            # min 100 to ensure that weights are small when the number of units is a few.
            # w = np.random.randn(num_units_prev_layer, num_units_this_layer) * (min(1.0/100.0, math.sqrt(2.0/num_units_prev_layer)))
            # w = np.random.randn(num_units_prev_layer, num_units_this_layer) / 100.0
            w = np.random.randn(num_units_prev_layer, num_units_this_layer) * 0.1
            self.weight.append(w)
            self.gradient_weight.append(np.zeros(w.shape))

            # b = np.random.rand(1, num_units_this_layer) * 1e-10  # See discussions on [IG] p.173
            b = np.zeros((1, num_units_this_layer))  # FIXME TMP
            self.bias.append(b)
            self.gradient_bias.append(np.zeros(b.shape))

            if self.optimizer == opt.ADAM:
                self.mt_weight.append(np.zeros(w.shape))
                self.mt_bias.append(np.zeros(b.shape))
                self.vt_weight.append(np.zeros(w.shape))
                self.vt_bias.append(np.zeros(b.shape))

    def __init__(self, model, cost_function=cf.MEAN_SQUARED_ERROR, learning_rate=0.001, optimizer=opt.BATCH,
                 optimizer_settings=None, batch_size=1):
        """
        Initialize the class.

        Parameters
        ----------
        num_units_per_layer: list
            Number of units for each layer including input
        learning_rate: float
            Controls the speed of gradient descent.  At the end of each each epoch,
            gradient is multiplied with the learning rate before subtracted from weight.
        optimizer: int
            Optimizer type
        Optimizer settings: Optimizer parameters object
            Optimizer parameters
        batch_size: int
            Size of the batch
        """

        self.model = model
        self.optimizer = optimizer
        self.optimizer_settings = optimizer_settings
        self.cost_function = cost_function
        self.learning_rate = learning_rate
        self.num_layers = len(model.layers()) - 1  # To exclude the input layer
        # self.num_units_per_layer = num_units_per_layer
        self._init_weight_forward_prop_data_list()
        self.dataset_size = 0  # Dataset size to be initialized in fit()
        self.batch_size = batch_size

    def _forward_prop(self, x, output_layer_index=-1):
        """
        Forward propagation

        Parameters
        ----------
        x: ndarray
            Input data
        output_layer_index: int
            1-based layer index to output.  If set to -1, forward prop proceeds to the last layer.
            This is used to output the activation of an intermediate layer.

        Returns
        -------
        out: ndarray
            Predicted values
        """
        a = x  # For the first layer, assign input as the activation
        self.a[0] = a

        if output_layer_index != -1:
            loop_count = output_layer_index
        else:
            loop_count = self.num_layers

        for i in range(loop_count):
            a = self._forward_prop_one_layer(a, i + 1)

        return (a)

    # forward prop
    def _forward_prop_one_layer(self, a_prev, current_layer_index):
        """
        Forward propagate one layer by applying affine transformation and activation

        Parameters
        ----------
        a_prev: ndarray
            Previous layer's activation
        current_layer_index: int
            Index of current layer. Index 0 is input.
        activation: str
            Activation function
        """
        # Affine transformation
        z = a_prev.dot(self.weight[current_layer_index]) + self.bias[current_layer_index]

        self.z[current_layer_index] = z

        # Activation
        if self.model.layers()[current_layer_index].activation() == af.SIGMOID:
            a = af.sigmoid(z)
        elif self.model.layers()[current_layer_index].activation() == af.RELU:
            a = af.relu(z)
        elif self.model.layers()[current_layer_index].activation() == af.LEAKY_RELU:
            a = af.leaky_relu(z)
        else:
            a = af.none(z)

        self.a[current_layer_index] = a

        return (a)

    def predict(self, x):
        """
        Predict based on the input x

        Parameters
        ----------
        x: ndarray
            Input data

        Returns
        -------
        out: ndarray
            Predicted values
        """
        return self._forward_prop(x)

    def predict_intermediate(self, x, output_layer_index):
        """
        Feedforward up to the layer specified.

        Parameters
        ----------
        x: ndarray
            Input data
        output_layer_index: int
            1-based layer index to output.  If set to -1, forward prop proceeds to the last layer.
            This is used to output the activation of an intermediate layer.

        Returns
        -------
        out: ndarray
            Predicted values
        """
        return self._forward_prop(x, output_layer_index)

    def _backprop(self, x, y, y_hat):
        """
        Backpropagation

        x: ndarray
            Input
        y: ndarray
            Observations
        y_hat: ndarray
            Predicted values

        Notes
        -----
        Gradient is calculated using the multivariable chain rule.
        A variable 'derivative_cumulative' carries this over from the last layer all the way to the first layer.
        """

        dj_wrt_a = self.derivative_j_wrt_a(y, y_hat, cost_function=self.cost_function)
        derivative_cumulative = dj_wrt_a

        for i in range(self.num_layers):
            derivative_cumulative = self._backprop_one_layer(derivative_cumulative, self.num_layers - i)

        self._update_weight()

    def _backprop_one_layer(self, derivative_cumulative, layer_index):
        """
        Backpropagate one layer

        derivative_cumulative: ndarray
            Accumulated derivative from the last layer in the network.
            At the entry point of this method, the shape of the array
            is the same as the shape of the layer (dataset size by the number of units for the layer).
        layer_index: int
            Current layer index
        dataset_size: int
            Size of input

        Returns
        -------
        derivative_cumulative: ndarray
            Updated accumulated derivative from the last layer

        """
        # Derivative of a with respect to z
        if self.model.layers()[layer_index].activation() == af.SIGMOID:
            pa_pz = self.sigmoid_derivative_with_z(layer_index)
        elif self.model.layers()[layer_index].activation() == af.RELU:
            pa_pz = self.relu_derivative_with_z(layer_index)
        elif self.model.layers()[layer_index].activation() == af.LEAKY_RELU:
            pa_pz = self.leaky_relu_derivative_with_z(layer_index)
        else:
            pa_pz = self.none_derivative_with_z(layer_index)

        cumulative_derivative_to_z = derivative_cumulative * pa_pz
        # Note that the shape is still the same as current layer.

        # Derivative of z with respect to weight
        pz_pw = self.partial_z_wrt_partial_w(layer_index)
        cumulative_derivative_to_w = pz_pw.T.dot(cumulative_derivative_to_z)
        # At this point, shape of cumulative_derivative_to_w is the same as the weight of this layer
        cumulative_derivative_to_w /= self.dataset_size
        self.gradient_weight[layer_index] = cumulative_derivative_to_w

        # Derivative of z with respect to bias
        pz_pb = self.partial_z_wrt_partial_b(layer_index)
        cumulative_derivative_to_b = np.sum(cumulative_derivative_to_z * pz_pb, axis=0)
        # At this point, shape of cumulative_derivative_to_b is the same as the bias of this layer

        cumulative_derivative_to_b /= self.dataset_size
        self.gradient_bias[layer_index] = cumulative_derivative_to_b

        # Derivative of z with respect to previous layer's activation
        pz_pa_prev = self.partial_z_wrt_partial_a_prev(layer_index)

        cumulative_derivative_to_a_prev = cumulative_derivative_to_z.dot(pz_pa_prev.T)
        return cumulative_derivative_to_a_prev  # Shape is the same as the previous layer's activation.

    def _update_weight(self):
        """
        Update weight and bias of the network by subtracting the gradient of weight and bias multiplied by the learning
        rate.
        """
        for i in range(self.num_layers):

            layer_index = self.num_layers - i

            if self.optimizer == opt.ADAM:
                beta1 = self.optimizer_settings.beta1
                beta2 = self.optimizer_settings.beta2
                beta1_to_t = self.optimizer_settings.beta1_to_t
                beta2_to_t = self.optimizer_settings.beta2_to_t
                epsilon = self.optimizer_settings.epsilon

                self.mt_weight[layer_index] = beta1 * self.mt_weight[layer_index] + \
                                              (1 - beta1) * self.gradient_weight[layer_index]

                self.vt_weight[layer_index] = beta2 * self.vt_weight[layer_index] + \
                                              (1 - beta2) * self.gradient_weight[layer_index] ** 2

                self.mt_bias[layer_index] = beta1 * self.mt_bias[layer_index] + \
                                            (1 - beta1) * self.gradient_bias[layer_index]

                self.vt_bias[layer_index] = beta2 * self.vt_bias[layer_index] + \
                                            (1 - beta2) * self.gradient_bias[layer_index] ** 2

                mt_weight_hat = self.mt_weight[layer_index] / (1.0 - beta1_to_t)
                vt_weight_hat = self.vt_weight[layer_index] / (1.0 - beta2_to_t)

                mt_bias_hat = self.mt_bias[layer_index] / (1.0 - beta1_to_t)
                vt_bias_hat = self.vt_bias[layer_index] / (1.0 - beta2_to_t)

                self.weight[layer_index] -= self.learning_rate * mt_weight_hat / (
                        np.sqrt(vt_weight_hat) + epsilon)
                self.bias[layer_index] -= self.learning_rate * mt_bias_hat / (
                        np.sqrt(vt_bias_hat) + epsilon)

            else:
                self.weight[layer_index] -= self.learning_rate * self.gradient_weight[layer_index]
                self.bias[layer_index] -= self.learning_rate * self.gradient_bias[layer_index]

        if self.optimizer == opt.ADAM:
            self.optimizer_settings.beta1_to_t *= self.optimizer_settings.beta1
            self.optimizer_settings.beta2_to_t *= self.optimizer_settings.beta2

    def fit(self, x, y, epochs, verbose=True, interval=1):
        """
        Train the model

        Parameters
        ----------
        x: ndarray
            Input
        y: ndarray
            Observations
        epochs: int
            Number of epochs to iterate
        verbose: bool
            Show the cost for each epoch
        interval: int
            Number of epochs to show the cost if verbose is set to true
        """

        def process_verbose(y, y_hat, batch_index, batch_loop_count, epoch_index, epoch_size, dataset_size):
            """
            Helper function to output cost at regular intervals

            Parameters
            ----------
            y: ndarray
                Ground-truth
            y_hat: ndarray
                Predicted values
            batch_index: int
                0-based batch index within the current epoch
            batch_loop_count: int
                Total number of batch loops within a epoch
            epoch_index: int
                0-based epoch index
            epoch_size: int
                Number of epochs
            dataset_size: int
                Dataset size in this batch
            """
            cost = -1
            if self.cost_function == cf.CROSS_ENTROPY:
                cost = cf.mean_cross_entropy(y, y_hat)
            elif self.cost_function == cf.MEAN_SQUARED_ERROR:
                cost = cf.mean_squared_error(y, y_hat)

            if (self.batch_size >= 32):
                print("[Epoch %d/%d - Batch %d/%d] Cost: %.07f. Batch size: %d" %
                      (epoch_index + 1, epoch_size, batch_index + 1, batch_loop_count, cost, dataset_size))
            else:
                if (batch_index % 100 == 0):
                    print("[Epoch %d/%d - Batch %d/%d] Cost: %.07f. Batch size: %d" %
                          (epoch_index + 1, epoch_size, batch_index + 1, batch_loop_count, cost, dataset_size))

        if self.optimizer in [opt.SGD, opt.ADAM]:

            if self.optimizer == opt.ADAM:
                self.optimizer_settings.beta1_to_t = self.optimizer_settings.beta1
                self.optimizer_settings.beta2_to_t = self.optimizer_settings.beta2

            for i in range(epochs):

                next_k = 0
                loop_count = int(x.shape[0] / self.batch_size)  # for m = 5, batch_size = 2, this results in [0, 1]
                for j in range(loop_count):

                    self.dataset_size = self.batch_size
                    k = j * self.dataset_size
                    next_k = k + self.dataset_size
                    x_sub = x[k:next_k]
                    y_sub = y[k:next_k]
                    y_hat = self._forward_prop(x_sub)

                    if verbose:
                        process_verbose(y_sub, y_hat, j, loop_count, i, epochs, self.dataset_size)

                    self._backprop(x_sub, y_sub, y_hat)

                # remainder
                self.dataset_size = x.shape[0] - next_k
                if self.dataset_size > 0:
                    k = next_k
                    x_sub = x[k:k + self.dataset_size]
                    y_sub = y[k:k + self.dataset_size]
                    y_hat = self._forward_prop(x_sub)

                    if verbose:
                        process_verbose(y_sub, y_hat, j + 1, loop_count + 1, i, epochs, self.dataset_size)

                    self._backprop(x_sub, y_sub, y_hat)

        else:  # Batch gradient
            self.dataset_size = x.shape[0]

            for i in range(epochs):
                y_hat = self._forward_prop(x)

                if verbose:
                    if self.cost_function == cf.CROSS_ENTROPY:
                        cost = cf.mean_cross_entropy(y, y_hat)
                    elif self.cost_function == cf.MEAN_SQUARED_ERROR:
                        cost = cf.mean_squared_error(y, y_hat)

                    if ((i + 1) % interval == 0):
                        print("[%d/%d epochs] Cost: %.07f" % (i + 1, epochs, cost))

                self._backprop(x, y, y_hat)

    # Partial derivatives
    def derivative_j_wrt_a(self, y, y_hat, cost_function):
        """
        Calculate the derivative of cost with respect to y hat (or the activation of the last layer).

        Parameters
        ----------
        y: ndarray
            Observations
        y_hat: ndarray
            Predicted values

        Returns
        -------
        out: ndarray
            The partial derivative of cost with respect to y hat

        Raises
        ------
        ValueError
            If unsupported cost function is specified
        """
        if cost_function == cf.CROSS_ENTROPY:
            d = cf.d_cross_entropy(y, y_hat)
        elif cost_function == cf.MEAN_SQUARED_ERROR:
            d = cf.d_squared_error(y, y_hat)
        else:
            raise ValueError("Unsupported cost function")

        return (d)  # we will multiply by 1/m later

    def sigmoid_derivative_with_z(self, layer_index):
        """
        Calculate the derivative of activation using the value of z used in forward prop.

        Parameters
        ----------
        layer_index: int
            Layer index to be used in retrieving the value of z

        Returns
        -------
        out: ndarray
            Partial derivative of a with respect to z
        """
        return af.d_sigmoid(self.z[layer_index])

    def relu_derivative_with_z(self, layer_index):
        """
        Calculate the derivative of activation using the value of z used in forward prop.

        Parameters
        ----------
        layer_index: int
            Layer index to be used in retrieving the value of z

        Returns
        -------
        out: ndarray
            Partial derivative of a with respect to z
        """
        return af.d_relu(self.z[layer_index])

    def leaky_relu_derivative_with_z(self, layer_index):
        """
        Calculate the derivative of activation using the value of z used in forward prop.

        Parameters
        ----------
        layer_index: int
            Layer index to be used in retrieving the value of z

        Returns
        -------
        out: ndarray
            Partial derivative of a with respect to z
        """
        return af.d_leaky_relu(self.z[layer_index])

    def none_derivative_with_z(self, layer_index):
        """
        Dummy derivative function to return 1.

        Parameters
        ----------
        layer_index: int
            Layer index to be used in retrieving the value of z

        Returns
        -------
        out: ndarray
            Partial derivative of a with respect to z
        """
        return af.d_none(self.z[layer_index])

    def partial_z_wrt_partial_w(self, current_layer_index):
        """
        Calculate the partial derivative of z with respect to weight.

        Parameters
        ----------
        layer_index: int
            Layer index to be used in retrieving the value of z

        Returns
        -------
        out: ndarray
            Partial derivative of z with respect to weight

        Notes
        -----
        Since z = w a_prev + b, the partial derivative is a_prev.
        """
        a_prev = self.a[current_layer_index - 1]
        return a_prev

    def partial_z_wrt_partial_a_prev(self, current_layer_index):
        """
        Calculate the partial derivative of z with respect to activation of the last layer.

        Parameters
        ----------
        layer_index: int
            Layer index to be used in retrieving the value of z

        Returns
        -------
        out: ndarray
            Partial derivative of z with respect to activation of the last layer

        Notes
        -----
        Since z = w a_prev + b, the partial derivative is z.
        """
        w = self.weight[current_layer_index]
        return w

    def partial_z_wrt_partial_b(self, current_layer_index):
        """
        Calculate the partial derivative of z with respect to bias.

        Parameters
        ----------
        layer_index: int
            Layer index. Not currently used.

        Returns
        -------
        out: ndarray
          Partial derivative of z with respect to bias

        Notes
        -----
        Since z = w a_prev + b, the partial derivative is 1.
        """
        return 1

    # Loading and saving weights
    def load(self, file_path):
        """
        Load the matrix weight from a file specified with file_path.

        Parameters
        ---------
        file_path: Pathlib.path
            Path to save the weights

        Returns
        -------
        Weights of the model

        Raises
        ------
        File not found
        """
        weight, bias = wp.load(file_path)

        self.weight = weight
        self.bias = bias

    def save(self, file_path):

        """
        Save the matrix weight in a file specified by file_path.

        Parameters
        ----------
        file_path: Pathlib.path
            Path to save the weights
        """

        # index corresponds to a layer.  layer 0 does not have weight, but wp is aware of this.
        wp.save(file_path,
                {
                    "weight": self.weight,
                    "bias": self.bias
                }
                )
