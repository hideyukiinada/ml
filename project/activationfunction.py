#!/usr/bin/env python
"""
Neural network activation functions.

__author__ = "Hide Inada"
__copyright__ = "Copyright 2018, Hide Inada"
__license__ = "The MIT License"
__email__ = "hideyuki@gmail.com"
"""
import os
import logging

import numpy as np

log = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))


class ActivationFunction():
    NONE = 0
    RELU = 1
    SIGMOID = 2

    @staticmethod
    def none(z):
        """
        No operation.

        Parameters
        ----------
        z: ndarray
            Input

        Returns
        -------
        out: ndarray
            Output the input z
        """
        return z

    @staticmethod
    def d_none(z):
        """
        Calculates the derivative of no operation for input z.

        Parameters
        ----------
        z: ndarray
            Input

        Returns
        -------
        out: int
            1
        """
        return 1.0

    @staticmethod
    def sigmoid(z):
        """
        Calculates a sigmoid function.

        Parameters
        ----------
        z: ndarray
            Input

        Returns
        -------
        out: ndarray
            Output of the sigmoid function for input z
        """
        return 1.0 / (1.0 + np.e ** (-z))

    @staticmethod
    def d_sigmoid(z):
        """
        Calculates the derivative of a sigmoid function for input z.

        Parameters
        ----------
        z: ndarray
            Input

        Returns
        -------
        out: ndarray
            Derivative of a sigmoid function for input z
        """
        return ActivationFunction.sigmoid(z) * (1.0 - ActivationFunction.sigmoid(z))

    @staticmethod
    def relu(z):
        """
        Calculates the ReLU function.

        Parameters
        -----------
        z: ndarray
            Input

        Returns
        --------
        out: ndarray

            Output of a ReLU function for input z
        """
        zero = np.zeros(z.shape)
        m = np.maximum(z, zero)
        return m

    @staticmethod
    def d_relu(z):
        """
        Calculates the derivative of a ReLU function.

        Parameters
        -----------
        z: ndarray
            Input

        Returns
        --------
        out: ndarray
            Derivative of a ReLU function for input z
        """
        zero = np.zeros(z.shape)
        m = np.maximum(1.0, zero)
        return m
