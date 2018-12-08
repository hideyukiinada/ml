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
from numba import jit
import numpy as np

log = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))

@jit(nopython=True)
def _sigmoid(z):
    """
    Calculates a sigmoid function.
    This function was taken out of the ActivationFunction class to enable JIT.

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

class ActivationFunction():
    NONE = 0
    RELU = 1
    SIGMOID = 2
    LEAKY_RELU = 3

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
        return _sigmoid(z)

    @staticmethod
    @jit(nopython=True)
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
        return _sigmoid(z) * (1.0 - _sigmoid(z))

    @staticmethod
    @jit(nopython=True)
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

        m = np.zeros(z.shape)
        np.place(m, z>=0, [1.0])
        return m

    @staticmethod
    def leaky_relu(z, alpha=0.2):
        """
        Calculates the Leaky ReLU function.

        Parameters
        -----------
        z: ndarray
            Input
        alpha: float
            if z is negative, output = z multiplied by alpha

        Returns
        --------
        out: ndarray

            Output of a Leaky ReLU function for input z
        """
        alpha_vals = np.full(z.shape, z*alpha)
        m = np.maximum(z, alpha_vals)
        return m

    @staticmethod
    def d_leaky_relu(z, alpha=0.2):
        """
        Calculates the derivative of a Leaky ReLU function.

        Parameters
        -----------
        z: ndarray
            Input
        alpha: float
            if z is negative, output = alpha

        Returns
        --------
        out: ndarray
            Derivative of a Leaky ReLU function for input z and alpha
        """

        m = np.ones(z.shape)
        np.place(m, z<0, [alpha])
        return m