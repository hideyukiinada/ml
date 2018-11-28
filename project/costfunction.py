#!/usr/bin/env python
"""
Calculates cost including cross-entropy (logistic regression cost)

Reference
---------
https://en.wikipedia.org/wiki/Cross_entropy

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


class CostFunction():
    MEAN_SQUARED_ERROR = 1
    CROSS_ENTROPY = 2

    @staticmethod
    def mean_cross_entropy(y, y_hat, epsilon=1e-10):
        """
        Calculates mean cross-entropy (logistic regression cost)

        Paramters
        ---------
        y: ndarray
            Ground-truth
        y_hat: ndarray
            Predictions
        epsilon: float
            A padding to ensure {y_hat: 0 < y_hat < 1}

        Returns
        -------
        out: float
            Mean cost
        """
        m = y.shape[0]
        return (1.0 / m) * CostFunction.cross_entropy(y, y_hat, epsilon).sum()

    @staticmethod
    def cross_entropy(y, y_hat, epsilon=1e-10):
        """
        Calculates cross-entropy (logistic regression cost) for each prediction

        Parameters
        ---------
        y: ndarray
            Ground-truth
        y_hat: ndarray
            Predictions
        epsilon: float
            A padding to ensure {y_hat: 0 < y_hat < 1}

        Returns
        -------
        out: ndarray
            Cost
        """
        # Tweak y_hat to avoid 0 or 1 which cannot be passed to the cost function
        y_hat_adj = np.clip(y_hat, a_min=epsilon, a_max=(1 - epsilon))

        cost = -y * np.log(y_hat_adj) - (1.0 - y) * np.log(1.0 - y_hat_adj)

        return cost

    @staticmethod
    def d_cross_entropy(y, y_hat, epsilon=1e-10):
        """
        Calculates the partial derivative of cross-entropy (logistic regression cost) for each prediction with respect to
        y hat.

        Parameters
        ---------
        y: ndarray
            Ground-truth
        y_hat: ndarray
            Predictions

        Returns
        -------
        out: ndarray
            Derivative of cost for each prediction
        """

        # y_hat_adj = np.clip(y_hat, a_min=epsilon, a_max=(1 - epsilon))
        y_hat_adj = y_hat
        d = - ( (y / y_hat_adj) - (1.0 - y) / (1.0 - y_hat_adj) )
        return d

    @staticmethod
    def mean_squared_error(y, y_hat, divide_by_two_m=False):
        """
        Calculates mean squared error for each sample data

        Paramters
        ---------
        y: ndarray
            Ground-truth
        y_hat: ndarray
            Predictions
        divide_by_two_m: bool
            Divide by the dataset size x 2 instead of the dataset size

        Returns
        -------
        out: float
            Mean squared error
        """
        m = y.shape[0]
        if divide_by_two_m:
            m *= 2
        return (1 / m) * CostFunction.squared_error(y, y_hat).sum()

    @staticmethod
    def squared_error(y, y_hat):
        """
        Calculates squared error for each prediction

        Parameters
        ---------
        y: ndarray
            Ground-truth
        y_hat: ndarray
            Predictions

        Returns
        -------
        out: ndarray
            Cost
        """
        cost = (y_hat - y) ** 2

        return cost

    @staticmethod
    def d_squared_error(y, y_hat):
        """
        Calculates the partial derivative of squared error cost for each prediction with respect to
        y hat.

        Parameters
        ---------
        y: ndarray
            Ground-truth
        y_hat: ndarray
            Predictions

        Returns
        -------
        out: ndarray
            Derivative of cost for each prediction
        """
        d = 2 * (y_hat - y)
        return d
