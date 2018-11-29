#!/usr/bin/env python
"""
Neural network optimizer definitions.

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


class Optimizer():
    BATCH = 0
    SGD = 1
    ADAM = 2

class OptimizerParameters():
    """
    Base class for storing optimizer parameters
    """
    def __init__(self):
        pass

class AdamOptimizer(OptimizerParameters):
    """
    Container for Adam optimizer hyper-parameters
    """
    def __init__(self, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """

        Parameters
        ----------
        beta1: float
            Exponential decay rate for the moment estimates
        beta2:
            Exponential decay rate for the moment estimates
        """
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta1_to_t = beta1
        self.beta2_to_t = beta2
        self.epsilon = epsilon

        super().__init__()