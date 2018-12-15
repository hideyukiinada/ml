#!/usr/bin/env python
"""
Weight and bias initialization parameters

__author__ = "Hide Inada"
__copyright__ = "Copyright 2018, Hide Inada"
__license__ = "The MIT License"
__email__ = "hideyuki@gmail.com"
"""
import os
import logging

import h5py
import numpy as np
from pathlib import Path

log = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOGLEVEL", "DEBUG"))


class WeightParameter():
    """
    Store parameters used in initializing NeuralNetwork's weight & bias matrices.
    """

    NORMAL = 0 # Normal distribution
    UNIFORM = 1 # Uniform distribution
    ZERO = 2 # Fill with 0s
    LAYER_UNIT_COUNT_PROPORTIONAL = 3 # sqrt(1/(number of units in previous layer)
    LAYER_UNIT_COUNT_PROPORTIONAL2 = 4 # sqrt(2/(number of units in previous layer)

    def __init__(self, init_type=NORMAL, mean=0.0, stddev=1.0, multiplier=1.0):
        """
        Initialize instance variables used to configure weights and biases.

        Parameters
        ----------
        init_type: int
            Type of numbers to populate weights and biases
        mean: float
            Mean of distribution
        stddev: float
            Stddev of distribution
        multiplier: float
            A number to multiply the random number distribution
        """
        self.init_type = init_type
        self.mean = mean
        self.stddev = stddev
        self.multiplier = multiplier
