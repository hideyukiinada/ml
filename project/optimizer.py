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
