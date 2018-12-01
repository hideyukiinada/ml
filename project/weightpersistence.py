#!/usr/bin/env python
"""
Save and load weights to/from a file on the file system

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


class WeightPersistence():

    @staticmethod
    def save(file_path, weight_dict):
        """
        Save the matrix weight in a file specified with file_path.

        Parameters
        ---------
        file_path: Pathlib.path
            Path to save the weights

        weight_dict: dict

            Each dict entry contains:
            "<weight type": list of weights

            For example,
                "weight": self.weight[1:,],
                "bias": self.bias[1:,]
        """
        log.debug("In WeightPersistence.save().  File to save: %s" % (file_path))
        with h5py.File(file_path, "w") as f:

            for k,v in weight_dict.items():
                layer_count = len(v)

                for i in range(layer_count):
                    if (i == 0): # Note matrix is stored in 0-based index in this list
                        continue
                    tag = k + str(i)
                    log.debug("Saving %s" % (tag))
                    matrix = v[i]
                    f.create_dataset(tag, data=matrix)

    @staticmethod
    def load(file_path):
        """
       Load the matrix weight from a file specified with file_path.

        Parameters
        ---------
        file_path: Pathlib.path
            Path to save the weights

        Returns
        -------
        Weights of the model
        """

        with h5py.File(file_path, "r") as f:

            weight_list = [None]
            bias_list = [None]

            # read weight
            i = 1
            while True:
                name = "weight" + str(i)
                if name not in f:
                    break
                log.debug("Loaded %s" % (name))
                w = f[name]
                weight_list.append(np.array(w))
                i += 1

            # read bias
            i = 1
            while True:
                name = "bias" + str(i)
                if name not in f:
                    break
                b = f[name]
                bias_list.append(np.array(b))
                log.debug("Loaded %s (shape:%s" % (name, str(b.shape)))

                i += 1

        return weight_list, bias_list
