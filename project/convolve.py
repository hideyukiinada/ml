#!/usr/bin/env python
"""
Calculates convolution in ML terminology (or cross-correlation in math)


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


class Convolve():

    @staticmethod
    def _pad_matrix(m, kernel):
        """
        Zero-pad before convolution to keep the output matrix the same size as the input matrix.

        Parameters
        ----------
        m: ndarray
            Matrix
        k: ndarray
            Convolution kernel

        Returns
        -------
        out: ndarray
            Matrix padded with 0 along the edges.
        """

        # row padding
        diff = int(kernel.shape[0] / 2)
        row_pad_count_top = diff
        row_pad_count_bottom = diff

        # col padding
        diff = int(kernel.shape[1] / 2)
        col_pad_count_left = diff
        col_pad_count_right = diff

        # Zero-pad
        out = np.lib.pad(m,
                         ((row_pad_count_top, row_pad_count_bottom), (col_pad_count_left, col_pad_count_right)),
                         'constant', constant_values=((0, 0), (0, 0)))

        return out

    @staticmethod
    def _convolve2d(m, kernel, strides=(1, 1)):
        """

        Parameters
        ----------
        m: ndarray
            Matrix
        k: ndarray
            Convolution kernel
        strides: tuple
            Step size in each axis

        Returns
        -------
        out: ndarray
            Matrix after the convolution operation with the kernel

        Raises
        ------
        ValueError
            If kernel size is greater than m in any axis after padding
        """
        source_height = m.shape[0]
        source_width = m.shape[1]

        kernel_height = kernel.shape[0]
        kernel_width = kernel.shape[1]

        if kernel_height > source_height or kernel_width > source_width:
            raise ValueError("Kernel size is larger than the matrix")

        row_stride = strides[0]
        col_stride = strides[1]

        # (source_height - kernel_height)/strides[0] is how many steps you can go down.
        # + 1 to include the start position.
        target_height = int((source_height - kernel_height) / row_stride) + 1
        target_width = int((source_width - kernel_width) / col_stride) + 1

        m_out = np.zeros((target_height, target_width))
        # Convolve
        for i in range(target_height):
            for j in range(target_width):
                m_out[i, j] = (m[i * row_stride:i * row_stride + kernel_height,
                               j * col_stride:j * col_stride + kernel_width] * kernel).sum()

        return m_out

    @staticmethod
    def convolve2d(m, kernel, strides=(1, 1), padding=True):
        """

        Parameters
        ----------
        m: ndarray
            Matrix
        k: ndarray
            Convolution kernel
        strides: tuple
            Step size in each axis
        padding: bool
            True if m should be zero-padded before convolution.  This is to keep the output matrix the same size.
            False if no padding should be applied before convolution.

        Returns
        -------
        out: ndarray
            Matrix after the convolution operation with the kernel

        Raises
        ------
        ValueError
            If kernel size is greater than m in any axis after padding

        Notes
        -----
        A kernel with an even-number element in an axis will result in a bigger matrix than the original matrix.
        Use an odd-number matrix (3x3, 5x5) to generate the output that is the same size as the original.
        """
        if padding:
            m = Convolve._pad_matrix(m, kernel)

        return Convolve._convolve2d(m, kernel, strides=strides)
