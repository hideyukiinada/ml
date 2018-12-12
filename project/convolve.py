#!/usr/bin/env python
"""
Calculates convolution in ML terminology (or cross-correlation in math)

This class utilizes numba JIT so functions are organized into two groups:
1) Internal functions that are mostly JIT-enabled.
2) Class methods to be called from outside.

__author__ = "Hide Inada"
__copyright__ = "Copyright 2018, Hide Inada"
__license__ = "The MIT License"
__email__ = "hideyuki@gmail.com"
"""
from numba import jit
import os
import logging

import numpy as np

log = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))


@jit(nopython=True)
def _convolve2d_jit(m, kernel, strides, target_height, target_width):
    row_stride = strides[0]
    col_stride = strides[1]
    kernel_height = kernel.shape[0]
    kernel_width = kernel.shape[1]

    m_out = np.zeros((target_height, target_width))

    # Convolve
    for i in range(target_height):
        for j in range(target_width):
            m_out[i, j] = (m[i * row_stride:i * row_stride + kernel_height,
                           j * col_stride:j * col_stride + kernel_width] * kernel).sum()

    return m_out


@jit(nopython=True)
def _calculate_target_matrix_dimension(m, kernel, paddings, strides):
    """
    Calculate the target matrix dimension.

    Parameters
    ----------
    m: ndarray
        2d Matrix
    k: ndarray
        2d Convolution kernel
    paddings: tuple
        Number of padding in (row, height) on one side.
        If you put 2 padding on the left and 2 padding on the right, specify 2.
    strides: tuple
        Step size in (row, height)

    Returns
    -------
    out: tuple
        Tuple containing (number of rows, number of columns)

    Raises
    ------
    ValueError
        If kernel size is greater than m in any axis after padding
    """
    source_height = m.shape[0]
    source_width = m.shape[1]

    padding_row = paddings[0]
    padding_column = paddings[1]

    kernel_height = kernel.shape[0]
    kernel_width = kernel.shape[1]

    if kernel_height > (source_height + padding_row) or kernel_width > (source_width + padding_column):
        raise ValueError("Kernel size is larger than the matrix")

    row_stride = strides[0]
    col_stride = strides[1]

    # (source_height - kernel_height)/strides[0] is how many steps you can go down.
    # + 1 to include the start position.
    target_height = int((source_height + padding_row - kernel_height) / row_stride) + 1
    target_width = int((source_width + padding_column - kernel_width) / col_stride) + 1

    return (target_height, target_width)


@jit(nopython=True)
def _calculate_padding(kernel):
    """
    Calculate padding size to keep the output matrix the same size as the input matrix.

    Parameters
    ----------
    k: ndarray
        Convolution kernel

    Returns
    -------
    out: tuple of tuple of int
        (row_pad_count_top, row_pad_count_bottom), (col_pad_count_left, col_pad_count_right)
    """

    # row padding
    diff = int(kernel.shape[0] / 2)
    row_pad_count_top = diff
    row_pad_count_bottom = diff

    # col padding
    diff = int(kernel.shape[1] / 2)
    col_pad_count_left = diff
    col_pad_count_right = diff

    return ((row_pad_count_top, row_pad_count_bottom), (col_pad_count_left, col_pad_count_right))


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

    (row_pad_count_top, row_pad_count_bottom), (
        col_pad_count_left, col_pad_count_right) = _calculate_padding(kernel)

    # Zero-pad
    return np.lib.pad(m,
                      ((row_pad_count_top, row_pad_count_bottom), (col_pad_count_left, col_pad_count_right)),
                      'constant', constant_values=((0, 0), (0, 0)))


class Convolve():

    @staticmethod
    def pad_matrix(m, kernel):
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
        return _pad_matrix(m, kernel)

    @staticmethod
    def _convolve2d(m, kernel, strides=(1, 1)):
        """
        Convolve a 2D matrix with a kernel.

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
        (target_height, target_width) = _calculate_target_matrix_dimension(m, kernel, (0, 0), strides)

        return _convolve2d_jit(m, kernel, strides, target_height, target_width)

    @staticmethod
    def convolve2d(m, kernel, strides=(1, 1), use_padding=True):
        """
        Convolve a 2D matrix with a kernel with padding if specified by the caller.

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

        if use_padding:
            m = _pad_matrix(m, kernel)

        return Convolve._convolve2d(m, kernel, strides=strides)

    @staticmethod
    def convolve_tensor(input_data_tensor, kernel_tensor, strides=(1, 1), use_padding=True):
        """
        Convolve stacked 2D matrices with the stacked 2D kernels.
        Sizes of volume from matrix and kernel need to match.

        Parameters
        ----------
        input_data_tensor: ndarray
            Stacked 2D Matrix of shape (row count, col count, input channels)
        kernel_tensor: ndarray
            Stacked 2D convolution kernel of shape (row cont, col count, input channels)
        strides: tuple
            Step size in each axis
        padding: bool
            True if m should be zero-padded before convolution.  This is to keep the output matrix the same size.
            False if no padding should be applied before convolution.

        Returns
        -------
        out: ndarray
            2D matrix that is the element-wise sum of layers of convoluted matrices.

        Raises
        ------
        ValueError
            If kernel size is greater than m in any axis after padding, or if the size of volume do not match between
            the matrix and the kernel.

        """

        input_channel_num = input_data_tensor.shape[2]
        kernel_input_channel_num = kernel_tensor.shape[2]

        if input_channel_num != kernel_input_channel_num:
            raise ValueError("Number of input channels do not match between the matrix and the kernel.")

        if use_padding:
            (row_pad_count_top, row_pad_count_bottom), (
                col_pad_count_left, col_pad_count_right) = _calculate_padding(kernel_tensor[:, :, 0])
            row_pads = row_pad_count_top + row_pad_count_bottom
            col_pads = col_pad_count_left + col_pad_count_right
        else:
            row_pads = 0
            col_pads = 0

        (target_height, target_width) = _calculate_target_matrix_dimension(input_data_tensor[:, :, 0],
                                                                           kernel_tensor[:, :, 0],
                                                                           (row_pads, col_pads), strides)
        target_tensor = np.zeros((target_height, target_width, input_channel_num))

        for i in range(input_channel_num):
            target_tensor[:, :, i] = Convolve.convolve2d(input_data_tensor[:, :, i], kernel_tensor[:, :, i],
                                                         strides=strides,
                                                         use_padding=use_padding)

        target_matrix = target_tensor.sum(axis=2)  # sum along the channels

        return target_matrix

    @staticmethod
    def convolve_tensor_multi_channel(input_data_tensor, kernel_tensor, bias=None, strides=(1, 1), use_padding=True):
        """
        Convolve stacked 2D matrices with the stacked 2D kernels.
        Sizes of volume from matrix and kernel need to match.

        Parameters
        ----------
        input_data_tensor: ndarray
            Stacked 2D Matrix of shape (row count, col count, input channels)
        kernel_tensor: ndarray
            Stacked 2D convolution kernel of shape (row count, col count, input channels, output channels)
        bias: ndarray
            Bias that is applied to each element after convolution of shape (1, output channels).
            There is 1 bias for each output channel.
        strides: tuple
            Step size in each axis
        padding: bool
            True if m should be zero-padded before convolution.  This is to keep the output matrix the same size.
            False if no padding should be applied before convolution.

        Returns
        -------
        target_tensor: ndarray
            Tensor.

        Raises
        ------
        ValueError
            If kernel size is greater than m in any axis after padding, or if the size of volume do not match between
            the matrix and the kernel.

        """

        if bias is None:
            if use_padding:
                paddings = ((kernel_tensor.shape[0] // 2)*2, (kernel_tensor.shape[1] // 2)*2)
            else:
                paddings = (0, 0)

            (target_height, target_width) = _calculate_target_matrix_dimension(input_data_tensor[:, :, 0],
                                                                               kernel_tensor[:, :, 0, 0], paddings,
                                                                               strides)
            bias = np.zeros((target_height, target_width, kernel_tensor.shape[3]))

        kernel_output_channel_num = kernel_tensor.shape[3]
        strides = strides
        use_padding = use_padding

        target_tensor = None
        out_shape = None

        output = list()
        for i in range(kernel_output_channel_num):
            unbiased_out = Convolve.convolve_tensor(input_data_tensor, kernel_tensor[:, :, :, i], strides=strides,
                                                    use_padding=use_padding)
            bias_to_add = bias[:, :, i]
            out_2d = unbiased_out + bias_to_add

            out_shape = out_2d.shape
            h = out_shape[0]
            w = out_shape[1]

            out_2d = out_2d.reshape(h, w, 1)
            output.append(out_2d)

        target_tensor = np.concatenate(output, axis=2)

        combined_shape = list()
        combined_shape = combined_shape + list(out_shape)
        combined_shape.append(kernel_output_channel_num)

        return target_tensor.reshape(combined_shape)

    @staticmethod
    def convolve_tensor_dataset(input_data_tensor, kernel_tensor, bias=None, strides=(1, 1), use_padding=True):
        """
        Convolve the dataset with the 2D conv kernels.

        Parameters
        ----------
        input_data_tensor: ndarray
            Training sample (ow count, col count, input channels)
        kernel_tensor: ndarray
            Stacked 2D convolution kernel of shape (row count, col count, input channels, output channels)
        bias: ndarray
            Bias that is applied to each element after convolution. There is 1 bias for each output channel.
        strides: tuple
            Step size in each axis
        padding: bool
            True if m should be zero-padded before convolution.  This is to keep the output matrix the same size.
            False if no padding should be applied before convolution.

        Returns
        -------
        target_tensor: ndarray
            Tensor.

        Raises
        ------
        ValueError
            If kernel size is greater than m in any axis after padding, or if the size of volume do not match between
            the matrix and the kernel.

        """
        sample_size = input_data_tensor.shape[0]
        strides = strides
        use_padding = use_padding

        target_tensor = None
        out_shape = None

        output = list()
        for i in range(sample_size):
            out_2d = Convolve.convolve_tensor_multi_channel(input_data_tensor[i], kernel_tensor, bias, strides=strides,
                                                            use_padding=use_padding)
            out_shape = out_2d.shape
            output.append(out_2d)

            target_tensor = np.concatenate(output)

        combined_shape = list()
        combined_shape.append(sample_size)
        combined_shape = combined_shape + list(out_shape)

        return target_tensor.reshape(combined_shape)
