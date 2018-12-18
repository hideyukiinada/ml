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
def _convolve_cube_jit(m, kernel, strides):
    row_stride = strides[0]
    col_stride = strides[1]
    kernel_height = kernel.shape[0]
    kernel_width = kernel.shape[1]

    target_height = (m.shape[0] - kernel_height) // row_stride + 1
    target_width = (m.shape[1] - kernel_width) // col_stride + 1

    m_out = np.zeros((target_height, target_width))

    # Convolve
    for i in range(target_height):
        for j in range(target_width):
            a = m[i * row_stride:i * row_stride + kernel_height,
                j * col_stride:j * col_stride + kernel_width]
            b = kernel
            c = a * b
            d = c.sum()
            m_out[i, j] = d

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


def _pad_matrix_uniform(m, pad_count):
    """
    Add the same number of padding to both row and column before and after the matrix .
    Total number of padding applied is pad_count*2 in each axis.

    Parameters
    ----------
    m: ndarray
        Matrix
    pad_count: int
        Number of padding to be added

    Returns
    -------
    out: ndarray
        Matrix padded with 0 along the edges.
    """

    # Zero-pad
    return np.lib.pad(m,
                      ((pad_count, pad_count), (pad_count, pad_count)),
                      'constant', constant_values=((0, 0), (0, 0)))


@jit(nopython=True)
def _pad_cube(m, h_pad, w_pad):
    """
    Zero-pad a numpy array that has the shape of (h, w, channels) in h and w.

    Parameters
    ----------
    m: ndarray
        3d tensor to be padded.
    h_pad: int
        The number of padding in axis 0 on the top edge and bottom edge.
        Total number of padding for axis 0 is h_pad * 2.
    w_pad: int
        The number of padding in axis 0 on the left edge and right edge.
        Total number of padding for axis 1 is w_pad * 2.

    Returns
    -------
    out: ndarray
        3d tensor padded with 0 along the edges.
    """

    padded_tensor = np.zeros((m.shape[0] + h_pad * 2, m.shape[1] + w_pad * 2, m.shape[2]))

    padded_tensor[h_pad:-h_pad, w_pad:-w_pad, ] = m

    return padded_tensor


def _zero_interweave(m, pad_count):
    """
    Add the same number of padding row and column to m in each axis.

    For example, if you specify 1 in pad_count, the following matrix
    1 2
    3 4

    is transformed to

    1 0 2 0
    0 0 0 0
    3 0 4 0
    0 0 0 0

    Parameters
    ----------
    m: ndarray
        Matrix
    pad_count: int
        Number of padding row and column to be added

    Returns
    -------
    out: ndarray
        Matrix padded with 0 along the edges.
    """

    # Vertically expand
    m_shape_plus_1 = list(m.shape)  # (2, 2) to [2, 2]
    m_shape_plus_1.append(1)  # (2, 2, 1)

    m2 = m.reshape(m_shape_plus_1)

    # Create zero matrix to add
    zero_shape = (m2.shape[0], m2.shape[1] * pad_count, m2.shape[2])
    m_zero = np.zeros(zero_shape)

    # Add two matrices
    m3 = np.concatenate((m2, m_zero), axis=1)
    m3 = m3.reshape((m.shape[0] * (1 + pad_count), m.shape[1]))

    # Horizontally expand
    m = m3
    m_shape_plus_1 = list(m.shape)  # (4, 2) to [4, 2]
    m_shape_plus_1.append(1)  # (4, 2, 1)

    m2 = m.reshape(m_shape_plus_1)

    # Create zero matrix to add
    zero_shape = (m2.shape[0], m2.shape[1], m2.shape[2] * pad_count)
    m_zero = np.zeros(zero_shape)

    # Add two matrices
    m3 = np.concatenate((m2, m_zero), axis=2)
    m3 = m3.reshape((m.shape[0], m.shape[1] * (1 + pad_count)))

    return m3


class Convolve():

    @staticmethod
    def pad_cube(m, padding):
        """
        Zero-pad a numpy array that has the shape of (h, w, channels) in h and w.

        Parameters
        ----------
        m: ndarray
            3d tensor to be padded.
        padding: tuple
            Number of padding in (row, height) on one side.
            If you put 2 padding on the left and 2 padding on the right, specify 2.

        Returns
        -------
        out: ndarray
            3d tensor padded with 0 along the edges.
        """
        return _pad_cube(m, int(padding[0]), int(padding[1]))

    @staticmethod
    def zero_interweave(m, pad_count):
        """
        Add the same number of padding row and column to m in each axis.

        For example, if you specify 1 in pad_count, the following matrix
        1 2
        3 4

        is transformed to

        1 0 2 0
        0 0 0 0
        3 0 4 0
        0 0 0 0

        Parameters
        ----------
        m: ndarray
            Matrix
        pad_count: int
            Number of padding row and column to be added

        Returns
        -------
        out: ndarray
            Matrix padded with 0 along the edges.
        """
        return _zero_interweave(m, pad_count)

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

    def pad_matrix_uniform(m, pad_count):
        """
        Add the same number of padding to both row and column before and after the matrix .
        Total number of padding applied is pad_count*2 in each axis.

        Parameters
        ----------
        m: ndarray
            Matrix
        pad_count: int
            Number of padding to be added

        Returns
        -------
        out: ndarray
            Matrix padded with 0 along the edges.
        """
        return _pad_matrix_uniform(m, pad_count)

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
    def convolve_cube(m, kernel, strides=(1, 1)):
        """
        Convolve a 3D tensor with a 3D kernel.

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
            2D matrix after the convolution operation with the kernel

        Raises
        ------
        ValueError
            If kernel size is greater than m in any axis after padding
        """

        return _convolve_cube_jit(m, kernel, strides)

    @staticmethod
    def convolve_two_datasets(input_data_tensor, kernel_data_tensor, strides=(1, 1), use_padding=True):
        """
        Convolve two datasets with second data acting a kernel.
        This is to be used for backprop to calculate the gradient of a kernel.

        Parameters
        ----------
        input_data_tensor: ndarray
            Input data (e.g. previous layer's activation).
            It has the shape (dataset size, row count, col count, input channels)
        kernel_data_tensor: ndarray
            Second input data (e.g. partial derivative of loss to this layer's Z).
            It has the shape (dataset size, row count, col count, input channels)
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
        input_channels_1 = input_data_tensor.shape[3]
        input_channels_2 = kernel_data_tensor.shape[3]
        target_tensor = None
        out_shape = None

        k = kernel_data_tensor

        output_i = list()
        for i in range(sample_size):
            output_j = list()
            for j in range(input_channels_1):
                output_k = list()
                for k in range(input_channels_2):
                    input_1 = input_data_tensor[i, :, :, j]
                    input_2 = kernel_data_tensor[i, :, :, k]

                    out_2d = Convolve.convolve2d(input_1, input_2, strides=strides,
                                                 use_padding=use_padding)
                    out_shape = out_2d.shape
                    out_shape2 = list(out_shape)  # add 2 more axes
                    out_shape3 = out_shape2 + [1, 1]
                    out_2d = out_2d.reshape(out_shape3)

                    output_k.append(out_2d)

                output_tmp = np.concatenate(output_k, axis=3)  # should be (h, w, 1, input_channel_2)
                output_j.append(output_tmp)

            output_tmp = np.concatenate(output_j, axis=2)  # should be (h, w, input_channel_1, input_channel_2)
            output_i.append(output_tmp)

        sample_shape = output_i[0].shape
        output = np.concatenate(output_i, axis=0)  # should be (m, h, w, input_channel_1, input_channel_2)

        final_shape = list()
        final_shape.append(sample_size)
        final_shape += sample_shape
        output = output.reshape(final_shape)

        return output

    @staticmethod
    def convolve_two_datasets_calc_mean(input_data_tensor, kernel_data_tensor, strides=(1, 1), use_padding=True):
        """
        Convolve two datasets with second data acting a kernel and calculate the mean.
        This is to be used for backprop to calculate the gradient of a kernel.

        Parameters
        ----------
        input_data_tensor: ndarray
            Input data (e.g. previous layer's activation).
            It has the shape (dataset size, row count, col count, input channels)
        kernel_data_tensor: ndarray
            Second input data (e.g. partial derivative of loss to this layer's Z).
            It has the shape (dataset size, row count, col count, input channels)
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
        k = Convolve.convolve_two_datasets(input_data_tensor, kernel_data_tensor, strides, use_padding)
        dataset_size = input_data_tensor.shape[0]

        k_sum = k.sum(axis=0)
        k_mean = k_sum / dataset_size

        k = k_mean.reshape((k.shape[1], k.shape[2], k.shape[3], k.shape[4]))  # h, w, prev_channels, channels

        return k

    @staticmethod
    def convolve_tensor_dataset_2(input_data_tensor, kernel_tensor, bias=None, strides=(1, 1), use_padding=True):
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

        data_height = input_data_tensor.shape[1]
        data_width = input_data_tensor.shape[2]
        input_channels = input_data_tensor.shape[3]

        kernel_height = kernel_tensor.shape[0]
        kernel_width = kernel_tensor.shape[1]
        input_channels2 = kernel_tensor.shape[2]
        output_channels = kernel_tensor.shape[3]

        if input_channels != input_channels2:
            raise ValueError("Input data channels do not match kernel channels")

        if use_padding:
            padding_h = (kernel_height // 2) * 2
            padding_w = (kernel_width // 2) * 2
        else:
            padding_h = 0
            padding_w = 0

        output_height = (data_height - kernel_height + padding_h) // strides[0] + 1
        output_width = (data_width - kernel_width + padding_w) // strides[1] + 1

        output = np.zeros((sample_size, output_height, output_width, output_channels))

        for i in range(sample_size):
            for j in range(output_channels):
                input_t = input_data_tensor[i, :, :, :]

                if use_padding:
                    input_t = Convolve.pad_cube(input_t, (padding_h / 2, padding_w / 2))

                kernel_t = kernel_tensor[:, :, :, j]
                output[i, :, :, j] = Convolve.convolve_cube(input_t, kernel_t, strides)
                if bias is not None:
                    output[i, :, :, j] += bias[j]

        return output

    @staticmethod
    def convolve_tensor_dataset_back_2(input_data_tensor, kernel_tensor, strides=(1, 1), use_padding=True):
        """
        Convolve the dataset with the 2D conv kernels.
        Convolve in the reverse channel direction.  This is to be used for backprop.

        Parameters
        ----------
        input_data_tensor: ndarray
            Training sample (ow count, col count, input channels)
        kernel_tensor: ndarray
            Stacked 2D convolution kernel of shape (row count, col count, input channels, output channels)
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

        data_height = input_data_tensor.shape[1]
        data_width = input_data_tensor.shape[2]
        input_channels = input_data_tensor.shape[3]

        kernel_height = kernel_tensor.shape[0]
        kernel_width = kernel_tensor.shape[1]
        kernel_input_channels = kernel_tensor.shape[2]
        output_channels = kernel_tensor.shape[3]

        if input_channels != output_channels:  # Note that kernel output_channels is the input channels for this method.
            raise ValueError("Input data channels do not match kernel channels")

        if use_padding:
            padding_h = (kernel_height // 2) * 2
            padding_w = (kernel_width // 2) * 2
        else:
            padding_h = 0
            padding_w = 0

        output_height = (data_height - kernel_height + padding_h) // strides[0] + 1
        output_width = (data_width - kernel_width + padding_w) // strides[1] + 1

        # For a kernel (7 high, 7 wide, 7 prev channels, 3 output channels),
        # we want 7 prev channels as the output of this function as we go backward.
        output = np.zeros((sample_size, output_height, output_width, kernel_input_channels))

        for i in range(sample_size):
            for j in range(kernel_input_channels):
                input_t = input_data_tensor[i, :, :, :]

                if use_padding:
                    input_t = Convolve.pad_cube(input_t, (padding_h / 2, padding_w / 2))

                kernel_t = kernel_tensor[:, :, j, :]
                output[i, :, :, j] = Convolve.convolve_cube(input_t, kernel_t, strides)

        return output
