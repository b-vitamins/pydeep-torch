"""
This module provides implementations for corrupting the training data,
now refactored to use PyTorch internally for array manipulations. We
still use NumPy's random calls to preserve the original test invariances.

:Implemented:
    - Identity
    - Sampling Binary
    - BinaryNoise
    - Additive Gauss Noise
    - Multiplicative Gauss Noise
    - Dropout
    - Random Permutation
    - KeepKWinner
    - KWinnerTakesAll

:Info:
    http://ufldl.stanford.edu/wiki/index.php/Sparse_Coding:_Autoencoder_Interpretation

:Version:
    1.1.0

:Date:
    13.03.2017

:Author:
    Jan Melchior

:Contact:
    JanMelchior@gmx.de

:License:

    Copyright (C) 2017 Jan Melchior

    This file is part of the Python library PyDeep.

    PyDeep is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""

import numpy as numx
import torch


def _ensure_torch_double(arr):
    """Convert arr (NumPy array) to torch double tensor."""
    return torch.as_tensor(arr, dtype=torch.float64)


class Identity(object):
    """Dummy corruptor object."""

    @classmethod
    def corrupt(cls, data):
        """The function corrupts the data.

        :param data: Input of the layer.
        :type data: numpy array [num samples, layer dim]

        :return: Corrupted data.
        :rtype: numpy array [num samples, layer dim]
        """
        return data


class AdditiveGaussNoise(object):
    """An object that corrupts data by adding Gauss noise."""

    def __init__(self, mean, std):
        """The function corrupts the data.

        :param mean: Constant the data is shifted
        :type mean: float

        :param std: Standard deviation Added to the data.
        :type std: float
        """
        self.mean = mean
        self.std = std

    def corrupt(self, data):
        """The function corrupts the data.

        :param data: Input of the layer.
        :type data: numpy array [num samples, layer dim]

        :return: Corrupted data.
        :rtype: numpy array [num samples, layer dim]
        """
        # Use NumPy random for invariance
        noise = numx.random.standard_normal(data.shape) * self.std
        return data + self.mean + noise


class MultiGaussNoise(object):
    """An object that corrupts data by multiplying Gauss noise."""

    def __init__(self, mean, std):
        """Corruptor constructor.

        :param mean: Constant the data is shifted
        :type mean: float

        :param std: Standard deviation Added to the data.
        :type std: float
        """
        self.mean = mean
        self.std = std

    def corrupt(self, data):
        """The function corrupts the data.

        :param data: Input of the layer.
        :type data: numpy array [num samples, layer dim]

        :return: Corrupted data.
        :rtype: numpy array [num samples, layer dim]
        """
        # Again, NumPy random to preserve test seeds
        noise = self.mean + numx.random.standard_normal(data.shape) * self.std
        return data * noise


class SamplingBinary(object):
    """Sample binary states (zero out) corruption."""

    @classmethod
    def corrupt(cls, data):
        """The function corrupts the data.

        :param data: Input of the layer.
        :type data: numpy array [num samples, layer dim]

        :return: Corrupted data.
        :rtype: numpy array [num samples, layer dim]
        """
        # data > rand => True/False => 1/0
        rand_mat = numx.random.random(data.shape)
        return (data > rand_mat).astype(numx.bool_)


class BinaryNoise(object):
    """Binary Noise."""

    def __init__(self, percentage):
        """Corruptor constructor.

        :param percentage: Percent of random chosen pixel/states.
        :type percentage: float [0,1]

        :param std: Standard deviation Added to the data.
        """
        self.percentage = percentage

    def corrupt(self, data):
        """The function corrupts the data.

        :param data: Input of the layer.
        :type data: numpy array [num samples, layer dim]

        :return: Corrupted data.
        :rtype: numpy array [num samples, layer dim]
        """
        # Generate binomial(1, self.percentage), then do abs(data - rand_binomial)
        noise = numx.random.binomial(1, self.percentage, data.shape)
        return numx.abs(data - noise)


class Dropout(object):
    """Dropout (zero out) corruption."""

    def __init__(self, dropout_percentage=0.2):
        """Corruptor constructor.

        :param dropout_percentage: Dropout percentage
        :type dropout_percentage: float
        """
        self.dropout_percentage = dropout_percentage

    def corrupt(self, data):
        """The function corrupts the data.

        :param data: Input of the layer.
        :type data: numpy array [num samples, layer dim]

        :return: Corrupted data.
        :rtype: numpy array [num samples, layer dim]
        """
        # NumPy random for mask
        keep_prob = 1.0 - self.dropout_percentage
        mask = numx.random.binomial(1, keep_prob, data.shape)
        # Scale by 1/(1-dropout) to preserve expected value
        return data * mask / (keep_prob)


class RandomPermutation(object):
    """RandomPermutation corruption, a fix number of units change their activation values."""

    def __init__(self, permutation_percentage=0.2):
        """Corruptor constructor.

        :param permutation_percentage: Percentage of states to permute
        :type permutation_percentage: float
        """
        self.permutation_percentage = permutation_percentage

    def corrupt(self, data):
        """The function corrupts the data.

        :param data: Input of the layer.
        :type data: numpy array [num samples, layer dim]

        :return: Corrupted data.
        :rtype: numpy array [num samples, layer dim]
        """
        result = numx.copy(data)
        # total columns = data.shape[1]
        # number of pairs to swap = num_switches
        num_switches = numx.int32(data.shape[1] * self.permutation_percentage * 0.5)
        for d in range(data.shape[0]):
            # produce a random permutation of columns
            tempset = numx.random.permutation(numx.arange(data.shape[1]))
            # swap the first num_switches with the next num_switches
            result[d][tempset[0:num_switches]] = data[d][
                tempset[num_switches : 2 * num_switches]
            ]
            result[d][tempset[num_switches : 2 * num_switches]] = data[d][
                tempset[0:num_switches]
            ]
        return result


class KeepKWinner(object):
    """Implements K Winner stay. Keep the k max values and set the rest to 0."""

    def __init__(self, k=10, axis=0):
        """Corruptor constructor.

        :param k: Keep the k max values and set the rest to 0.
        :type k: int

        :param axis: Axis =0 across min batch, axis = 1 across hidden units
        :type axis: int
        """
        self.k = k
        self.axis = axis

    def corrupt(self, data):
        """The function corrupts the data.

        :param data: Input of the layer.
        :type data: numpy array [num samples, layer dim]

        :return: Corrupted data.
        :rtype: numpy array [num samples, layer dim]
        """
        data_t = _ensure_torch_double(data)
        if self.axis == 0:
            # sort each column, select threshold in row dimension
            sorted_t, _ = torch.sort(data_t, dim=0)
            # threshold => row (rows - k)
            threshold = sorted_t[sorted_t.shape[0] - self.k, :]
            # mask => data_t >= threshold (broadcast across rows)
            mask = data_t >= threshold
            out_t = data_t * mask
        else:
            # axis=1 => sort each row. We'll do the same trick by transposing
            # or replicate the logic from old code: sort(data, axis=1) => shape NxD
            # threshold => .[:, -k], shape (N,)
            # we broadcast it over columns => data >= threshold
            # or we can transpose
            sorted_t, _ = torch.sort(data_t, dim=1)
            # The threshold is the item in column (columns - k)
            # => shape (N,)
            row_indices = torch.arange(sorted_t.shape[0], dtype=torch.long)
            threshold = sorted_t[row_indices, sorted_t.shape[1] - self.k]
            # We want data_t >= threshold row-wise
            # broadcast => we unsqueeze threshold
            mask = data_t >= threshold.unsqueeze(1)
            out_t = data_t * mask

        return out_t.cpu().numpy()


class KWinnerTakesAll(object):
    """Implements K Winner takes all. Keep the k max values => 1, rest => 0."""

    def __init__(self, k=10, axis=0):
        """Corruptor constructor.

        :param k: Keep the k max values and set the rest to 0.
        :type k: int

        :param axis: Axis =0 across min batch, axis = 1 across hidden units
        :type axis: int
        """
        self.k = k
        self.axis = axis

    def corrupt(self, data):
        """The function corrupts the data.

        :param data: Input of the layer.
        :type data: numpy array [num samples, layer dim]

        :return: Corrupted data.
        :rtype: numpy array [num samples, layer dim]
        """
        data_t = _ensure_torch_double(data)
        if self.axis == 0:
            # similar to KeepKWinner but we set to 1.0 instead of data
            sorted_t, _ = torch.sort(data_t, dim=0)
            threshold = sorted_t[sorted_t.shape[0] - self.k, :]
            mask = data_t >= threshold
            out_t = mask.to(torch.float64)  # => 1.0 or 0.0
        else:
            sorted_t, _ = torch.sort(data_t, dim=1)
            row_indices = torch.arange(sorted_t.shape[0], dtype=torch.long)
            threshold = sorted_t[row_indices, sorted_t.shape[1] - self.k]
            mask = data_t >= threshold.unsqueeze(1)
            out_t = mask.to(torch.float64)

        return out_t.cpu().numpy()
