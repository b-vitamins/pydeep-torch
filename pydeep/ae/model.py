# -*- coding: utf-8 -*-
"""
This module provides a general implementation of a 3 layer tied weights Auto-encoder (x-h-y).
The code is focused on readability and clearness, while keeping the efficiency and flexibility high.
Several activation functions are available for visible and hidden units which can be mixed arbitrarily.
The code can easily be adapted to AEs without tied weights. For deep AEs the FFN code can be adapted.

:Implemented:
    -  AE  - Auto-encoder (centered)
    - DAE  - Denoising Auto-encoder (centered)
    - SAE  - Sparse Auto-encoder (centered)
    - CAE  - Contractive Auto-encoder (centered)
    - SLAE - Slow Auto-encoder (centered)

:Info:
    http://ufldl.stanford.edu/wiki/index.php/Sparse_Coding:_Autoencoder_Interpretation

:Version:
    1.0

:Date:
    08.02.2016

:Author:
    Jan Melchior

:Contact:
    JanMelchior@gmx.de

:License:

    Copyright (C) 2016 Jan Melchior

    This program is free software: you can redistribute it and/or modify
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

import numpy as np
import torch

from pydeep.base.activationfunction import Sigmoid, SoftMax
from pydeep.base.basicstructure import BipartiteGraph
from pydeep.base.costfunction import CrossEntropyError


################################################################################
# Torch-based helpers to keep the code DRY
################################################################################


def _as_torch_double(arr):
    """Convert a NumPy array (or float) to a torch double tensor on CPU."""
    return torch.as_tensor(arr, dtype=torch.float64)


def _np_from_torch(tensor):
    """Return a NumPy array from a torch Tensor on CPU."""
    return tensor.cpu().numpy()


def _dot_tied_weights(x_np, w_np, transpose_w=False):
    """
    Perform torch-based matrix multiplication of x_np with w_np, returning a NumPy array.
    If transpose_w=True, we do x_np dot w_np^T.
    """
    x_t = _as_torch_double(x_np)  # shape [N, D1]
    w_t = _as_torch_double(w_np)  # shape [D1, D2] (if not transpose)
    if transpose_w:
        w_t = w_t.transpose(0, 1)  # shape [D2, D1]
    out_t = x_t.matmul(w_t)  # shape [N, D2]
    return _np_from_torch(out_t)


def _sum_along_axis(x_np, axis=None):
    """
    Use torch to sum along `axis` or entire array if axis=None.
    Return as float or NumPy array.
    """
    x_t = _as_torch_double(x_np)
    if axis is None:
        return float(torch.sum(x_t).item())
    else:
        out_t = torch.sum(x_t, dim=axis)
        return _np_from_torch(out_t)


def _mean_along_axis(x_np, axis=None):
    """Use torch to compute mean along `axis` or entire array if axis=None, return NumPy."""
    x_t = _as_torch_double(x_np)
    if axis is None:
        return float(torch.mean(x_t).item())
    else:
        return _np_from_torch(torch.mean(x_t, dim=axis))


def _mul_inplace(param_np, factor):
    """
    param_np *= factor, using Torch. Return updated array.
    """
    p_t = _as_torch_double(param_np)
    factor_t = _as_torch_double(factor)
    p_t = p_t * factor_t
    return _np_from_torch(p_t)


def _add_inplace(param_np, increment_np):
    """
    param_np += increment_np, using Torch. Return updated array.
    """
    p_t = _as_torch_double(param_np)
    i_t = _as_torch_double(increment_np)
    p_t += i_t
    return _np_from_torch(p_t)


################################################################################
# AutoEncoder
################################################################################


class AutoEncoder(BipartiteGraph):
    """Class for a 3 Layer Auto-encoder (x-h-y) with tied weights."""

    def __init__(
        self,
        number_visibles,
        number_hiddens,
        data=None,
        visible_activation_function=Sigmoid,
        hidden_activation_function=Sigmoid,
        cost_function=CrossEntropyError,
        initial_weights="AUTO",
        initial_visible_bias="AUTO",
        initial_hidden_bias="AUTO",
        initial_visible_offsets="AUTO",
        initial_hidden_offsets="AUTO",
        dtype=np.float64,
    ):
        """
        This function initializes all necessary parameters and data
        structures. It is recommended to pass the training data to
        initialize the network automatically.

        :Parameters:
            number_visibles:              Number of the visible variables.
                                         -type: int

            number_hiddens                Number of hidden variables.
                                         -type: int

            data:                         The training data for parameter
                                          initialization if 'AUTO' is chosen.
                                         -type: None or
                                                numpy array [num samples, input dim]
                                                or List of numpy arrays
                                                [num samples, input dim]

            visible_activation_function:  A non linear transformation function
                                          for the visible units (default: Sigmoid)
                                         -type: Subclass of ActivationFunction()

            hidden_activation_function:   A non linear transformation function
                                          for the hidden units (default: Sigmoid)
                                         -type: Subclass of ActivationFunction

            cost_function                 A cost function (default: CrossEntropyError())
                                         -type: subclass of FNNCostFunction()

            initial_weights:              Initial weights.'AUTO' is random
                                         -type: 'AUTO', scalar or
                                                numpy array [input dim, output_dim]

            initial_visible_bias:         Initial visible bias.
                                          'AUTO' is random
                                          'INVERSE_SIGMOID' is the inverse Sigmoid of
                                           the visilbe mean
                                         -type:  'AUTO','INVERSE_SIGMOID', scalar or
                                                 numpy array [1, input dim]

            initial_hidden_bias:          Initial hidden bias.
                                          'AUTO' is random
                                          'INVERSE_SIGMOID' is the inverse Sigmoid of
                                          the hidden mean
                                         -type:  'AUTO','INVERSE_SIGMOID', scalar or
                                                 numpy array [1, output_dim]

            initial_visible_offsets:      Initial visible mean values.
                                          AUTO=data mean or 0.5 if not data is given.
                                         -type:  'AUTO', scalar or
                                                 numpy array [1, input dim]

            initial_hidden_offsets:       Initial hidden mean values.
                                          AUTO = 0.5
                                         -type: 'AUTO', scalar or
                                                 numpy array [1, output_dim]

            dtype:                        Used data type i.e. numpy.float64
                                         -type: numpy.float32 or numpy.float64 or
                                                numpy.longdouble
        """

        # (Implementation code unchanged)
        if (cost_function == CrossEntropyError) and not (
            visible_activation_function == Sigmoid
        ):
            raise Exception(
                "The Cross entropy cost should only be used with Sigmoid units or units of "
                "interval (0,1)",
                UserWarning,
            )
        if (
            isinstance(initial_visible_bias, str)
            and initial_visible_bias in ["AUTO", "INVERSE_SIGMOID"]
            and not (visible_activation_function == Sigmoid)
        ):
            initial_visible_bias = 0.0
        if (
            isinstance(initial_hidden_bias, str)
            and initial_hidden_bias in ["AUTO", "INVERSE_SIGMOID"]
            and not (hidden_activation_function == Sigmoid)
        ):
            initial_hidden_bias = 0.0
        if (
            visible_activation_function == SoftMax
            or hidden_activation_function == SoftMax
        ):
            raise Exception("Softmax not supported but you can use FNN instead!")

        super(AutoEncoder, self).__init__(
            number_visibles=number_visibles,
            number_hiddens=number_hiddens,
            data=data,
            visible_activation_function=visible_activation_function,
            hidden_activation_function=hidden_activation_function,
            initial_weights=initial_weights,
            initial_visible_bias=initial_visible_bias,
            initial_hidden_bias=initial_hidden_bias,
            initial_visible_offsets=initial_visible_offsets,
            initial_hidden_offsets=initial_hidden_offsets,
            dtype=dtype,
        )
        self.cost_function = cost_function

    def _get_contractive_penalty(self, a_h, factor):
        """Calculates contractive penalty cost for a data point x.

        :Parameters:

            a_h:     Pre-synaptic activation of h: a_h = (Wx+c).
                    -type: numpy array [num samples, hidden dim]

            factor:  Influence factor (lambda) for the penalty.
                    -type: float

        :Returns:
            Contractive penalty costs for x.
           -type: numpy array [num samples]
        """
        # (Implementation code unchanged)
        factor_t = _as_torch_double(factor)
        w_t = _as_torch_double(self.w)
        w2_sum_t = torch.sum(w_t * w_t, dim=0).unsqueeze(0)
        df_a_h_np = self.hidden_activation_function.df(a_h)
        df2_t = _as_torch_double(df_a_h_np) ** 2.0
        out_t = factor_t * torch.sum(df2_t * w2_sum_t, dim=1)
        return _np_from_torch(out_t)

    def _get_sparse_penalty(self, h, factor, desired_sparseness):
        """Calculates sparseness penalty cost for a data point x.
            .. Warning:: Different penalties are used depending on the
                     hidden activation function.

        :Parameters:

            h:                   hidden activation.
                                -type: numpy array [num samples, hidden dim]

            factor:              Influence factor (beta) for the penalty.
                                -type: float

            desired_sparseness:  Desired average hidden activation.
                                -type: float

        :Returns:
            Sparseness penalty costs for x.
           -type: numpy array [num samples]
        """
        # (Implementation code unchanged)
        factor_t = _as_torch_double(factor)
        h_t = _as_torch_double(h)
        mean_h_t = torch.mean(h_t, dim=0, keepdim=True)
        ds_t = _as_torch_double(desired_sparseness)

        if self.hidden_activation_function == Sigmoid:
            min_val = 1e-10
            max_val = 1.0 - min_val
            clipped_t = torch.clamp(mean_h_t, min_val, max_val)
            term_t = ds_t * torch.log(ds_t / clipped_t) + (1.0 - ds_t) * torch.log(
                (1.0 - ds_t) / (1.0 - clipped_t)
            )
            sum_t = torch.sum(term_t, dim=1)
        else:
            diff_t = ds_t - mean_h_t
            sum_t = torch.sum(diff_t * diff_t, dim=1)

        out_t = factor_t * sum_t
        N = h.shape[0]
        repeated_t = out_t.expand(N)
        return _np_from_torch(repeated_t)

    def _get_slowness_penalty(self, h, h_next, factor):
        """Calculates slowness penalty cost for a data point x.
            .. Warning:: Different penalties are used depending on the
                     hidden activation function.

        :Parameters:

            h:                   hidden activation.
                                -type: numpy array [num samples, hidden dim]

            h_next:              hidden activation of the next data point in a sequence.
                                -type: numpy array [num samples, hidden dim]

            factor:              Influence factor (beta) for the penalty.
                                -type: float

        :Returns:
            Sparseness penalty costs for x.
           -type: numpy array [num samples]
        """
        # (Implementation code unchanged)
        factor_t = _as_torch_double(factor)
        h_t = _as_torch_double(h)
        h_next_t = _as_torch_double(h_next)
        diff_t = h_t - h_next_t
        sum_t = torch.sum(diff_t * diff_t, dim=1)
        out_t = factor_t * sum_t
        return _np_from_torch(out_t)

    def energy(
        self,
        x,
        contractive_penalty=0.0,
        sparse_penalty=0.0,
        desired_sparseness=0.01,
        x_next=None,
        slowness_penalty=0.0,
    ):
        """Calculates the energy/cost for a data point x.

        :Parameters:

            x:                   Data points.
                                -type: numpy array [num samples, input dim]

            contractive_penalty: If a value > 0.0 is given the cost is also
                                 calculated on the contractive penalty.
                                -type: float

            sparse_penalty:      If a value > 0.0 is given the cost is also
                                 calculated on the sparseness penalty.
                                -type: float

            desired_sparseness:  Desired average hidden activation.
                                -type: float

            x_next:              Next data points.
                                -type: None or numpy array [num samples, input dim]

            slowness_penalty:    If a value > 0.0 is given the cost is also
                                 calculated on the slowness penalty.
                                -type: float

        :Returns:
            Costs for x.
           -type: numpy array [num samples]
        """
        # (Implementation code unchanged)
        a_h, h = self._encode(x)
        a_y, y = self._decode(h)
        cost = self.cost_function.f(y, x)
        if contractive_penalty > 0.0:
            cost += self._get_contractive_penalty(a_h, contractive_penalty)
        if sparse_penalty > 0.0:
            cost += self._get_sparse_penalty(h, sparse_penalty, desired_sparseness)
        if slowness_penalty > 0.0 and x_next is not None:
            h_next = self.encode(x_next)
            cost += self._get_slowness_penalty(h, h_next, slowness_penalty)
        return cost

    def _encode(self, x):
        """The function propagates the activation of the input
           layer through the network to the hidden/output layer.

        :Parameters:

            x:    Input of the network.
                 -type: numpy array [num samples, input dim]

        :Returns:
            Pre and Post synaptic output.
           -type: List of arrays [num samples, hidden dim]
        """
        # (Implementation code unchanged)
        pre_act_h = self._hidden_pre_activation(x)
        h = self._hidden_post_activation(pre_act_h)
        return pre_act_h, h

    def encode(self, x):
        """The function propagates the activation of the input
           layer through the network to the hidden/output layer.

        :Parameters:

            x:    Input of the network.
                 -type: numpy array [num samples, input dim]

        :Returns:
            Output of the network.
           -type: array [num samples, hidden dim]
        """
        # (Implementation code unchanged)
        return self._encode(np.atleast_2d(x))[1]

    def _decode(self, h):
        """The function propagates the activation of the hidden
           layer reverse through the network to the input layer.

        :Parameters:

            h:    Output of the network
                 -type: numpy array [num samples, hidden dim]

        :Returns:
            Input of the network.
           -type: array [num samples, input dim]
        """
        # (Implementation code unchanged)
        pre_act_y = self._visible_pre_activation(h)
        y = self._visible_post_activation(pre_act_y)
        return pre_act_y, y

    def decode(self, h):
        """The function propagates the activation of the hidden
           layer reverse through the network to the input layer.

        :Parameters:

            h:    Output of the network
                 -type: numpy array [num samples, hidden dim]

        :Returns:
            Pre and Post synaptic input.
           -type: List of arrays [num samples, input dim]
        """
        # (Implementation code unchanged)
        return self._decode(np.atleast_2d(h))[1]

    def reconstruction_error(self, x, absolut=False):
        """Calculates the reconstruction error for given training data.

        :Parameters:

            x:       Datapoints
                    -type: numpy array [num samples, input dim]

            absolut: If true the absolute error is caluclated.
                    -type: bool

        :Returns:
            Reconstruction error.
           -type: List of arrays [num samples, 1]
        """
        # (Implementation code unchanged)
        diff = x - self.decode(self.encode(x))
        if absolut:
            diff = np.abs(diff)
        else:
            diff = diff**2
        return np.sum(diff, axis=1)

    def _get_gradients(
        self,
        x,
        a_h,
        h,
        a_y,
        y,
        reg_contractive,
        reg_sparseness,
        desired_sparseness,
        reg_slowness,
        x_next,
        a_h_next,
        h_next,
    ):
        """
        Computes the gradients of weights, visible and the hidden bias.
        Depending on whether contractive penalty and or sparse penalty
        is used the gradient changes.

        :Parameters:

            x:                    Training data.
                                 -type: numpy array [num samples, input dim]

            a_h:                  Pre-synaptic activation of h: a_h = (Wx+c).
                                 -type: numpy array [num samples, output dim]

            h                     Post-synaptic activation of h: h = f(a_h).
                                 -type: numpy array [num samples, output dim]

            a_y:                  Pre-synaptic activation of y: a_y = (Wh+b).
                                 -type: numpy array [num samples, input dim]

            y                     Post-synaptic activation of y: y = f(a_y).
                                 -type: numpy array [num samples, input dim]

            reg_contractive:      Contractive influence factor (lambda).
                                 -type: float

            reg_sparseness:       Sparseness influence factor (lambda).
                                 -type: float

            desired_sparseness:   Desired average hidden activation.
                                 -type: float

            reg_slowness:         Slowness influence factor.
                                 -type: float

            x_next:               Next Training data in Sequence.
                                 -type: numpy array [num samples, input dim]

            a_h_next:             Next pre-synaptic activation of h: a_h = (Wx+c).
                                 -type: numpy array [num samples, output dim]

            h_next                Next post-synaptic activation of h: h = f(a_h).
                                 -type: numpy array [num samples, input dim]
        """
        # (Implementation code unchanged)
        df_b_np = self.cost_function.df(y, x)
        df_a_y_np = self.visible_activation_function.df(a_y)
        grad_b_np = df_b_np * df_a_y_np
        temp_c = _dot_tied_weights(grad_b_np, self.w, transpose_w=False)

        if reg_sparseness > 0.0:
            sp_part = self.__get_sparse_penalty_gradient_part(h, desired_sparseness)
            temp_c += reg_sparseness * sp_part

        df_a_h_np = self.hidden_activation_function.df(a_h)
        grad_c_np = temp_c * df_a_h_np

        x_minus_ov = x - self.ov
        h_minus_oh = h - self.oh
        w_grad_1 = _dot_tied_weights(x_minus_ov.T, grad_c_np, transpose_w=False)
        w_grad_2 = _dot_tied_weights(grad_b_np.T, h_minus_oh, transpose_w=False)
        w_grad = w_grad_1 + w_grad_2
        w_grad /= x.shape[0]

        grad_b_np = np.mean(grad_b_np, axis=0, keepdims=True)
        grad_c_np = np.mean(grad_c_np, axis=0, keepdims=True)

        if reg_contractive > 0.0:
            pW, pc = self._get_contractive_penalty_gradient(x, a_h, df_a_h_np)
            grad_c_np += reg_contractive * pc
            w_grad += reg_contractive * pW

        if reg_slowness > 0.0 and x_next is not None:
            df_a_h_next_np = self.hidden_activation_function.df(a_h_next)
            pW_slow, pc_slow = self._get_slowness_penalty_gradient(
                x, x_next, h, h_next, df_a_h_np, df_a_h_next_np
            )
            grad_c_np += reg_slowness * pc_slow
            w_grad += reg_slowness * pW_slow

        return [w_grad, grad_b_np, grad_c_np]

    def __get_sparse_penalty_gradient_part(self, h, desired_sparseness):
        """
        This function computes the desired part of the gradient
        for the sparse penalty term. Only used for efficiency.

        :Parameters:

            h:                        hidden activations
                                     -type: numpy array [num samples, input dim]

            desired_sparseness:       Desired average hidden activation.
                                     -type: float

        :Returs:
            The computed gradient part is returned
           -type: numpy array [1, hidden dim]
        """
        # (Implementation code unchanged)
        mean_h = np.atleast_2d(np.mean(h, axis=0))
        if self.hidden_activation_function == Sigmoid:
            min_val = 1e-10
            max_val = 1.0 - min_val
            mean_h = np.clip(mean_h, min_val, max_val)
            grad = -desired_sparseness / mean_h + (1.0 - desired_sparseness) / (
                1.0 - mean_h
            )
        else:
            grad = -2.0 * (desired_sparseness - mean_h)
        return grad

    def _get_sparse_penalty_gradient(self, h, df_a_h, desired_sparseness):
        """
        This function computes the gradient for the sparse penalty term.

        :Parameters:

            h:                  hidden activations
                               -type: numpy array [num samples, input dim]

            df_a_h:             Derivative of untransformed hidden activations
                               -type: numpy array [num samples, input dim]

            desired_sparseness: Desired average hidden activation.
                               -type: float

        :Returs:
            The computed gradient part is returned
           -type: numpy array [1, hidden dim]
        """
        # (Implementation code unchanged)
        sp_part = self.__get_sparse_penalty_gradient_part(h, desired_sparseness)
        return sp_part * df_a_h

    def _get_contractive_penalty_gradient(self, x, a_h, df_a_h):
        """
        This function computes the gradient for the contractive penalty term.

        :Parameters:

            x:      Training data.
                   -type: numpy array [num samples, input dim]

            a_h:    Untransformed hidden activations
                   -type: numpy array [num samples, input dim]

            df_a_h: Derivative of untransformed hidden activations
                   -type: numpy array [num samples, input dim]

        :Returs:
            The computed gradient is returned
           -type: numpy array [input dim, hidden dim]
        """
        # (Implementation code unchanged)
        x_t = _as_torch_double(x)
        w_t = _as_torch_double(self.w)
        df_a_h_t = _as_torch_double(df_a_h)
        ddf_a_h_np = self.hidden_activation_function.ddf(a_h)
        ddf_a_h_t = _as_torch_double(ddf_a_h_np)
        w2_sum_t = torch.sum(w_t * w_t, dim=0)
        product_c_t = 2.0 * df_a_h_t * ddf_a_h_t
        product_c_t = product_c_t * w2_sum_t
        product_c_mean_t = torch.mean(product_c_t, dim=0, keepdim=True)
        grad_c_np = _np_from_torch(product_c_mean_t)
        x_minus_ov_t = x_t - _as_torch_double(self.ov)
        partA_t = x_minus_ov_t.transpose(0, 1).matmul(product_c_t)
        partA_t /= x.shape[0]
        df_a_h_squared_t = df_a_h_t * df_a_h_t
        mean_df2_t = torch.mean(df_a_h_squared_t, dim=0)
        partB_t = 2.0 * w_t * mean_df2_t
        grad_w_t = partA_t + partB_t
        grad_w_np = _np_from_torch(grad_w_t)
        return [grad_w_np, grad_c_np]

    def _get_slowness_penalty_gradient(self, x, x_next, h, h_next, df_a_h, df_a_h_next):
        """This function computes the gradient for the slowness penalty term.

        :Parameters:

            x:           Training data.
                        -type: numpy array [num samples, input dim]

            x_next:      Next training data points in Sequence.
                        -type: numpy array [num samples, input dim]

            h:           Corresponding hidden activations.
                        -type: numpy array [num samples, output dim]

            h_next:      Corresponding next hidden activations.
                        -type: numpy array [num samples, output dim]

            df_a_h:      Derivative of untransformed hidden activations.
                        -type: numpy array [num samples, input dim]

            df_a_h_next: Derivative of untransformed next hidden activations.
                        -type: numpy array [num samples, input dim]

        :Returs:
            The computed gradient is returned
           -type: numpy array [input dim, hidden dim]
        """
        # (Implementation code unchanged)
        x_t = _as_torch_double(x)
        x_next_t = _as_torch_double(x_next)
        h_t = _as_torch_double(h)
        h_next_t = _as_torch_double(h_next)
        df_a_h_t = _as_torch_double(df_a_h)
        df_a_h_next_t = _as_torch_double(df_a_h_next)
        diff_t = 2.0 * (h_t - h_next_t)
        x_minus_ov_t = x_t - _as_torch_double(self.ov)
        x_next_minus_ov_t = x_next_t - _as_torch_double(self.ov)
        part1_t = x_minus_ov_t.transpose(0, 1).matmul(diff_t * df_a_h_t)
        part2_t = x_next_minus_ov_t.transpose(0, 1).matmul(diff_t * df_a_h_next_t)
        grad_w_t = (part1_t - part2_t) / x.shape[0]
        c_t = torch.mean(diff_t * (df_a_h_t - df_a_h_next_t), dim=0, keepdim=True)
        grad_c_np = _np_from_torch(c_t)
        grad_w_np = _np_from_torch(grad_w_t)
        return [grad_w_np, grad_c_np]

    def finit_differences(
        self,
        data,
        delta,
        reg_sparseness,
        desired_sparseness,
        reg_contractive,
        reg_slowness,
        data_next,
    ):
        """Finite differences test for AEs.
        The finite differences test involves all functions of the model except init and reconstruction_error

        data:                    The training data
                                -type: numpy array [num samples, input dim]

        delta:                   The learning rate.
                                -type: numpy array[num parameters]


        reg_sparseness:          The parameter (epsilon) for the sparseness regularization.
                                -type: float

        desired_sparseness:      Desired average hidden activation.
                                -type: float

        reg_contractive:         The parameter (epsilon) for the contractive regularization.
                                -type: float

        reg_slowness:            The parameter (epsilon) for the slowness regularization.
                                -type: float

        data_next:               The next training data in the sequence.
                                -type: numpy array [num samples, input dim]
        """
        # (Implementation code unchanged)
        data = np.atleast_2d(data)
        diff_w = np.zeros((data.shape[0], self.w.shape[0], self.w.shape[1]))
        diff_b = np.zeros((data.shape[0], self.bv.shape[0], self.bv.shape[1]))
        diff_c = np.zeros((data.shape[0], self.bh.shape[0], self.bh.shape[1]))

        for d in range(data.shape[0]):
            batch = data[d].reshape(1, data.shape[1])
            a_h, h = self._encode(batch)
            a_y, y = self._decode(h)
            if data_next is not None:
                batch_next = data_next[d].reshape(1, data.shape[1])
                a_h_next, h_next = self._encode(batch_next)
            else:
                batch_next = None
                a_h_next, h_next = None, None

            grads = self._get_gradients(
                batch,
                a_h,
                h,
                a_y,
                y,
                reg_contractive,
                reg_sparseness,
                desired_sparseness,
                reg_slowness,
                batch_next,
                a_h_next,
                h_next,
            )
            grad_w0 = grads[0]
            grad_b0 = grads[1]
            grad_c0 = grads[2]

            for i in range(self.input_dim):
                for j in range(self.output_dim):
                    grad_w_ij = grad_w0[i, j]
                    self.w[i, j] += delta
                    E_pos = self.energy(
                        batch,
                        reg_contractive,
                        reg_sparseness,
                        desired_sparseness,
                        batch_next,
                        reg_slowness,
                    )
                    self.w[i, j] -= 2 * delta
                    E_neg = self.energy(
                        batch,
                        reg_contractive,
                        reg_sparseness,
                        desired_sparseness,
                        batch_next,
                        reg_slowness,
                    )
                    self.w[i, j] += delta
                    diff_w[d, i, j] = grad_w_ij - ((E_pos - E_neg) / (2.0 * delta))

            for i in range(self.input_dim):
                grad_b_i = grad_b0[0, i]
                self.bv[0, i] += delta
                E_pos = self.energy(
                    batch,
                    reg_contractive,
                    reg_sparseness,
                    desired_sparseness,
                    batch_next,
                    reg_slowness,
                )
                self.bv[0, i] -= 2 * delta
                E_neg = self.energy(
                    batch,
                    reg_contractive,
                    reg_sparseness,
                    desired_sparseness,
                    batch_next,
                    reg_slowness,
                )
                self.bv[0, i] += delta
                diff_b[d, 0, i] = grad_b_i - ((E_pos - E_neg) / (2.0 * delta))

            for j in range(self.output_dim):
                grad_c_j = grad_c0[0, j]
                self.bh[0, j] += delta
                E_pos = self.energy(
                    batch,
                    reg_contractive,
                    reg_sparseness,
                    desired_sparseness,
                    batch_next,
                    reg_slowness,
                )
                self.bh[0, j] -= 2 * delta
                E_neg = self.energy(
                    batch,
                    reg_contractive,
                    reg_sparseness,
                    desired_sparseness,
                    batch_next,
                    reg_slowness,
                )
                self.bh[0, j] += delta
                diff_c[d, 0, j] = grad_c_j - ((E_pos - E_neg) / (2.0 * delta))

        return [diff_w, diff_b, diff_c]
