# -*- coding: utf-8 -*-
"""
  Feed Forward Neural Network Layers.

    .. Note::
    
        Due to computational benefits the common notation for the delta terms is 
        split in a delta term for the common layer and the error signal passed 
        to the layer below. See the following Latex code for details. This allows
        to store all layer depending results in the corresponding layer and avoid 
        useless computations without messing up the code.
        
        \begin{eqnarray}
            \delta^{(n)} &=& Cost'(a^{(n)} ,label) \bullet \sigma'(z^{(n)}) \\
            error^{(i)} &=& (W^{(i)})^T \delta^{(i)} \\
            \delta^{(i)} &=&  error^{(i+1)} \bullet \sigma'(z^{(i)})
        \end{eqnarray}

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

import numpy as numx
import torch

import pydeep.base.activationfunction as AFct
import pydeep.base.costfunction as CFct


###############################################################################
# Torch-based helper functions
###############################################################################


def _as_torch_double(arr):
    """Convert a NumPy array (or scalar) to a torch double Tensor on CPU."""
    return torch.as_tensor(arr, dtype=torch.float64)


def _np_from_torch(tensor):
    """Convert a torch Tensor to a NumPy array on CPU."""
    return tensor.cpu().numpy()


def _dot_torch(x_np, w_np, transpose_w=False):
    """
    Perform torch-based matrix multiplication: x_np dot w_np (or w_np^T).
    Returns a NumPy array.
    """
    x_t = _as_torch_double(x_np)
    w_t = _as_torch_double(w_np)
    if transpose_w:
        w_t = w_t.transpose(0, 1)
    out_t = x_t.matmul(w_t)
    return _np_from_torch(out_t)


def _mean_torch(x_np, axis=None, keepdim=False):
    """
    Compute mean using torch. If axis=None, returns a Python float of entire array's mean.
    Otherwise returns a NumPy array.
    keepdim => whether to keep the dimension for broadcasting.
    """
    x_t = _as_torch_double(x_np)
    if axis is None:
        return float(torch.mean(x_t).item())
    out_t = torch.mean(x_t, dim=axis, keepdim=keepdim)
    return _np_from_torch(out_t)


###############################################################################
# FullConnLayer refactored to use Torch internally
###############################################################################


class FullConnLayer(object):
    """Represents a simple 1D Hidden-Layer."""

    def __init__(
        self,
        input_dim,
        output_dim,
        activation_function=AFct.SoftSign,
        initial_weights="AUTO",
        initial_bias=0.0,
        initial_offset=0.5,
        connections=None,
        dtype=numx.float64,
    ):
        """
        This function initializes all necessary parameters and data structures.

        :Parameters:
            input_dim:            Number of input dimensions.
                                 -type: int

            output_dim            Number of output dimensions.
                                 -type: int

            activation_function:  Activation function.
                                 -type: pydeep.base.activationfunction

            initial_weights:      Initial weights.
                                  'AUTO' = .. seealso:: "Understanding the difficulty of training deep feedforward neural
                                           networks - X Glo, Y Bengio - 2015"
                                  scalar = sampling values from a zero mean Gaussian with std='scalar',
                                  numpy array  = pass an array, for example to implement tied weights.
                                 -type: 'AUTO', scalar or numpy array [input dim, output_dim]

            initial_bias:         Initial bias.
                                  scalar = all values will be set to 'initial_bias',
                                  numpy array  = pass an array
                                 -type: 'AUTO', scalar or numpy array [1, output dim]

            initial_offset:       Initial offset values.
                                  scalar = all values will be set to 'initial_offset',
                                  numpy array  = pass an array
                                 -type: 'AUTO', scalar or numpy array [1, input dim]

            connections:          Connection matrix containing 0 and 1 entries, where 0 connections disable the
                                  corresponding weight.
                                  Example: pydeep.base.numpyextension.generate_2D_connection_matrix() can be used to
                                  construct such a matrix.
                                 -type: numpy array [input dim, output_dim] or None

            dtype:                Used data type i.e. numpy.float64
                                 -type: numpy.float32 or numpy.float64 or numpy.longdouble
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation_function = activation_function
        self.connections = connections
        self.dtype = dtype

        # Temp pre-synaptic output
        self.temp_z = None
        # Temp post-synaptic output
        self.temp_a = None
        # Temp input
        self.temp_x = None
        # Temp error/delta values
        self.temp_deltas = None

        # Initialize weights
        if isinstance(initial_weights, str) and (initial_weights == "AUTO"):
            # "AUTO" logic
            sig_factor = 1.0
            if activation_function == AFct.Sigmoid:
                sig_factor = 4.0
            w_np = (2.0 * numx.random.rand(self.input_dim, self.output_dim) - 1.0) * (
                sig_factor * numx.sqrt(6.0 / (self.input_dim + self.output_dim))
            )
            self.weights = numx.array(w_np, dtype=self.dtype)
        else:
            # scalar or array
            if numx.isscalar(initial_weights):
                w_np = (
                    numx.random.randn(self.input_dim, self.output_dim) * initial_weights
                )
                self.weights = numx.array(w_np, dtype=self.dtype)
            else:
                self.weights = numx.array(initial_weights, dtype=self.dtype)
                if self.weights.shape != (self.input_dim, self.output_dim):
                    raise Exception(
                        "Weight matrix dim. and input dim/output dim. have to match!"
                    )

        # If a connections mask is provided
        if connections is not None:
            self.weights *= connections

        # Initialize bias
        if isinstance(initial_bias, str) and (initial_bias == "AUTO"):
            self.bias = numx.array(numx.zeros((1, self.output_dim)), dtype=self.dtype)
        elif numx.isscalar(initial_bias):
            b_np = numx.zeros((1, self.output_dim)) + initial_bias
            self.bias = numx.array(b_np, dtype=self.dtype)
        else:
            self.bias = numx.array(initial_bias, dtype=self.dtype)
            if self.bias.shape != (1, self.output_dim):
                raise Exception("Bias dim. and output dim. have to match!")

        # Initialize offset
        if isinstance(initial_offset, str) and (initial_offset == "AUTO"):
            off_np = numx.zeros((1, self.input_dim)) + 0.5
            self.offset = numx.array(off_np, dtype=self.dtype)
        elif numx.isscalar(initial_offset):
            off_np = numx.zeros((1, self.input_dim)) + initial_offset
            self.offset = numx.array(off_np, dtype=self.dtype)
        else:
            self.offset = numx.array(initial_offset, dtype=self.dtype)
            if self.offset.shape != (1, self.input_dim):
                raise Exception("Offset dim. and input dim. have to match!")

    def clear_temp_data(self):
        """Sets all temp variables to None."""
        self.temp_z = None
        self.temp_a = None
        self.temp_x = None
        self.temp_deltas = None

    def get_parameters(self):
        """
        This function returns all model parameters in a list.

        :Returns:
            The parameter references in a list.
           -type: list
        """
        return [self.weights, self.bias]

    def update_parameters(self, parameter_updates):
        """
        This function updates all parameters given the updates derived by the training methods.

        :Parameters:
            parameter_updates:  Parameter gradients.
                               -type: list of numpy arrays (num parameters x [para.shape])
        """
        # Simply do p -= u in NumPy
        for p, u in zip(self.get_parameters(), parameter_updates):
            p -= u

    def update_offsets(self, shift=1.0, new_mean=None):
        """
        This function updates the offsets.
        Example: update_offsets(1, 0) reparameterizes to an uncentered model.

        :Parameters:

            shift:    Shifting factor for the offset shift.
                     -type: float

            new_mean: New mean value if None the activation from the last forward
                      propagation is used to calculate the mean.
                     -type: float, numpy array or None
        """
        if shift > 0.0:
            # Use torch for the offset reparam.
            if new_mean is None:
                # mean across the batch, shape => [1, input_dim]
                new_mean_val = _mean_torch(self.temp_x, axis=0, keepdim=True)
            else:
                new_mean_val = new_mean
            new_mean_t = _as_torch_double(new_mean_val)  # shape [1, input_dim]
            offset_t = _as_torch_double(self.offset)  # shape [1, input_dim]
            w_t = _as_torch_double(self.weights)  # shape [input_dim, output_dim]
            bias_t = _as_torch_double(self.bias)  # shape [1, output_dim]
            shift_t = _as_torch_double(shift)  # scalar

            # Reparameterize => self.bias += shift * dot((new_mean - offset), weights)
            diff_t = new_mean_t - offset_t  # shape [1, input_dim]
            # shape => [1, output_dim]
            shift_bias = diff_t.matmul(w_t)

            # Add to bias
            bias_t = bias_t + shift_t * shift_bias

            # Exp. mov. avg. update to offset => offset = (1-shift)*offset + shift*new_mean
            # offset *= 1.0 - shift
            # offset += shift * new_mean
            offset_t = offset_t * (1.0 - shift_t) + shift_t * new_mean_t

            # Store back to NumPy
            self.bias = _np_from_torch(bias_t)
            self.offset = _np_from_torch(offset_t)

    def forward_propagate(self, x):
        """
        Forward-propagates the data through the network and stores pre-syn activation,
        post-syn activation and input mean internally.

        :Parameters:
            x:  Data
                -type: numpy arrays [batchsize, input dim]

        :Returns:
            Post activation
           -type: numpy arrays [batchsize, output dim]
        """
        self.temp_x = x
        x_t = _as_torch_double(x)  # [N, in_dim]
        off_t = _as_torch_double(self.offset)  # [1, in_dim]
        w_t = _as_torch_double(self.weights)  # [in_dim, out_dim]
        b_t = _as_torch_double(self.bias)  # [1, out_dim]

        # z = (x - offset) dot W + b
        z_t = (x_t - off_t).matmul(w_t) + b_t  # [N, out_dim]
        z_np = _np_from_torch(z_t)  # convert back to NumPy
        self.temp_z = z_np
        # post-syn = activation_function
        self.temp_a = self.activation_function.f(z_np)
        return self.temp_a

    def _backward_propagate(self):
        """
        Back-propagates the error signal.

        :Returns:
            Backprop Signal, delta value for the layer below.
           -type: numpy arrays [batchsize, input dim]
        """
        # out = self.temp_deltas dot W^T
        # We'll do it in Torch for consistency
        deltas_t = _as_torch_double(self.temp_deltas)  # [N, out_dim]
        w_t = _as_torch_double(self.weights)  # [in_dim, out_dim]
        # want => dot(deltas, w.T) => shape [N, in_dim]
        out_t = deltas_t.matmul(w_t.transpose(0, 1))
        return _np_from_torch(out_t)

    def _calculate_gradient(self):
        """
        Calculates the gradient for the parameters.

        :Returns:
            The parameters gradient in a list.
           -type: list
        """
        # gradW = ( (temp_x - offset).T dot temp_deltas ) / batch_size
        # gradB = mean(temp_deltas, axis=0)
        x_minus_off_t = _as_torch_double(self.temp_x) - _as_torch_double(self.offset)
        deltas_t = _as_torch_double(self.temp_deltas)

        # shape => [in_dim, out_dim]
        gradW_t = x_minus_off_t.transpose(0, 1).matmul(deltas_t) / self.temp_x.shape[0]

        # If connections is not None => multiply elementwise
        if self.connections is not None:
            conn_t = _as_torch_double(self.connections)
            gradW_t = gradW_t * conn_t

        # shape => [1, out_dim]
        gradB_t = torch.mean(deltas_t, dim=0, keepdim=True)

        # Convert back to NumPy
        gradW_np = _np_from_torch(gradW_t)
        gradB_np = _np_from_torch(gradB_t)

        return [gradW_np, gradB_np]

    def _get_deltas(
        self,
        deltas,
        labels,
        cost,
        reg_cost,
        desired_sparseness,
        cost_sparseness,
        reg_sparseness,
        check_gradient=False,
    ):
        """
        Computes the delta value/ error terms for the layer.

        :Parameters:
            deltas:             Delta values from the layer above or None if top-layer.
                               -type: None or numpy arrays [batchsize, output dim]

            labels:             numpy array or None if the current layer has no cost.
                               -type: None or numpy arrays [batchsize, output dim]

            cost:               Cost function for the layer.
                               -type: pydeep.base.costfunction

            reg_cost:           Strength of the cost function.
                               -type: scalar

            desired_sparseness: Desired sparseness value/average hidden activity.
                               -type: scalar

            cost_sparseness:    Cost function for the sparseness regularization.
                               -type: pydeep.base.costfunction

            reg_sparseness:     Strength of the sparseness term.
                               -type: scalar

            check_gradient:     False for gradient checking mode.
                               -type: bool

        :Returns:
            Delta values for the current layer.
           -type: numpy arrays [batchsize, output dim]
        """
        # We replicate the logic in the original code,
        # using torch for the 'mean' and partial sums if needed.

        optimized = False
        if deltas is None:
            deltas = 0.0
            # We can do the "optimized" approach if cost=CrossEntropy and
            # activation=SoftMax or Sigmoid, and not check_gradient
            if (
                cost == CFct.CrossEntropyError
                or isinstance(cost, CFct.CrossEntropyError)
            ) and (
                self.activation_function == AFct.SoftMax
                or isinstance(self.activation_function, AFct.SoftMax)
                or self.activation_function == AFct.Sigmoid
                or isinstance(self.activation_function, AFct.Sigmoid)
            ):
                optimized = True
                if check_gradient:
                    optimized = False

        # If sparseness is enabled => no optimization
        if reg_sparseness != 0.0:
            # cost_sparseness.df( mean(self.temp_a,axis=0), desired_sparseness )
            mean_act_t = _as_torch_double(self.temp_a)
            # shape => [1, out_dim]
            mean_act_t = torch.mean(mean_act_t, dim=0, keepdim=True)
            mean_act_np = _np_from_torch(mean_act_t)
            # partial derivative wrt. activation
            deltas_sparse = cost_sparseness.df(mean_act_np, desired_sparseness)
            deltas += reg_sparseness * deltas_sparse
            optimized = False

        self.temp_deltas = None

        # If we can't do the optimized path => do the standard approach
        if not optimized:
            # If there's a label cost => add cost derivative
            if reg_cost != 0.0:
                deltas_targets = cost.df(self.temp_a, labels)
                deltas += reg_cost * deltas_targets

            # If activation is SoftMax => big Jacobian
            if self.activation_function == AFct.SoftMax or isinstance(
                self.activation_function, AFct.SoftMax
            ):
                # shape => [N, out_dim, out_dim]
                J = AFct.SoftMax.df(self.temp_a)
                # For each sample in the batch
                res_list = []
                if deltas is None:
                    raise Exception("No cost was specified in top layer!")
                for i in range(self.temp_a.shape[0]):
                    # shape => (1, out_dim)
                    top_delta_i = deltas[i : i + 1, :]
                    # shape => (1, out_dim)
                    temp_i = numx.dot(top_delta_i, J[i])
                    res_list.append(temp_i)
                self.temp_deltas = numx.vstack(res_list)
            else:
                # elementwise multiply
                df_a = self.activation_function.df(self.temp_z)
                self.temp_deltas = df_a * deltas
        else:
            # We are in the "optimized" path => CrossEntropy + Sigmoid/SoftMax
            if reg_cost != 0.0:
                self.temp_deltas = reg_cost * (self.temp_a - labels)

        return self.temp_deltas
