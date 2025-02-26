"""
This module provides basic structural elements, which different models have in common.

:Implemented:
    - BipartiteGraph
    - StackOfBipartiteGraphs

:Version:
    1.1.0

:Date:
    06.04.2017

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

import numpy as np
import torch
from pydeep.base.activationfunction import Sigmoid
from pydeep.misc.io import save_object

##############################################################################
# Utility functions for PyTorch-based dot + add.
##############################################################################


def _torch_dot_add(a_np, b_np, bias_np=None, transpose_b=False):
    """
    Performs torch-based matrix multiplication:
        out = (a_np) x (b_np) + bias_np
    and returns the result as a NumPy array.

    :param a_np: NumPy array shape [N, D]
    :param b_np: NumPy array shape [D, M] (or [M, D] if transpose_b=True)
    :param bias_np: NumPy array shape [1, M] or broadcastable to [N, M], or None
    :param transpose_b: If True, b_np is shape [M, D] so we transpose
    :return: NumPy array shape [N, M]
    """
    a_t = torch.as_tensor(a_np, dtype=torch.float64)
    b_t = torch.as_tensor(b_np, dtype=torch.float64)
    if transpose_b:
        b_t = b_t.transpose(0, 1)
    out_t = a_t.matmul(b_t)
    if bias_np is not None:
        bias_t = torch.as_tensor(bias_np, dtype=torch.float64)
        out_t = out_t + bias_t  # broadcast
    return out_t.cpu().numpy()


def _torch_add_inplace(param_np, update_np):
    """
    param_np += update_np in Torch. Returns an updated NumPy array (in place).
    """
    p_t = torch.as_tensor(param_np, dtype=torch.float64)
    u_t = torch.as_tensor(update_np, dtype=torch.float64)
    p_t += u_t
    return p_t.cpu().numpy()


##############################################################################
# BipartiteGraph
##############################################################################


class BipartiteGraph(object):
    """Implementation of a bipartite graph structure."""

    def __init__(
        self,
        number_visibles,
        number_hiddens,
        data=None,
        visible_activation_function=Sigmoid,
        hidden_activation_function=Sigmoid,
        initial_weights="AUTO",
        initial_visible_bias="AUTO",
        initial_hidden_bias="AUTO",
        initial_visible_offsets="AUTO",
        initial_hidden_offsets="AUTO",
        dtype=np.float64,
    ):
        """ This function initializes all necessary parameters and data structures. It is recommended to pass the \
            training data to initialize the network automatically.

        :param number_visibles: Number of the visible variables.
        :type number_visibles: int

        :param number_hiddens: Number of the hidden variables.
        :type number_hiddens: int

        :param data: The training data for parameter initialization if 'AUTO' is chosen for the corresponding parameter.
        :type data: None or numpy array [num samples, input dim]

        :param visible_activation_function: Activation function for the visible units.
        :type visible_activation_function: pydeep.base.activationFunction

        :param hidden_activation_function: Activation function for the hidden units.
        :type hidden_activation_function: pydeep.base.activationFunction

        :param initial_weights: Initial weights. 'AUTO' and a scalar are random init.
        :type initial_weights: 'AUTO', scalar or numpy array [input dim, output_dim]

        :param initial_visible_bias: Initial visible bias. 'AUTO' is random, 'INVERSE_SIGMOID' is the inverse Sigmoid \
                                     of the visible mean. If a scalar is passed all values are initialized with it.
        :type initial_visible_bias: 'AUTO','INVERSE_SIGMOID', scalar or numpy array [1, input dim]

        :param initial_hidden_bias: Initial hidden bias. 'AUTO' is random, 'INVERSE_SIGMOID' is the inverse Sigmoid of \
                                    the hidden mean. If a scalar is passed all values are initialized with it.
        :type initial_hidden_bias: 'AUTO','INVERSE_SIGMOID', scalar or numpy array [1, output_dim]

        :param initial_visible_offsets: Initial visible offset values. AUTO=data mean or 0.5 if no data is given. If a \
                                        scalar is passed all values are initialized with it
        :type initial_visible_offsets: 'AUTO', scalar or numpy array [1, input dim]

        :param initial_hidden_offsets: Initial hidden offset values. AUTO = 0.5 If a scalar is passed all values are \
                                       initialized with it.
        :type initial_hidden_offsets: 'AUTO', scalar or numpy array [1, output_dim]

        :param dtype: Used data type i.e. numpy.float64.
        :type dtype: numpy.float32 or numpy.float64 or numpy.longdouble
        """
        self.dtype = dtype
        self.input_dim = number_visibles
        self.output_dim = number_hiddens

        self.visible_activation_function = visible_activation_function
        self.hidden_activation_function = hidden_activation_function

        # If data is provided, store its mean/std
        self._data_mean = 0.5 * np.ones((1, self.input_dim), dtype=self.dtype)
        self._data_std = np.ones((1, self.input_dim), dtype=self.dtype)
        if data is not None:
            if isinstance(data, list):
                data = np.concatenate(data)
            if self.input_dim != data.shape[1]:
                raise ValueError(
                    "Data dimension and model input dimension have to be equal!"
                )
            self._data_mean = data.mean(axis=0).reshape(1, data.shape[1])
            self._data_std = data.std(axis=0).reshape(1, data.shape[1])

        # Initialize Weights
        if isinstance(initial_weights, str) and initial_weights == "AUTO":
            self.w = np.array(
                (2.0 * np.random.rand(self.input_dim, self.output_dim) - 1.0)
                * (4.0 * np.sqrt(6.0 / (self.input_dim + self.output_dim))),
                dtype=dtype,
            )
        elif np.isscalar(initial_weights):
            self.w = np.array(
                np.random.randn(self.input_dim, self.output_dim) * initial_weights,
                dtype=dtype,
            )
        else:
            # It's presumably an array
            self.w = np.array(initial_weights, dtype=dtype)

        # Initialize visible offset
        self.ov = np.zeros((1, self.input_dim), dtype=self.dtype)
        if (
            isinstance(initial_visible_offsets, str)
            and initial_visible_offsets == "AUTO"
        ):
            if data is not None:
                self.ov += self._data_mean
            else:
                self.ov += 0.5
        elif np.isscalar(initial_visible_offsets):
            self.ov += initial_visible_offsets
        else:
            self.ov += initial_visible_offsets.reshape(1, self.input_dim)

        # Initialize visible bias
        if isinstance(initial_visible_bias, str) and initial_visible_bias == "AUTO":
            if data is None:
                self.bv = np.zeros((1, self.input_dim), dtype=dtype)
            else:
                tmp = np.clip(self._data_mean, 0.001, 0.999)
                self.bv = np.array(Sigmoid.g(tmp), dtype=dtype)
        elif (
            isinstance(initial_visible_bias, str)
            and initial_visible_bias == "INVERSE_SIGMOID"
        ):
            tmp = np.clip(self.ov, 0.001, 0.999)
            self.bv = np.array(Sigmoid.g(tmp), dtype=dtype)
        elif np.isscalar(initial_visible_bias):
            arr = initial_visible_bias + np.zeros((1, self.input_dim))
            self.bv = np.array(arr, dtype=dtype)
        else:
            self.bv = np.array(initial_visible_bias, dtype=dtype)

        # Initialize hidden offset
        self.oh = np.zeros((1, self.output_dim), dtype=self.dtype)
        if isinstance(initial_hidden_offsets, str) and initial_hidden_offsets == "AUTO":
            self.oh += 0.5
        elif np.isscalar(initial_hidden_offsets):
            self.oh += initial_hidden_offsets
        else:
            self.oh += initial_hidden_offsets.reshape(1, self.output_dim)

        # Initialize hidden bias
        if isinstance(initial_hidden_bias, str) and initial_hidden_bias == "AUTO":
            self.bh = np.zeros((1, self.output_dim), dtype=dtype)
        elif (
            isinstance(initial_hidden_bias, str)
            and initial_hidden_bias == "INVERSE_SIGMOID"
        ):
            tmp = np.clip(self.oh, 0.001, 0.999)
            self.bh = np.array(Sigmoid.g(tmp), dtype=dtype)
        elif np.isscalar(initial_hidden_bias):
            arr = initial_hidden_bias + np.zeros((1, self.output_dim))
            self.bh = np.array(arr, dtype=dtype)
        else:
            self.bh = np.array(initial_hidden_bias, dtype=dtype)

    ########################################################################
    # Forward/Backward
    ########################################################################

    def _visible_pre_activation(self, h):
        """
        Computes the visible pre-activations from hidden activations:
        pre_act_v = (h - oh) dot w^T + bv
        """
        return _torch_dot_add(h - self.oh, self.w, bias_np=self.bv, transpose_b=True)

    def _visible_post_activation(self, pre_act_v):
        """
        Applies the visible activation function to the pre-activations.
        """
        return self.visible_activation_function.f(pre_act_v)

    def visible_activation(self, h):
        """
        Full visible activation from hidden activations:
           v = f( (h - oh) dot w^T + bv )
        """
        pre_v = self._visible_pre_activation(h)
        return self._visible_post_activation(pre_v)

    def _hidden_pre_activation(self, v):
        """
        Computes the hidden pre-activations from visible activations:
        pre_act_h = (v - ov) dot w + bh
        """
        return _torch_dot_add(v - self.ov, self.w, bias_np=self.bh, transpose_b=False)

    def _hidden_post_activation(self, pre_act_h):
        """
        Applies the hidden activation function to the pre-activations.
        """
        return self.hidden_activation_function.f(pre_act_h)

    def hidden_activation(self, v):
        """
        Full hidden activation from visible activations:
           h = f( (v - ov) dot w + bh )
        """
        pre_h = self._hidden_pre_activation(v)
        return self._hidden_post_activation(pre_h)

    ########################################################################
    # Add/Remove Hidden Units
    ########################################################################

    def _add_hidden_units(
        self,
        num_new_hiddens,
        position=0,
        initial_weights="AUTO",
        initial_bias="AUTO",
        initial_offsets="AUTO",
    ):
        """ This function adds new hidden units at the given position to the model. \
            .. Warning:: If the parameters are changed, the trainer needs to be reinitialized.
        """
        # We do random init in NumPy for test invariance
        if isinstance(initial_weights, str) and initial_weights == "AUTO":
            new_weights = (
                2.0 * np.random.rand(self.input_dim, num_new_hiddens) - 1.0
            ) * (
                4.0
                * np.sqrt(6.0 / (self.input_dim + self.output_dim + num_new_hiddens))
            )
        elif np.isscalar(initial_weights):
            new_weights = (
                np.random.randn(self.input_dim, num_new_hiddens) * initial_weights
            )
        else:
            new_weights = initial_weights

        self.w = np.insert(
            self.w,
            np.array(np.ones(num_new_hiddens) * position, dtype=int),
            new_weights,
            axis=1,
        ).astype(self.dtype, copy=False)

        if isinstance(initial_offsets, str) and initial_offsets == "AUTO":
            new_oh = 0.5 * np.ones((1, num_new_hiddens))
        elif np.isscalar(initial_offsets):
            new_oh = np.zeros((1, num_new_hiddens)) + initial_offsets
        else:
            new_oh = initial_offsets

        self.oh = np.insert(
            self.oh,
            np.array(np.ones(num_new_hiddens) * position, dtype=int),
            new_oh,
            axis=1,
        ).astype(self.dtype, copy=False)

        if isinstance(initial_bias, str) and initial_bias == "AUTO":
            new_bias = np.zeros((1, num_new_hiddens))
        elif isinstance(initial_bias, str) and initial_bias == "INVERSE_SIGMOID":
            new_bias = Sigmoid.g(np.clip(new_oh, 0.01, 0.99))
        elif np.isscalar(initial_bias):
            new_bias = initial_bias + np.zeros((1, num_new_hiddens))
        else:
            new_bias = initial_bias

        self.bh = np.insert(
            self.bh,
            np.array(np.ones(num_new_hiddens) * position, dtype=int),
            new_bias,
            axis=1,
        ).astype(self.dtype, copy=False)

        self.output_dim = self.w.shape[1]

    def _remove_hidden_units(self, indices):
        """Removes the hidden units whose indices are given."""
        idx = np.array(indices)
        self.w = np.delete(self.w, idx, axis=1)
        self.bh = np.delete(self.bh, idx, axis=1)
        self.oh = np.delete(self.oh, idx, axis=1)
        self.output_dim = self.w.shape[1]

    ########################################################################
    # Add/Remove Visible Units
    ########################################################################

    def _add_visible_units(
        self,
        num_new_visibles,
        position=0,
        initial_weights="AUTO",
        initial_bias="AUTO",
        initial_offsets="AUTO",
        data=None,
    ):
        """Adds new visible units at the given position to the model."""
        new_data_mean = 0.5 * np.ones((1, num_new_visibles), dtype=self.dtype)
        new_data_std = np.ones((1, num_new_visibles), dtype=self.dtype) / 12.0
        if data is not None:
            if isinstance(data, list):
                data = np.concatenate(data)
            new_data_mean = data.mean(axis=0).reshape(1, num_new_visibles)
            new_data_std = data.std(axis=0).reshape(1, num_new_visibles)

        self._data_mean = np.insert(
            self._data_mean,
            np.array(np.ones(num_new_visibles) * position, dtype=int),
            new_data_mean,
            axis=1,
        ).astype(self.dtype, copy=False)
        self._data_std = np.insert(
            self._data_std,
            np.array(np.ones(num_new_visibles) * position, dtype=int),
            new_data_std,
            axis=1,
        ).astype(self.dtype, copy=False)

        if isinstance(initial_weights, str) and initial_weights == "AUTO":
            new_w = (2.0 * np.random.rand(num_new_visibles, self.output_dim) - 1.0) * (
                4.0
                * np.sqrt(6.0 / (self.input_dim + self.output_dim + num_new_visibles))
            )
        elif np.isscalar(initial_weights):
            new_w = np.random.randn(num_new_visibles, self.output_dim) * initial_weights
        else:
            new_w = initial_weights

        self.w = np.insert(
            self.w,
            np.array(np.ones(num_new_visibles) * position, dtype=int),
            new_w,
            axis=0,
        ).astype(self.dtype, copy=False)

        if isinstance(initial_offsets, str) and initial_offsets == "AUTO":
            if data is not None:
                new_ov = new_data_mean
            else:
                new_ov = 0.5 * np.ones((1, num_new_visibles))
        elif np.isscalar(initial_offsets):
            new_ov = np.zeros((1, num_new_visibles)) + initial_offsets
        else:
            new_ov = initial_offsets

        self.ov = np.insert(
            self.ov,
            np.array(np.ones(num_new_visibles) * position, dtype=int),
            new_ov,
            axis=1,
        ).astype(self.dtype, copy=False)

        if isinstance(initial_bias, str) and initial_bias == "AUTO":
            if data is not None:
                new_bv = np.zeros((1, num_new_visibles))
            else:
                new_bv = new_data_mean
        elif np.isscalar(initial_bias):
            new_bv = np.zeros((1, num_new_visibles)) + initial_bias
        else:
            new_bv = initial_bias

        self.bv = np.insert(
            self.bv,
            np.array(np.ones(num_new_visibles) * position, dtype=int),
            new_bv,
            axis=1,
        ).astype(self.dtype, copy=False)

        self.input_dim = self.w.shape[0]

    def _remove_visible_units(self, indices):
        """Removes the visible units whose indices are given."""
        idx = np.array(indices)
        self.w = np.delete(self.w, idx, axis=0)
        self.bv = np.delete(self.bv, idx, axis=1)
        self.ov = np.delete(self.ov, idx, axis=1)
        self._data_mean = np.delete(self._data_mean, idx, axis=1)
        self._data_std = np.delete(self._data_std, idx, axis=1)
        self.input_dim = self.w.shape[0]

    ########################################################################
    # Utilities
    ########################################################################

    def get_parameters(self):
        """This function returns all model parameters in a list."""
        return [self.w, self.bv, self.bh]

    def update_parameters(self, updates):
        """This function updates all parameters given the updates derived by the training methods."""
        param_list = self.get_parameters()
        for i, p in enumerate(param_list):
            param_list[i] = _torch_add_inplace(p, updates[i])
        self.w, self.bv, self.bh = param_list

    def update_offsets(
        self,
        new_visible_offsets=0.0,
        new_hidden_offsets=0.0,
        update_visible_offsets=1.0,
        update_hidden_offsets=1.0,
    ):
        """| This function updates the visible and hidden offsets.
           | --> update_offsets(0,0,1,1) reparameterizes to the normal binary RBM.

        :param new_visible_offsets: New visible means.
        :type new_visible_offsets: numpy arrays [1, input dim]

        :param new_hidden_offsets: New hidden means.
        :type new_hidden_offsets: numpy arrays [1, output dim]

        :param update_visible_offsets: Update/Shifting factor for the visible means.
        :type update_visible_offsets: float

        :param update_hidden_offsets: Update/Shifting factor for the hidden means.
        :type update_hidden_offsets: float
        """
        if update_hidden_offsets != 0.0:
            diff_h = new_hidden_offsets - self.oh
            delta_bv = _torch_dot_add(diff_h, self.w.T, bias_np=None, transpose_b=False)
            self.bv = _torch_add_inplace(self.bv, update_hidden_offsets * delta_bv)
            self.oh = (
                1.0 - update_hidden_offsets
            ) * self.oh + update_hidden_offsets * new_hidden_offsets

        if update_visible_offsets != 0.0:
            diff_v = new_visible_offsets - self.ov
            delta_bh = _torch_dot_add(diff_v, self.w, bias_np=None, transpose_b=False)
            self.bh = _torch_add_inplace(self.bh, update_visible_offsets * delta_bh)
            self.ov = (
                1.0 - update_visible_offsets
            ) * self.ov + update_visible_offsets * new_visible_offsets


class StackOfBipartiteGraphs(object):
    """Stacked network layers"""

    def __init__(self, list_of_layers):
        """Initializes the network with auto encoders.

        :param list_of_layers: List of Layers i.e. BipartiteGraph.
        :type list_of_layers: list
        """
        self._layers = list_of_layers
        self.input_dim = None
        self.output_dim = None
        self.states = [None]
        if len(list_of_layers) > 0:
            self.states = [None for _ in range(len(list_of_layers) + 1)]
            self._check_network()
            self.input_dim = self._layers[0].input_dim
            self.output_dim = self._layers[-1].output_dim

    def _check_network(self):
        """Check whether the network is consistent and raise an exception if it is not the case."""
        for i in range(1, len(self._layers)):
            if self._layers[i - 1].output_dim != self._layers[i].input_dim:
                raise Exception(
                    "Output_dim of layer "
                    + str(i - 1)
                    + " has to match input_dim of layer "
                    + str(i)
                    + "!"
                )

    @property
    def depth(self):
        """Networks depth/ number of layers + 1 states."""
        return len(self.states)

    @property
    def num_layers(self):
        """Networks depth/ number of layers."""
        return len(self._layers)

    def __getitem__(self, key):
        """Indexing returns the layer at index 'key'."""
        return self._layers[key]

    def __setitem__(self, key, value):
        """Replaces layer at index 'key' with 'value'."""
        if (
            value.input_dim == self._layers[key].input_dim
            and value.output_dim == self._layers[key].output_dim
        ):
            self._layers[key] = value
        else:
            raise Exception(f"New model at layer {key} has wrong dimensionality!")

    def append_layer(self, layer):
        """Appends the model to the network."""
        self._layers.append(layer)
        self.states.append(None)
        self.output_dim = layer.output_dim
        self._check_network()

    def pop_last_layer(self):
        """Removes/pops the last layer in the network."""
        if len(self._layers) > 0:
            self._layers.pop()
            self.states.pop()
        if len(self._layers) > 0:
            self.input_dim = self._layers[0].input_dim
            self.output_dim = self._layers[-1].output_dim
        else:
            self.input_dim = None
            self.output_dim = None
        self._check_network()

    def save(self, path, save_states=False):
        """Saves the network.

        :param path: Filename+path.
        :type path: string.

        :param save_states: If true the current states are saved.
        :type save_states: bool
        """
        if not save_states:
            for c in range(len(self.states)):
                self.states[c] = None
        save_object(self, path)

    def forward_propagate(self, input_data):
        """Propagates the data through the network.

        :param input_data: Input data.
        :type input_data: numpy array [batchsize x input dim]

        :return: Output of the network.
        :rtype: numpy array [batchsize x output dim]
        """
        if input_data.shape[1] != self.input_dim:
            raise Exception(
                "Input dimensionality mismatch with first layer's input_dim!"
            )
        self.states[0] = input_data
        for i, layer in enumerate(self._layers):
            self.states[i + 1] = layer.hidden_activation(self.states[i])
        return self.states[len(self._layers)]

    def backward_propagate(self, output_data):
        """Propagates the output back through the input.

        :param output_data: Output data.
        :type output_data: numpy array [batchsize x output dim]

        :return: Input of the network.
        :rtype: numpy array [batchsize x input dim]
        """
        if output_data.shape[1] != self.output_dim:
            raise Exception(
                "Output dimensionality mismatch with last layer's output_dim!"
            )
        self.states[-1] = output_data
        for layer in range(len(self._layers), 0, -1):
            self.states[layer - 1] = self._layers[layer - 1].visible_activation(
                self.states[layer]
            )
        return self.states[0]

    def reconstruct(self, input_data):
        """Reconstructs the data by propagating the data to the output and back to the input."""
        return self.backward_propagate(self.forward_propagate(input_data))
