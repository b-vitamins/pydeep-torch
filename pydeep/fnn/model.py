"""
  Feed Forward Neural Network Model.

    .. Note::
    
        Due to computational benefits the common notation for the delta terms is 
        split in a delta term for the common layer and the error signal passed 
        to the layer below. See the following Latex code for details. This allows
        to store all layer depending results in the corresponding layer and avoid
        useless computations without messing up the code.
        .. math::
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

import numpy as np
import torch


###############################################################################
# Torch-based helpers
###############################################################################


def _as_torch_double(arr):
    """Convert a NumPy array (or float) to a torch double tensor on CPU."""
    return torch.as_tensor(arr, dtype=torch.float64)


def _np_from_torch(tensor):
    """Return a NumPy array from a torch Tensor on CPU."""
    return tensor.cpu().numpy()


def _mean_along_axis(x_np, axis=None):
    """
    Compute mean along a given axis using Torch, then return NumPy array or float.
    If axis=None, returns a Python float of the global mean.
    """
    x_t = _as_torch_double(x_np)
    if axis is None:
        return float(torch.mean(x_t).item())
    else:
        out_t = torch.mean(x_t, dim=axis)
        return _np_from_torch(out_t)


###############################################################################
# Model class (with minimal changes to use Torch for means, etc.)
###############################################################################


class Model(object):
    """Model which stores the layers."""

    def __init__(self, layers=[]):
        """
        Constructor takes a list of layers or an empty list.

        :Parameters:
            layers:  List of layers or empty list.
                     -type: list of layers.

        """
        self.layers = layers
        self.num_layers = len(self.layers)
        if self.num_layers > 0:
            self.input_dim = self.layers[0].input_dim
            self.output_dim = self.layers[self.num_layers - 1].output_dim
            self.consistency_check()
        else:
            self.input_dim = 0
            self.output_dim = 0

    def clear_temp_data(self):
        """Sets all temp variables to None."""
        for layer in self.layers:
            layer.clear_temp_data()

    def calculate_cost(
        self,
        labels,
        costs,
        reg_costs,
        desired_sparseness,
        costs_sparseness,
        reg_sparseness,
    ):
        """
        Calculates the cost for given labels. You need to call model.forward_propagate before!

        :Parameters:

            labels:             list of numpy arrays, entries can be None but
                                the last layer needs labels!
                               -type: list of None and/or numpy arrays

            costs:              Cost functions for the layers, entries can be None
                                but the last layer needs a cost function!
                               -type: list of None and/or pydeep.base.costfunction

            reg_costs:          list of scalars controlling the strength
                                of the cost functions.
                               -type: list of scalars

            desired_sparseness: List of desired sparseness values/average hidden activities.
                               -type: list of scalars

            costs_sparseness:   Cost functions for the sparseness, entries can be None.
                               -type: list of None and/or pydeep.base.costfunction

            reg_sparseness:     Strength of the sparseness term.
                               -type: list of scalars

        :return:
            Cost values for the datapoints.
           -type: numpy array [batchsize, 1] or float
        """
        cost = 0.0
        # Add all intermediate costs
        for layer_idx in range(self.num_layers):
            # Sparse penalty
            if reg_sparseness[layer_idx] != 0.0:
                # Use torch-based mean for consistency
                mean_h_t = _as_torch_double(self.layers[layer_idx].temp_a)
                mean_h_t = torch.mean(
                    mean_h_t, dim=0, keepdim=True
                )  # shape [1, out_dim]
                # Convert to NumPy for the cost function
                mean_h = _np_from_torch(mean_h_t)
                # Possibly ensure it's 2D
                mean_h = np.atleast_2d(mean_h)
                cost += reg_sparseness[layer_idx] * costs_sparseness[layer_idx].f(
                    mean_h, desired_sparseness[layer_idx]
                )

            # Standard cost
            if reg_costs[layer_idx] != 0.0:
                cost += reg_costs[layer_idx] * costs[layer_idx].f(
                    self.layers[layer_idx].temp_a, labels[layer_idx]
                )

        return cost

    def consistency_check(self):
        """
        Raises exceptions if the network structure is incorrect,
        e.g. output dim layer 0 != input dim layer 1
        """
        num_layers = len(self.layers)
        for i in range(num_layers - 1):
            if self.layers[i].output_dim != self.layers[i + 1].input_dim:
                raise Exception(
                    "Output dimensions mismatch layer "
                    + str(i)
                    + " and "
                    + str(i + 1)
                    + " !",
                    UserWarning,
                )

    def append_layer(self, layer):
        """
        Appends a layer to the network.

        :Parameters:
            layer:  Neural network layer.
                   -type: Neural network layer.

        """
        if self.num_layers > 0:
            if self.layers[-1].output_dim != layer.input_dim:
                raise Exception(
                    "Output dimensions mismatch last layer and new layer !", UserWarning
                )
        else:
            self.input_dim = layer.input_dim
        self.output_dim = layer.output_dim
        self.layers.append(layer)
        self.num_layers += 1

    def pop_layer(self):
        """Pops the last layer in the network."""
        if self.num_layers > 0:
            self.layers.pop(self.num_layers - 1)
            self.num_layers -= 1
            if self.num_layers > 0:
                self.output_dim = self.layers[self.num_layers - 1].output_dim
            else:
                self.input_dim = 0
                self.output_dim = 0

    def forward_propagate(self, data, corruptor=None):
        """
        Propagates the input data through the network.

        :Parameters:
            data:      Input data to propagate.
                       -type: numpy array [batchsize, self.input_dim]

            corruptor: None or list of corruptors, one for the input followed by
                       one for every hidden layer's output.
                       -type: None or list of corruptors

        :Returns:
            Output of the network. Every unit state is also stored in
            the corresponding layer.
           -type: numpy array [batchsize, self.output_dim]
        """
        if corruptor is None:
            output = data
            for layer_idx in range(self.num_layers):
                output = self.layers[layer_idx].forward_propagate(output)
        else:
            # copy data
            output = np.copy(data)
            for layer_idx in range(self.num_layers):
                if corruptor[layer_idx] is not None:
                    output = corruptor[layer_idx].corrupt(output)
                output = self.layers[layer_idx].forward_propagate(output)
            # Possibly corrupt final output
            if corruptor[self.num_layers] is not None:
                output = corruptor[self.num_layers].corrupt(output)
        return output

    def _get_gradients(
        self,
        labels,
        costs,
        reg_costs,
        desired_sparseness,
        costs_sparseness,
        reg_sparseness,
        check_gradient=False,
    ):
        """
        Calculates the gradient for the network (Used in finit_differences()).
        You need to call model.forward_propagate before!

        :Parameters:

            labels:             list of numpy arrays, entries can be None but
                                the last layer needs labels!
                               -type: list of None and/or numpy arrays

            costs:              Cost functions for the layers, entries can be None
                                but the last layer needs a cost function!
                               -type: list of None and/or pydeep.base.costfunction

            reg_costs:          list of scalars controlling the strength
                                of the cost functions.
                               -type: list of scalars

            desired_sparseness: List of desired sparseness values/average hidden activities.
                               -type: list of scalars

            costs_sparseness:   Cost functions for the sparseness, entries can be None.
                               -type: list of None and/or pydeep.base.costfunction

            reg_sparseness:     Strength of the sparseness term.
                               -type: list of scalars

            check_gradient:     False for gradient checking mode.
                               -type: bool

        :return:
            gradient for the network.
           -type: list of list of numpy arrays
        """
        grad = []
        deltas = None
        # Go from top layer to first
        for layer_idx in range(self.num_layers - 1, -1, -1):
            # Calculate the delta values
            deltas = self.layers[layer_idx]._get_deltas(
                deltas=deltas,
                labels=labels[layer_idx],
                cost=costs[layer_idx],
                reg_cost=reg_costs[layer_idx],
                desired_sparseness=desired_sparseness[layer_idx],
                cost_sparseness=costs_sparseness[layer_idx],
                reg_sparseness=reg_sparseness[layer_idx],
                check_gradient=check_gradient,
            )
            # backprop the error if it is not the bottom layer
            if layer_idx > 0:
                deltas = self.layers[layer_idx]._backward_propagate()

            # Now we can calculate the gradient
            grad.insert(0, self.layers[layer_idx]._calculate_gradient())
        return grad

    def finit_differences(
        self,
        delta,
        data,
        labels,
        costs,
        reg_costs,
        desired_sparseness,
        costs_sparseness,
        reg_sparseness,
    ):
        """
        Calculates the finite differences for the network's gradient.

        :Parameters:

            delta:              Small delta value added to the parameters.
                               -type: float

            data:               Input data of the network.
                               -type: numpy arrays

            labels:             list of numpy arrays, entries can be None but
                                the last layer needs labels!
                               -type: list of None and/or numpy arrays

            costs:              Cost functions for the layers, entries can be None
                                but the last layer needs a cost function!
                               -type: list of None and/or pydeep.base.costfunction

            reg_costs:          list of scalars controlling the strength
                                of the cost functions.
                               -type: list of scalars

            desired_sparseness: List of desired sparseness values/average hidden activities.
                               -type: list of scalars

            costs_sparseness:   Cost functions for the sparseness, entries can be None.
                               -type: list of None and/or pydeep.base.costfunction

            reg_sparseness:     Strength of the sparseness term.
                               -type: list of scalars

        :return:
            Finite differences W,b, max W, max b
           -type: list of list of numpy arrays
        """
        data = np.atleast_2d(data)
        # Lists of the difference
        diffs_w = []
        diffs_b = []
        # Vars for tracking the maximal value
        max_diffb = -99999
        max_diffw = -99999

        # Loop through all layers
        for layer_idx in range(len(self.layers)):
            layer = self.layers[layer_idx]
            diff_w = np.zeros(layer.weights.shape)
            diff_b = np.zeros(layer.bias.shape)

            # Loop over each weight
            for i in range(layer.input_dim):
                for j in range(layer.output_dim):
                    # Only calculate if there's a valid connection
                    if layer.connections is None or (
                        layer.connections is not None and layer.connections[i][j] > 0.0
                    ):
                        # Forward + gradient
                        self.forward_propagate(data=data)
                        grad_w_ij = self._get_gradients(
                            labels=labels,
                            costs=costs,
                            reg_costs=reg_costs,
                            desired_sparseness=desired_sparseness,
                            costs_sparseness=costs_sparseness,
                            reg_sparseness=reg_sparseness,
                            check_gradient=True,
                        )[layer_idx][0][i][j]
                        # + delta
                        layer.weights[i, j] += delta
                        self.forward_propagate(data)
                        E_pos = self.calculate_cost(
                            labels=labels,
                            costs=costs,
                            reg_costs=reg_costs,
                            desired_sparseness=desired_sparseness,
                            costs_sparseness=costs_sparseness,
                            reg_sparseness=reg_sparseness,
                        )
                        # - 2*delta
                        layer.weights[i, j] -= 2 * delta
                        self.forward_propagate(data)
                        E_neg = self.calculate_cost(
                            labels=labels,
                            costs=costs,
                            reg_costs=reg_costs,
                            desired_sparseness=desired_sparseness,
                            costs_sparseness=costs_sparseness,
                            reg_sparseness=reg_sparseness,
                        )
                        # Restore
                        layer.weights[i, j] += delta
                        approx = (E_pos - E_neg) / (2.0 * delta)
                        diff_w[i, j] = np.abs(grad_w_ij - approx)
                        if np.abs(diff_w[i, j]) > max_diffw:
                            max_diffw = np.abs(diff_w[i, j])

            # Loop over each bias
            for j in range(layer.output_dim):
                self.forward_propagate(data)
                grad_b_j = self._get_gradients(
                    labels=labels,
                    costs=costs,
                    reg_costs=reg_costs,
                    desired_sparseness=desired_sparseness,
                    costs_sparseness=costs_sparseness,
                    reg_sparseness=reg_sparseness,
                    check_gradient=True,
                )[layer_idx][1][0][j]
                # + delta
                layer.bias[0, j] += delta
                self.forward_propagate(data)
                E_pos = self.calculate_cost(
                    labels=labels,
                    costs=costs,
                    reg_costs=reg_costs,
                    desired_sparseness=desired_sparseness,
                    costs_sparseness=costs_sparseness,
                    reg_sparseness=reg_sparseness,
                )
                # - 2*delta
                layer.bias[0, j] -= 2 * delta
                self.forward_propagate(data)
                E_neg = self.calculate_cost(
                    labels=labels,
                    costs=costs,
                    reg_costs=reg_costs,
                    desired_sparseness=desired_sparseness,
                    costs_sparseness=costs_sparseness,
                    reg_sparseness=reg_sparseness,
                )
                # restore
                layer.bias[0, j] += delta
                approx = (E_pos - E_neg) / (2.0 * delta)
                diff_b[0, j] = np.abs(grad_b_j - approx)
                if np.abs(diff_b[0, j]) > max_diffb:
                    max_diffb = np.abs(diff_b[0, j])

            diffs_w.append(diff_w)
            diffs_b.append(diff_b)

        return diffs_w, diffs_b, max_diffw, max_diffb
