"""
  Feed Forward Neural Network Trainer.

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

import numpy as numx
import torch

import pydeep.base.numpyextension as numxExt
import pydeep.base.activationfunction as AFct
import pydeep.base.costfunction as CFct
import pydeep.base.corruptor as Corr


###############################################################################
# Torch-based helper functions
###############################################################################


def _as_torch_double(arr):
    """Convert a NumPy array (or float) to a torch double Tensor on CPU."""
    return torch.as_tensor(arr, dtype=torch.float64)


def _np_from_torch(tensor):
    """Convert a torch Tensor to a NumPy array (on CPU)."""
    return tensor.cpu().numpy()


def _sign_torch(x_np):
    """Elementwise sign function for a NumPy array, in Torch."""
    x_t = _as_torch_double(x_np)
    return _np_from_torch(torch.sign(x_t))


def _restrict_norms_torch(grad_w_np, max_norm, axis=None):
    """
    Restrict the L2 norm(s) of a 2D gradient matrix `grad_w_np` to <= max_norm,
    akin to numxExt.restrict_norms(...). We'll do it in Torch, then return NumPy.
    - axis=None => restrict entire matrix norm
    - axis=0    => restrict each column
    - axis=1    => restrict each row
    """
    w_t = _as_torch_double(grad_w_np)
    m_t = _as_torch_double(max_norm)
    if axis is None:
        # entire matrix
        norm = w_t.norm(p=2)
        if norm > m_t:
            w_t = w_t * (m_t / norm)
    elif axis == 0:
        # columns
        col_norms = w_t.norm(p=2, dim=0, keepdim=True)
        mask = col_norms > m_t
        factor = m_t / (col_norms + 1e-12)
        scale = torch.where(mask, factor, torch.ones_like(col_norms))
        w_t = w_t * scale
    elif axis == 1:
        # rows
        row_norms = w_t.norm(p=2, dim=1, keepdim=True)
        mask = row_norms > m_t
        factor = m_t / (row_norms + 1e-12)
        scale = torch.where(mask, factor, torch.ones_like(row_norms))
        w_t = w_t * scale
    return _np_from_torch(w_t)


###############################################################################
# GDTrainer
###############################################################################


class GDTrainer(object):
    """Gradient decent feed forward neural network trainer."""

    def __init__(self, model):
        """
        Constructor takes a model.

        :Parameters:
            model: FNN model to train.
                  -type: FNN model.
        """
        self.model = model
        # Storage variable for the old gradient
        # (Now just keep them in NumPy so user code remains the same.)
        self._old_grad = []
        for layer in self.model.layers:
            w_shape = (layer.input_dim, layer.output_dim)
            b_shape = (1, layer.output_dim)
            self._old_grad.append(
                [
                    numx.zeros(w_shape, dtype=numx.float64),
                    numx.zeros(b_shape, dtype=numx.float64),
                ]
            )

    def calculate_errors(self, output_label):
        """
        Calculates the errors for the output of the model and given output_labels.
        You need to call model.forward_propagate before!

        :Parameters:

            output_label: numpy array containing the labels for the network output.
                         -type: list of None and/or numpy arrays

        :return:
            Bool array 0=True if prediction was correct, 1=False otherwise.
           -type: numpy array [batchsize, 1]
        """
        # Get the index of the maximum value along axis 1 from label and output and compare it
        return numxExt.compare_index_of_max(
            self.model.layers[self.model.num_layers - 1].temp_a, output_label
        )

    def check_setup(
        self,
        data,
        labels,
        costs,
        reg_costs,
        epsilon,
        momentum,
        update_offsets,
        corruptor,
        reg_L1Norm,
        reg_L2Norm,
        reg_sparseness,
        desired_sparseness,
        costs_sparseness,
        restrict_gradient,
        restriction_norm,
    ):
        """
        The function checks for valid training and network configuration.
        Warning are printed if a valid is wrong or suspicious.

        ...
        (Docstring unchanged)
        ...
        """
        # Original code, unchanged
        failed = False
        if data.shape[1] != self.model.input_dim:
            print(Warning("Data dimension does not match the models output dimension"))
            failed = True
        if labels[len(labels) - 1].shape[1] != self.model.output_dim:
            print(
                Warning(
                    "Labels["
                    + str(len(labels) - 1)
                    + "] dimension does not match the models output dimension"
                )
            )
            failed = True

        if (
            not numx.isscalar(reg_costs[len(reg_costs) - 1])
            or reg_costs[len(reg_costs) - 1] != 1
        ):
            print(
                Warning(
                    "reg_costs["
                    + str(len(reg_costs) - 1)
                    + "], which is the main cost should be 1.0"
                )
            )
            failed = True
            if reg_costs[len(reg_costs) - 1] > 0.0:
                if labels[len(reg_costs) - 1] is None:
                    print(
                        Warning(
                            "reg_costs["
                            + str(len(reg_costs) - 1)
                            + "] > 0 then labels["
                            + str(len(reg_costs) - 1)
                            + "] has to be an array!"
                        )
                    )
                    failed = True
        if labels[len(labels) - 1] is None:
            print(
                Warning("labels[" + str(len(labels) - 1) + "] has to contain values.")
            )
            failed = True

        if (
            restriction_norm != "Cols"
            and restriction_norm != "Rows"
            and restriction_norm != "Mat"
        ):
            print(Warning("restriction_norm has to be Cols, Rows or Mat"))
            failed = True

        if not isinstance(restrict_gradient, list):
            print(
                Warning(
                    "restrict_gradient has to be a list of length "
                    + str(self.model.num_layers)
                )
            )
            failed = True
        if not isinstance(epsilon, list):
            print(
                Warning(
                    "epsilon has to be a list of length " + str(self.model.num_layers)
                )
            )
            failed = True
        if not isinstance(momentum, list):
            print(
                Warning(
                    "momentum has to be a list of length " + str(self.model.num_layers)
                )
            )
            failed = True
        if not isinstance(update_offsets, list):
            print(
                Warning(
                    "update_offsets has to be a list of length "
                    + str(self.model.num_layers)
                )
            )
            failed = True
        if not isinstance(reg_L1Norm, list):
            print(
                Warning(
                    "reg_L1Norm has to be a list of length "
                    + str(self.model.num_layers)
                )
            )
            failed = True
        if not isinstance(reg_L2Norm, list):
            print(
                Warning(
                    "reg_L2Norm has to be a list of length "
                    + str(self.model.num_layers)
                )
            )
            failed = True
        if not isinstance(corruptor, list) and corruptor is not None:
            print(
                Warning(
                    "corruptor has to be None or a list of length "
                    + str(self.model.num_layers)
                )
            )
            failed = True
        if not isinstance(reg_sparseness, list):
            print(
                Warning(
                    "reg_sparseness has to be a list of length "
                    + str(self.model.num_layers)
                )
            )
            failed = True
        if not isinstance(desired_sparseness, list):
            print(
                Warning(
                    "desired_sparseness has to be a list of length "
                    + str(self.model.num_layers)
                )
            )
            failed = True
        if not isinstance(costs_sparseness, list):
            print(
                Warning(
                    "costs_sparseness has to be a list of length "
                    + str(self.model.num_layers)
                )
            )
            failed = True
        if not isinstance(reg_costs, list):
            print(
                Warning(
                    "reg_costs has to be a list of length " + str(self.model.num_layers)
                )
            )
            failed = True
        if not isinstance(costs, list):
            print(
                Warning(
                    "costs has to be a list of length " + str(self.model.num_layers)
                )
            )
            failed = True
        if not isinstance(labels, list):
            print(
                Warning(
                    "labels has to be a list of length " + str(self.model.num_layers)
                )
            )
            failed = True

        if len(epsilon) != self.model.num_layers:
            print(Warning("len(epsilon) has to be equal to num _layers"))
            failed = True
        if len(momentum) != self.model.num_layers:
            print(Warning("len(momentum) has to be equal to num _layers"))
            failed = True
        if len(update_offsets) != self.model.num_layers:
            print(Warning("len(update_offsets) has to be equal to num _layers"))
            failed = True
        if len(reg_L1Norm) != self.model.num_layers:
            print(Warning("len(reg_L1Norm) has to be equal to num _layers"))
            failed = True
        if len(reg_L2Norm) != self.model.num_layers:
            print(Warning("len(reg_L2Norm) has to be equal to num _layers"))
            failed = True
        if corruptor is not None:
            if len(corruptor) != self.model.num_layers + 1:
                print(Warning("len(corruptor) has to be equal to num _layers+1"))
                failed = True
        if len(reg_sparseness) != self.model.num_layers:
            print(Warning("len(reg_sparseness) has to be equal to num _layers"))
            failed = True
        if len(desired_sparseness) != self.model.num_layers:
            print(Warning("len(desired_sparseness) has to be equal to num _layers"))
            failed = True
        if len(costs_sparseness) != self.model.num_layers:
            print(Warning("len(costs_sparseness) has to be equal to num _layers"))
            failed = True
        if len(reg_costs) != self.model.num_layers:
            print(Warning("len(reg_costs) has to be equal to num _layers"))
            failed = True

        if len(costs) != self.model.num_layers:
            print(Warning("len(costs) has to be equal to num _layers"))
            failed = True
        if len(labels) != self.model.num_layers:
            print(Warning("len(labels) has to be equal to num _layers"))
            failed = True

        if corruptor is not None:
            if corruptor[0] is not None and not isinstance(corruptor[0], Corr.Identity):
                print(
                    Warning(
                        "corruptor[" + str(0) + "] has to be None or CFct.CostFunction"
                    )
                )
                failed = True
        for layer in range(self.model.num_layers):
            if epsilon[layer] < 0.0 or epsilon[layer] > 1.0:
                print(
                    Warning(
                        "epsilon["
                        + str(layer)
                        + "] should to be a positive scalar in range [0,1]"
                    )
                )
                failed = True
            if momentum[layer] < 0.0 or momentum[layer] > 1.0:
                print(
                    Warning(
                        "momentum["
                        + str(layer)
                        + "] should to be a positive scalar in range [0,1]"
                    )
                )
                failed = True
            if update_offsets[layer] < 0.0 or update_offsets[layer] > 1:
                print(
                    Warning(
                        "reg_L2Norm["
                        + str(layer)
                        + "] has to be a positive scalar in range [0,1]"
                    )
                )
                failed = True
            if reg_L1Norm[layer] < 0.0 or reg_L1Norm[layer] > 0.001:
                print(
                    Warning(
                        "reg_L1Norm["
                        + str(layer)
                        + "] should to be a positive scalar in range [0,0.001]"
                    )
                )
                failed = True
            if reg_L2Norm[layer] < 0.0 or reg_L2Norm[layer] > 0.001:
                print(
                    Warning(
                        "reg_L2Norm["
                        + str(layer)
                        + "] should to be a positive scalar in range [0,0.001]"
                    )
                )
                failed = True
            if corruptor is not None:
                if corruptor[layer + 1] is not None and not isinstance(
                    corruptor[layer + 1], Corr.Identity
                ):
                    print(
                        Warning(
                            "corruptor["
                            + str(layer + 1)
                            + "] has to be None or CFct.CostFunction"
                        )
                    )
                    failed = True
            if not numx.isscalar(reg_sparseness[layer]) or reg_sparseness[layer] < 0.0:
                print(
                    Warning(
                        "reg_sparseness[" + str(layer) + "] has to be a positive scalar"
                    )
                )
                failed = True
            if reg_sparseness[layer] > 0.0:
                if reg_sparseness[layer] > 1.0:
                    print(
                        Warning(
                            "reg_sparseness["
                            + str(layer)
                            + "] should not be greater than 1"
                        )
                    )
                    failed = True
                if (
                    not numx.isscalar(desired_sparseness[layer])
                    or not desired_sparseness[layer] > 0.0
                ):
                    print(
                        Warning(
                            "reg_sparseness["
                            + str(layer)
                            + "] > 0 then desired_sparseness["
                            + str(layer)
                            + "] has to be a positive scalar!"
                        )
                    )
                    failed = True
                if not costs_sparseness[layer] is not None:
                    print(
                        Warning(
                            "costs_sparseness[" + str(layer) + "] should not be None"
                        )
                    )
                    failed = True
            if not numx.isscalar(reg_costs[layer]) or reg_costs[layer] < 0.0:
                print(
                    Warning("reg_costs[" + str(layer) + "] has to be a positive scalar")
                )
                failed = True
            if reg_costs[layer] > 0.0:
                if reg_costs[layer] > 1.0:
                    print(
                        Warning(
                            "reg_costs[" + str(layer) + "] should not be greater than 1"
                        )
                    )
                    failed = True
                if labels[layer] is None:
                    print(
                        Warning(
                            "reg_costs["
                            + str(layer)
                            + "] > 0 then labels["
                            + str(layer)
                            + "] has to be an array!"
                        )
                    )
                    failed = True
                else:
                    if labels[layer].shape[1] != self.model.layers[layer].output_dim:
                        print(
                            Warning(
                                "Label["
                                + str(layer)
                                + "] dim. does not match layer["
                                + str(layer)
                                + "] output dim"
                            )
                        )
                        failed = True
                if costs[layer] is not None:
                    if (
                        costs[layer] == CFct.CrossEntropyError
                        or costs[layer] == CFct.NegLogLikelihood
                    ) and not (
                        self.model.layers[layer].activation_function == AFct.SoftMax
                        or self.model.layers[layer].activation_function == AFct.Sigmoid
                    ):
                        print(
                            Warning(
                                "Layer "
                                + str(layer)
                                + ": Activation function "
                                + str(self.model.layers[layer].activation_function)
                                + " and cost "
                                + str(costs[layer])
                                + " incompatible"
                            )
                        )
                        failed = True
                else:
                    print(Warning("costs[" + str(layer) + "] should not be None"))
                    failed = True
        return not failed

    def train(
        self,
        data,
        labels,
        costs,
        reg_costs,
        epsilon,
        momentum,
        update_offsets,
        corruptor,
        reg_L1Norm,
        reg_L2Norm,
        reg_sparseness,
        desired_sparseness,
        costs_sparseness,
        restrict_gradient,
        restriction_norm,
    ):
        """
        Train function which performes one step of gradient descent.
        Use check_setup() to check whether your training setup is valid.

        ...
        (Docstring unchanged)
        ...
        """
        # Forward propagate
        _ = self.model.forward_propagate(data=data, corruptor=corruptor)

        # Reparameterize offsets
        for layer_idx in range(len(self.model.layers)):
            self.model.layers[layer_idx].update_offsets(
                shift=update_offsets[layer_idx], new_mean=None
            )

        deltas = None
        # Backprop from top to bottom
        for layer_idx in range(self.model.num_layers - 1, -1, -1):
            deltas = self.model.layers[layer_idx]._get_deltas(
                deltas=deltas,
                labels=labels[layer_idx],
                cost=costs[layer_idx],
                reg_cost=reg_costs[layer_idx],
                desired_sparseness=desired_sparseness[layer_idx],
                cost_sparseness=costs_sparseness[layer_idx],
                reg_sparseness=reg_sparseness[layer_idx],
            )
            if layer_idx > 0:
                deltas = self.model.layers[layer_idx]._backward_propagate()

            # Calculate gradient
            grad_list = self.model.layers[
                layer_idx
            ]._calculate_gradient()  # [gradW, gradB]
            gradW_np, gradB_np = grad_list[0], grad_list[1]

            # L1, L2 regularization
            if reg_L1Norm[layer_idx] > 0.0:
                # gradW += reg_L1Norm * sign(weights)
                sign_w = _sign_torch(self.model.layers[layer_idx].weights)
                gradW_np += reg_L1Norm[layer_idx] * sign_w
            if reg_L2Norm[layer_idx] > 0.0:
                gradW_np += reg_L2Norm[layer_idx] * self.model.layers[layer_idx].weights

            # Convert to torch, apply LR, momentum, gradient restrict in torch
            gradW_t = _as_torch_double(gradW_np)
            gradB_t = _as_torch_double(gradB_np)

            # multiply by epsilon
            eps_t = _as_torch_double(epsilon[layer_idx])
            gradW_t = gradW_t * eps_t
            gradB_t = gradB_t * eps_t

            # Possibly restrict gradient norm
            if (
                numx.isscalar(restrict_gradient[layer_idx])
                and restrict_gradient[layer_idx] > 0
            ):
                if restriction_norm == "Cols":
                    # restrict column norms
                    gradW_np = _restrict_norms_torch(
                        _np_from_torch(gradW_t), restrict_gradient[layer_idx], axis=0
                    )
                    gradW_t = _as_torch_double(gradW_np)
                elif restriction_norm == "Rows":
                    gradW_np = _restrict_norms_torch(
                        _np_from_torch(gradW_t), restrict_gradient[layer_idx], axis=1
                    )
                    gradW_t = _as_torch_double(gradW_np)
                elif restriction_norm == "Mat":
                    gradW_np = _restrict_norms_torch(
                        _np_from_torch(gradW_t), restrict_gradient[layer_idx], axis=None
                    )
                    gradW_t = _as_torch_double(gradW_np)

            # Momentum
            mom_t = _as_torch_double(momentum[layer_idx])

            old_gradW_t = _as_torch_double(self._old_grad[layer_idx][0])
            old_gradB_t = _as_torch_double(self._old_grad[layer_idx][1])

            gradW_t = gradW_t + mom_t * old_gradW_t
            gradB_t = gradB_t + mom_t * old_gradB_t

            # Convert final gradient back to NumPy
            final_gradW_np = _np_from_torch(gradW_t)
            final_gradB_np = _np_from_torch(gradB_t)

            # Update model parameters
            self.model.layers[layer_idx].update_parameters(
                [final_gradW_np, final_gradB_np]
            )

            # Store into self._old_grad
            self._old_grad[layer_idx][0] = final_gradW_np
            self._old_grad[layer_idx][1] = final_gradB_np


###############################################################################
# ADAGDTrainer
###############################################################################


class ADAGDTrainer(GDTrainer):
    """ADA-Gradient decent feed forward neural network trainer."""

    def __init__(self, model, numerical_stabilty=1e-6):
        """
        Constructor takes a model.

        :Parameters:
            model:              FNN model to train.
                               -type: FNN model.

            master_epsilon:     Master/Default learning rate.
                               -type: float.

            numerical_stabilty: Value added to avoid numerical instabilties by division by zero.
                               -type: float.

        """
        self._numerical_stabilty = numerical_stabilty
        super(ADAGDTrainer, self).__init__(model=model)

    def train(
        self,
        data,
        labels,
        costs,
        reg_costs,
        epsilon,
        update_offsets,
        corruptor,
        reg_L1Norm,
        reg_L2Norm,
        reg_sparseness,
        desired_sparseness,
        costs_sparseness,
        restrict_gradient,
        restriction_norm,
    ):
        """
        Train function which performes one step of gradient descent.
        Use check_setup() to check whether your training setup is valid.

        ...
        (Docstring unchanged)
        ...
        """
        # Forward propagate
        _ = self.model.forward_propagate(data=data, corruptor=corruptor)

        # Reparameterize offsets
        for layer_idx in range(len(self.model.layers)):
            self.model.layers[layer_idx].update_offsets(
                shift=update_offsets[layer_idx], new_mean=None
            )

        deltas = None
        # Backprop from top to bottom
        for layer_idx in range(self.model.num_layers - 1, -1, -1):
            deltas = self.model.layers[layer_idx]._get_deltas(
                deltas=deltas,
                labels=labels[layer_idx],
                cost=costs[layer_idx],
                reg_cost=reg_costs[layer_idx],
                desired_sparseness=desired_sparseness[layer_idx],
                cost_sparseness=costs_sparseness[layer_idx],
                reg_sparseness=reg_sparseness[layer_idx],
            )
            if layer_idx > 0:
                deltas = self.model.layers[layer_idx]._backward_propagate()

            # Gradient
            grad_list = self.model.layers[layer_idx]._calculate_gradient()
            gradW_np, gradB_np = grad_list[0], grad_list[1]

            # L1, L2 decay
            if reg_L1Norm[layer_idx] > 0.0:
                sign_w = _sign_torch(self.model.layers[layer_idx].weights)
                gradW_np += reg_L1Norm[layer_idx] * sign_w
            if reg_L2Norm[layer_idx] > 0.0:
                gradW_np += reg_L2Norm[layer_idx] * self.model.layers[layer_idx].weights

            # Convert to torch for ADA updates
            gradW_t = _as_torch_double(gradW_np)
            gradB_t = _as_torch_double(gradB_np)

            old_gradW_t = _as_torch_double(
                self._old_grad[layer_idx][0]
            )  # accum of squares
            old_gradB_t = _as_torch_double(self._old_grad[layer_idx][1])

            # Accumulate squares
            old_gradW_t = old_gradW_t + gradW_t**2
            old_gradB_t = old_gradB_t + gradB_t**2

            # Now gradW /= (num_stability + sqrt(old_gradW))
            stabil_t = _as_torch_double(self._numerical_stabilty)
            gradW_t = gradW_t / (stabil_t + torch.sqrt(old_gradW_t))
            gradB_t = gradB_t / (stabil_t + torch.sqrt(old_gradB_t))

            # Multiply by epsilon
            eps_t = _as_torch_double(epsilon[layer_idx])
            gradW_t = gradW_t * eps_t
            gradB_t = gradB_t * eps_t

            # Possibly restrict gradient
            if numx.isscalar(restrict_gradient) and restrict_gradient > 0:
                if restriction_norm == "Cols":
                    gradW_np = _restrict_norms_torch(
                        _np_from_torch(gradW_t), restrict_gradient, axis=0
                    )
                    gradW_t = _as_torch_double(gradW_np)
                elif restriction_norm == "Rows":
                    gradW_np = _restrict_norms_torch(
                        _np_from_torch(gradW_t), restrict_gradient, axis=1
                    )
                    gradW_t = _as_torch_double(gradW_np)
                elif restriction_norm == "Mat":
                    gradW_np = _restrict_norms_torch(
                        _np_from_torch(gradW_t), restrict_gradient, axis=None
                    )
                    gradW_t = _as_torch_double(gradW_np)

            # Convert final result back to NumPy
            final_gradW_np = _np_from_torch(gradW_t)
            final_gradB_np = _np_from_torch(gradB_t)

            # Update the model parameters
            self.model.layers[layer_idx].update_parameters(
                [final_gradW_np, final_gradB_np]
            )

            # Store the updated accum of squares (ADAGD) in _old_grad
            self._old_grad[layer_idx][0] = _np_from_torch(old_gradW_t)
            self._old_grad[layer_idx][1] = _np_from_torch(old_gradB_t)
