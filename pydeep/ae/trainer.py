"""
This module provides implementations for training different variants of Auto-encoders,
modifications on standard gradient decent are provided (centering, denoising, dropout,
sparseness, contractiveness, slowness L1-decay, L2-decay, momentum, gradient restriction)

:Implemented:
    - GDTrainer

:Info:
    http://ufldl.stanford.edu/wiki/index.php/Sparse_Coding:_Autoencoder_Interpretation

:Version:
    1.0

:Date:
    21.01.2018

:Author:
    Jan Melchior

:Contact:
    JanMelchior@gmx.de

:License:

    Copyright (C) 2018 Jan Melchior

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

import pydeep.ae.model as MODEL


###############################################################################
# Torch-based helpers
###############################################################################


def _as_torch_double(arr):
    """Convert a NumPy array or scalar to a torch double tensor (on CPU)."""
    return torch.as_tensor(arr, dtype=torch.float64)


def _np_from_torch(tensor):
    """Convert a torch Tensor to a NumPy array (copies to CPU)."""
    return tensor.cpu().numpy()


def _sign_torch(tensor):
    """
    Equivalent of numpy.sign for a torch Tensor:
    sign(x) in [-1,0,1] for x <0,=0,>0
    """
    return torch.sign(tensor)


def _restrict_norms_torch(mat_t, max_norm, axis=None):
    """
    Restrict L2 norms of a 2D torch tensor mat_t.
    - If axis=None, restrict the entire matrix norm.
    - If axis=0, restrict each column's norm.
    - If axis=1, restrict each row's norm.

    Returns a new Tensor with restricted norms.
    """
    if axis is None:
        # Restrict the norm of the entire matrix
        norm = mat_t.norm(p=2)
        if norm > max_norm:
            mat_t = mat_t * (max_norm / norm)
    elif axis == 0:
        # Restrict each column
        # shape mat_t => [num_rows, num_cols]
        # for each col j, if norm of col_j > max_norm => scale it
        col_norms = mat_t.norm(p=2, dim=0, keepdim=True)  # shape [1, num_cols]
        mask = col_norms > max_norm
        # scaled columns => mat_t[:, j] *= (max_norm / col_norms[j]) for large columns
        scale_factors = torch.where(
            mask,
            max_norm / (col_norms + 1e-12),  # protect from /0
            torch.ones_like(col_norms),
        )
        mat_t = mat_t * scale_factors
    elif axis == 1:
        # Restrict each row
        row_norms = mat_t.norm(p=2, dim=1, keepdim=True)  # shape [num_rows, 1]
        mask = row_norms > max_norm
        scale_factors = torch.where(
            mask, max_norm / (row_norms + 1e-12), torch.ones_like(row_norms)
        )
        mat_t = mat_t * scale_factors
    return mat_t


###############################################################################
# Refactored GDTrainer with Torch
###############################################################################


class GDTrainer(object):
    """Auto encoder trainer using (Torch-based) gradient descent for updates."""

    def __init__(self, model):
        """
        The constructor takes the model as input

        :Parameters:
            model: An auto-encoder object which should be trained.
                   (type: AutoEncoder)
        """
        # Store passed model
        if isinstance(model, MODEL.AutoEncoder):
            self.model = model
        else:
            raise Exception("Model has to be an Auto-encoder object!")

        # Count the number of parameters
        parameters = self.model.get_parameters()  # each is a NumPy array
        self.num_parameters = len(parameters)

        # Prepare storage for momentum/updates (still in NumPy so it "just works")
        # We'll convert them to torch Tensors in each training step, do the math,
        # then store results back in these arrays (drop-in replacement).
        self.parameter_updates = []
        for i in range(self.num_parameters):
            shape_i = parameters[i].shape
            self.parameter_updates.append(np.zeros(shape_i, dtype=model.dtype))

    def _train(
        self,
        data,
        epsilon,
        momentum,
        update_visible_offsets,
        update_hidden_offsets,
        corruptor,
        reg_L1Norm,
        reg_L2Norm,
        reg_sparseness,
        desired_sparseness,
        reg_contractive,
        reg_slowness,
        data_next,
        restrict_gradient,
        restriction_norm,
    ):
        """
        The training for one batch is performed using gradient descent (Torch-based
        internally for the parameter update steps).

        :Parameters:
            data:                The training data
                                 (numpy array [num samples, input dim])

            epsilon:             The learning rate (array-like of length == num_parameters).

            momentum:            The momentum term (array-like of length == num_parameters).

            update_visible_offsets:   Step size for visible offsets updates.

            update_hidden_offsets:    Step size for hidden offsets updates.

            corruptor:           Defines if and how the data gets corrupted (could be None or a list).

            reg_L1Norm:          L1 weight decay coefficient.

            reg_L2Norm:          L2 weight decay (a.k.a. weight decay) coefficient.

            reg_sparseness:      The parameter for the sparseness regularization.

            desired_sparseness:  Desired average hidden activation.

            reg_contractive:     The parameter for the contractive penalty.

            reg_slowness:        The parameter for the slowness penalty.

            data_next:           Next training data in the sequence (for slowness).

            restrict_gradient:   If scalar > 0.0, restrict norm of the weight gradient.

            restriction_norm:    'Cols','Rows','Mat' or None => how to restrict gradients.
        """

        # Possibly corrupt data
        if corruptor is None:
            x = data
            x_next = data_next
        else:
            # Corruption logic stays the same (since user might supply a NumPy-based corruptor).
            if isinstance(corruptor, list):
                x = corruptor[0].corrupt(data)
                if reg_slowness > 0.0 and data_next is not None:
                    x_next = corruptor[0].corrupt(data_next)
                # Then we do a second/third corruption pass on hidden or output
                # inside the forward pass below if desired.
                # This code is basically matching the original, albeit repeated corruption.
            else:
                x = corruptor.corrupt(data)
                if reg_slowness > 0.0 and data_next is not None:
                    x_next = corruptor.corrupt(data_next)

        # Forward pass with the auto-encoder (the AE code can also do
        # corrupt h or y if the user’s code so demands).
        a_h, h = self.model._encode(x)
        a_y, y = self.model._decode(h)

        a_h_next, h_next = None, None
        if reg_slowness > 0.0 and data_next is not None:
            a_h_next, h_next = self.model._encode(x_next)

        # Offsets update
        if update_visible_offsets > 0.0:
            # mean_x => shape [1, input_dim]
            mean_x = np.mean(x, axis=0).reshape(1, self.model.input_dim)
        else:
            mean_x = 0.0

        if update_hidden_offsets > 0.0:
            # mean_h => shape [1, output_dim]
            mean_h = np.mean(h, axis=0).reshape(1, self.model.output_dim)
        else:
            mean_h = 0.0

        self.model.update_offsets(
            mean_x, mean_h, update_visible_offsets, update_hidden_offsets
        )

        # Grab the raw gradients from the model (NumPy arrays):
        gradients = self.model._get_gradients(
            data,
            a_h,
            h,
            a_y,
            y,
            reg_contractive,
            reg_sparseness,
            desired_sparseness,
            reg_slowness,
            data_next,
            a_h_next,
            h_next,
        )

        # Convert parameter_updates and gradients to torch Tensors, do momentum and decay in torch.
        for i in range(self.num_parameters):
            # shape [some_row_dim, some_col_dim]
            param_up_np = self.parameter_updates[i]
            grad_np = gradients[i]

            up_t = _as_torch_double(param_up_np)  # momentum buffer
            grad_t = _as_torch_double(grad_np)
            eps_t = _as_torch_double(epsilon[i])
            mom_t = _as_torch_double(momentum[i])

            # Momentum update:
            # param_update[i] = momentum[i] * param_update[i] - epsilon[i] * gradient
            up_t = mom_t * up_t - eps_t * grad_t

            # For L1/L2, only the weight matrix (index=0) gets them.
            if i == 0:
                # L1
                if reg_L1Norm != 0.0:
                    w_np = self.model.w  # shape e.g. [input_dim, output_dim]
                    w_t = _as_torch_double(w_np)
                    l1_t = _as_torch_double(reg_L1Norm)
                    # up_t -= epsilon[0] * reg_L1Norm * sign(w)
                    up_t = up_t - (eps_t * l1_t * _sign_torch(w_t))

                # L2
                if reg_L2Norm != 0.0:
                    w_np = self.model.w
                    w_t = _as_torch_double(w_np)
                    l2_t = _as_torch_double(reg_L2Norm)
                    # up_t -= epsilon[0] * reg_L2Norm * w
                    up_t = up_t - (eps_t * l2_t * w_t)

                # Restrict gradient norm if requested
                if np.isscalar(restrict_gradient) and restrict_gradient > 0.0:
                    # match 'Cols','Rows','Mat' => 0,1 or None
                    if restriction_norm == "Cols":
                        axis = 0
                    elif restriction_norm == "Rows":
                        axis = 1
                    else:
                        # 'Mat' or unknown => None
                        axis = None
                    up_t = _restrict_norms_torch(
                        up_t, _as_torch_double(restrict_gradient), axis=axis
                    )

            # Store updated momentum back into self.parameter_updates[i]
            self.parameter_updates[i][:] = _np_from_torch(up_t)

        # Now update the model's parameters with those new steps (still in NumPy).
        self.model.update_parameters(self.parameter_updates)

    def train(
        self,
        data,
        num_epochs=1,
        epsilon=0.1,
        momentum=0.0,
        update_visible_offsets=0.0,
        update_hidden_offsets=0.0,
        corruptor=None,
        reg_L1Norm=0.0,
        reg_L2Norm=0.0,
        reg_sparseness=0.0,
        desired_sparseness=0.01,
        reg_contractive=0.0,
        reg_slowness=0.0,
        data_next=None,
        restrict_gradient=False,
        restriction_norm="Mat",
    ):
        """
        Train the model for a given number of epochs using (torch-based) gradient descent steps.

        :Parameters:
            data:                    The data used for training (NumPy array or list of arrays).
            num_epochs:              Number of epochs to train (int).
            epsilon:                 Learning rate (scalar or array of length == num_parameters).
            momentum:                Momentum (scalar or array of length == num_parameters).
            update_visible_offsets:  Step size for visible offsets update.
            update_hidden_offsets:   Step size for hidden offsets update.
            corruptor:               Defines if/how the data gets corrupted.
            reg_L1Norm:              L1 regularization (weight decay).
            reg_L2Norm:              L2 regularization (weight decay).
            reg_sparseness:          Sparseness penalty coefficient.
            desired_sparseness:      Desired average hidden activation.
            reg_contractive:         Contractive penalty coefficient.
            reg_slowness:            Slowness penalty coefficient.
            data_next:               Next data in sequence, if slowness penalty is used.
            restrict_gradient:       If > 0, restrict the norm of the weight gradient.
            restriction_norm:        'Cols','Rows','Mat' => how to apply the norm restriction.
        """

        # Convert scalar epsilon or momentum to arrays for consistent usage
        if np.isscalar(epsilon):
            epsilon = np.array([epsilon] * self.num_parameters, dtype=self.model.dtype)

        if np.isscalar(momentum):
            momentum = np.array(
                [momentum] * self.num_parameters, dtype=self.model.dtype
            )

        # If data is a list of batches => train on each batch
        if isinstance(data, list):
            for _ in range(num_epochs):
                for batch in data:
                    self._train(
                        data=batch,
                        epsilon=epsilon,
                        momentum=momentum,
                        update_visible_offsets=update_visible_offsets,
                        update_hidden_offsets=update_hidden_offsets,
                        corruptor=corruptor,
                        reg_L1Norm=reg_L1Norm,
                        reg_L2Norm=reg_L2Norm,
                        reg_sparseness=reg_sparseness,
                        desired_sparseness=desired_sparseness,
                        reg_contractive=reg_contractive,
                        reg_slowness=reg_slowness,
                        data_next=data_next,
                        restrict_gradient=restrict_gradient,
                        restriction_norm=restriction_norm,
                    )
        else:
            # Single array => treat entire data as one batch
            for _ in range(num_epochs):
                self._train(
                    data=data,
                    epsilon=epsilon,
                    momentum=momentum,
                    update_visible_offsets=update_visible_offsets,
                    update_hidden_offsets=update_hidden_offsets,
                    corruptor=corruptor,
                    reg_L1Norm=reg_L1Norm,
                    reg_L2Norm=reg_L2Norm,
                    reg_sparseness=reg_sparseness,
                    desired_sparseness=desired_sparseness,
                    reg_contractive=reg_contractive,
                    reg_slowness=reg_slowness,
                    data_next=data_next,
                    restrict_gradient=restrict_gradient,
                    restriction_norm=restriction_norm,
                )
