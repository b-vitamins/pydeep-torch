"""
Different kind of cost functions and their derivatives.
Now refactored to use PyTorch internally, returning NumPy arrays
(or scalars) for drop-in compatibility.

:Implemented:
    - Squared error
    - Absolute error
    - Cross entropy
    - Negative Log-likelihood

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

import numpy as np
import torch

MIN_VALUE = 1e-10


def _ensure_torch_double(arr):
    """Convert a NumPy array (or list) to a torch double tensor."""
    return torch.as_tensor(arr, dtype=torch.float64)


def _np_clip(x_np, a_min, a_max):
    """NumPy-like clip but in torch, returning torch tensor."""
    x_t = _ensure_torch_double(x_np)
    return torch.clamp(x_t, min=a_min, max=a_max)


class SquaredError(object):
    """Mean Squared error."""

    @classmethod
    def f(cls, x, t):
        """
        0.5 * sum((x - t)^2, axis=1).
        Returns shape (N,) if x is shape (N,D).
        """
        x_t = _ensure_torch_double(x)
        t_t = _ensure_torch_double(t)
        diff = x_t - t_t
        # sum along axis=1
        val = 0.5 * torch.sum(diff * diff, dim=1)
        return val.cpu().numpy()

    @classmethod
    def df(cls, x, t):
        """
        Derivative => (x - t), same shape as x.
        """
        x_t = _ensure_torch_double(x)
        t_t = _ensure_torch_double(t)
        diff = x_t - t_t
        return diff.cpu().numpy()


class AbsoluteError(object):
    """Absolute error."""

    @classmethod
    def f(cls, x, t):
        """
        sum(|x - t|, axis=1).
        Shape => (N,) for x shape (N,D).
        """
        x_t = _ensure_torch_double(x)
        t_t = _ensure_torch_double(t)
        val = torch.sum(torch.abs(x_t - t_t), dim=1)
        return val.cpu().numpy()

    @classmethod
    def df(cls, x, t):
        """
        sign(x - t).
        Same shape as x.
        """
        x_t = _ensure_torch_double(x)
        t_t = _ensure_torch_double(t)
        diff_t = x_t - t_t
        # sign in torch => torch.sign (returns -1/0/+1).
        # The old code doesn't produce 0 for exact zero, it does np.sign(...).
        # We'll replicate that by default (PyTorch's sign gives 0 if diff=0).
        # That is the same as NumPy sign. So it's fine.
        s = torch.sign(diff_t)
        return s.cpu().numpy()


class CrossEntropyError(object):
    """Cross entropy functions."""

    @classmethod
    def f(cls, x, t):
        """
        - sum( t*log(x) + (1-t)*log(1-x), axis=1 ), with x clipped.
        Returns shape (N,).
        """
        # clip in torch:
        x_clipped = torch.clamp(
            _ensure_torch_double(x), min=MIN_VALUE, max=1.0 - MIN_VALUE
        )
        t_t = _ensure_torch_double(t)
        # cross-entropy
        # - sum( t log(x) + (1-t) log(1-x), axis=1 )
        term = t_t * torch.log(x_clipped) + (1.0 - t_t) * torch.log(1.0 - x_clipped)
        val = -torch.sum(term, dim=1)
        return val.cpu().numpy()

    @classmethod
    def df(cls, x, t):
        """
        Derivative => - t/x + (1 - t)/(1 - x), with x clipped.
        """
        x_clipped = torch.clamp(
            _ensure_torch_double(x), min=MIN_VALUE, max=1.0 - MIN_VALUE
        )
        t_t = _ensure_torch_double(t)
        dx = -t_t / x_clipped + (1.0 - t_t) / (1.0 - x_clipped)
        return dx.cpu().numpy()


class NegLogLikelihood(object):
    """Negative log likelihood function."""

    @classmethod
    def f(cls, x, t):
        """
        - sum( t log(x), axis=1 ).
        x clipped. Returns shape (N,).
        """
        x_clipped = torch.clamp(
            _ensure_torch_double(x), min=MIN_VALUE, max=1.0 - MIN_VALUE
        )
        t_t = _ensure_torch_double(t)
        # cost => - sum( t * log(x), axis=1 )
        val = -torch.sum(t_t * torch.log(x_clipped), dim=1)
        return val.cpu().numpy()

    @classmethod
    def df(cls, x, t):
        """
        Derivative => - t/x, with x clipped.
        Same shape as x.
        """
        x_clipped = torch.clamp(
            _ensure_torch_double(x), min=MIN_VALUE, max=1.0 - MIN_VALUE
        )
        t_t = _ensure_torch_double(t)
        dx = -t_t / x_clipped
        return dx.cpu().numpy()
