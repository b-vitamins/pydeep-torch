"""
Different kind of non linear activation functions and their derivatives.
Now internally uses PyTorch for array-based computations, returning NumPy
arrays (or scalars) to preserve drop-in compatibility.

:Implemented:

# Unbounded
    # Linear
        - Identity
    # Piecewise-linear
        - Rectifier
        - RestrictedRectifier (hard bounded)
        - LeakyRectifier
    # Soft-linear
        - ExponentialLinear
        - SigmoidWeightedLinear
        - SoftPlus
# Bounded
    # Step
        - Step
    # Soft-Step
        - Sigmoid
        - SoftSign
        - HyperbolicTangent
        - SoftMax
        - K-Winner takes all
    # Symmetric, periodic
        - RadialBasis function
        - Sinus

:Info:
    http://en.wikipedia.org/wiki/Activation_function

:Version:
    1.1.1

:Date:
    16.01.2018

:Author:
    Jan Melchior

:Contact:
    JanMelchior@gmx.de

:License:

    Copyright (C) 2018 Jan Melchior

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
from pydeep.base.numpyextension import log_sum_exp


##############################################################################
# Helpers to consistently convert scalar/array to torch, compute, then back.
##############################################################################
def _ensure_torch_double(x):
    """Converts scalar or array 'x' to a torch double tensor."""
    return torch.as_tensor(x, dtype=torch.float64)


def _return_nparray_or_scalar(x, out_t):
    """
    Returns a scalar if x was a scalar, otherwise
    a NumPy array with the same shape as x.
    """
    if np.isscalar(x):
        # single float
        return float(out_t.item())
    else:
        # shape match
        return out_t.cpu().numpy()


##############################################################################
# Unbounded
##############################################################################


# Linear
class Identity(object):
    """Identity function."""

    @classmethod
    def f(cls, x):
        if np.isscalar(x):
            return x
        else:
            # Just return x as-is (NumPy array)
            return x

    @classmethod
    def g(cls, y):
        if np.isscalar(y):
            return y
        else:
            return y

    @classmethod
    def df(cls, x):
        if np.isscalar(x):
            return 1.0
        else:
            return np.ones_like(x)

    @classmethod
    def ddf(cls, x):
        if np.isscalar(x):
            return 0.0
        else:
            return np.zeros_like(x)

    @classmethod
    def dg(cls, y):
        if np.isscalar(y):
            return 1.0
        else:
            return np.ones_like(y)


# Piecewise-linear
class Rectifier(object):
    """Rectifier activation function: max(0, x)."""

    @classmethod
    def f(cls, x):
        if np.isscalar(x):
            return max(0.0, x)
        x_t = _ensure_torch_double(x)
        out_t = torch.clamp(x_t, min=0.0)
        return _return_nparray_or_scalar(x, out_t)

    @classmethod
    def df(cls, x):
        # derivative is 1 if x>0, else 0
        if np.isscalar(x):
            return 1.0 if x > 0.0 else 0.0
        x_t = _ensure_torch_double(x)
        out_t = (x_t > 0.0).to(torch.float64)
        return _return_nparray_or_scalar(x, out_t)

    @classmethod
    def ddf(cls, x):
        # second derivative is 0 everywhere
        return 0.0


class RestrictedRectifier(Rectifier):
    """Restricted Rectifier with an upper limit."""

    def __init__(self, restriction=1.0):
        self.restriction = restriction

    def f(self, x):
        if np.isscalar(x):
            return min(max(0.0, x), self.restriction)
        x_t = _ensure_torch_double(x)
        # clamp to [0, self.restriction]
        out_t = torch.clamp(x_t, min=0.0, max=self.restriction)
        return _return_nparray_or_scalar(x, out_t)

    def df(self, x):
        """1.0 if 0<x<restriction, else 0.0."""
        if np.isscalar(x):
            return float((x > 0.0) and (x < self.restriction))
        x_t = _ensure_torch_double(x)
        out_t = ((x_t > 0.0) & (x_t < self.restriction)).to(torch.float64)
        return _return_nparray_or_scalar(x, out_t)


class LeakyRectifier(Rectifier):
    """Leaky Rectifier activation function."""

    def __init__(self, negativeSlope=0.01, positiveSlope=1.0):
        self.negativeSlope = negativeSlope
        self.positiveSlope = positiveSlope

    def f(self, x):
        """
        out = x * (positiveSlope)      if x >= 0
            x * (negativeSlope)       if x < 0
        """
        if np.isscalar(x):
            return self.positiveSlope * x if x >= 0 else self.negativeSlope * x
        x_t = _ensure_torch_double(x)
        # We can do: slope = posSlope + (negSlope - posSlope)*(x<0)
        # or simpler: torch.where
        out_t = torch.where(
            x_t >= 0, x_t * self.positiveSlope, x_t * self.negativeSlope
        )
        return _return_nparray_or_scalar(x, out_t)

    def df(self, x):
        """Derivative is positiveSlope if x>=0 else negativeSlope."""
        if np.isscalar(x):
            return self.positiveSlope if x >= 0 else self.negativeSlope
        x_t = _ensure_torch_double(x)
        out_t = torch.where(
            x_t >= 0.0,
            torch.tensor(self.positiveSlope, dtype=torch.float64),
            torch.tensor(self.negativeSlope, dtype=torch.float64),
        )
        return _return_nparray_or_scalar(x, out_t)


# Soft-linear
class ExponentialLinear(object):
    """Exponential Linear activation function (ELU)."""

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def f(self, x):
        # x if x>0 else alpha*(exp(x)-1)
        if np.isscalar(x):
            if x > 0.0:
                return x
            else:
                return self.alpha * (np.exp(x) - 1.0)
        x_t = _ensure_torch_double(x)
        pos_mask = x_t > 0
        out_t = torch.where(pos_mask, x_t, self.alpha * (torch.exp(x_t) - 1.0))
        return _return_nparray_or_scalar(x, out_t)

    def df(self, x):
        # derivative => 1 if x>0 else alpha*exp(x)
        if np.isscalar(x):
            return 1.0 if x > 0.0 else self.alpha * np.exp(x)
        x_t = _ensure_torch_double(x)
        pos_mask = x_t > 0
        out_t = torch.where(pos_mask, torch.ones_like(x_t), self.alpha * torch.exp(x_t))
        return _return_nparray_or_scalar(x, out_t)


class SigmoidWeightedLinear(object):
    """Sigmoid weighted linear units (Swish)."""

    def __init__(self, beta=1.0):
        self.beta = beta

    def f(self, x):
        # x * sigmoid(beta*x)
        if np.isscalar(x):
            return x * Sigmoid.f(self.beta * x)
        x_np = np.array(x, copy=False)
        sig = Sigmoid.f(self.beta * x_np)
        return x_np * sig

    def df(self, x):
        # derivative: sig * (1 + x*(1 - sig)) with sig = sigmoid(beta*x)
        if np.isscalar(x):
            sig_val = Sigmoid.f(self.beta * x)
            return sig_val * (1.0 + x * (1.0 - sig_val))
        x_np = np.array(x, copy=False)
        sig = Sigmoid.f(self.beta * x_np)
        return sig * (1.0 + x_np * (1.0 - sig))


class SoftPlus(object):
    """SoftPlus: log(1 + exp(x))."""

    @classmethod
    def f(cls, x):
        if np.isscalar(x):
            return np.log(1.0 + np.exp(x))
        x_t = _ensure_torch_double(x)
        out_t = torch.log(1.0 + torch.exp(x_t))
        return _return_nparray_or_scalar(x, out_t)

    @classmethod
    def g(cls, y):
        # inverse: log( exp(y) - 1 )
        if np.isscalar(y):
            return np.log(np.exp(y) - 1.0)
        y_t = _ensure_torch_double(y)
        out_t = torch.log(torch.exp(y_t) - 1.0)
        return _return_nparray_or_scalar(y, out_t)

    @classmethod
    def df(cls, x):
        # derivative: 1 / (1 + exp(-x))
        if np.isscalar(x):
            return 1.0 / (1.0 + np.exp(-x))
        x_t = _ensure_torch_double(x)
        out_t = 1.0 / (1.0 + torch.exp(-x_t))
        return _return_nparray_or_scalar(x, out_t)

    @classmethod
    def ddf(cls, x):
        # second derivative: exp(x) / (1+exp(x))^2
        if np.isscalar(x):
            ex = np.exp(x)
            return ex / ((1.0 + ex) ** 2)
        x_t = _ensure_torch_double(x)
        ex_t = torch.exp(x_t)
        out_t = ex_t / torch.pow(1.0 + ex_t, 2)
        return _return_nparray_or_scalar(x, out_t)

    @classmethod
    def dg(cls, y):
        # derivative of inverse: 1 / (1 - exp(-y))
        if np.isscalar(y):
            return 1.0 / (1.0 - np.exp(-y))
        y_t = _ensure_torch_double(y)
        out_t = 1.0 / (1.0 - torch.exp(-y_t))
        return _return_nparray_or_scalar(y, out_t)


##############################################################################
# Bounded
##############################################################################


# Step
class Step(object):
    """Step activation function."""

    @classmethod
    def f(cls, x):
        if np.isscalar(x):
            return float(x > 0)
        x_t = _ensure_torch_double(x)
        out_t = (x_t > 0).to(torch.float64)
        return _return_nparray_or_scalar(x, out_t)

    @classmethod
    def df(cls, x):
        # derivative is 0
        return 0.0

    @classmethod
    def ddf(cls, x):
        # second derivative also 0
        return 0.0


# Soft-step
class Sigmoid(object):
    """Sigmoid function: 0.5 + 0.5*tanh(0.5*x)."""

    @classmethod
    def f(cls, x):
        # replicate 0.5 + 0.5*tanh(0.5*x)
        if np.isscalar(x):
            return 0.5 + 0.5 * np.tanh(0.5 * x)
        x_t = _ensure_torch_double(x)
        out_t = 0.5 + 0.5 * torch.tanh(0.5 * x_t)
        return _return_nparray_or_scalar(x, out_t)

    @classmethod
    def g(cls, y):
        # inverse = 2*arctanh(2*y-1)
        if np.isscalar(y):
            return 2.0 * np.arctanh(2.0 * y - 1.0)
        y_t = _ensure_torch_double(y)
        # torch doesn't have a native arctanh in older versions, but as of 1.7+ there is torch.atanh
        # We'll just do the numeric approach: 0.5*ln((1+x)/(1-x)) => 2.0* that:
        out_t = 2.0 * torch.atanh(2.0 * y_t - 1.0)
        return _return_nparray_or_scalar(y, out_t)

    @classmethod
    def df(cls, x):
        # derivative of the above: s*(1-s). We'll just do s = f(x).
        s = cls.f(x)
        return s * (1.0 - s)

    @classmethod
    def ddf(cls, x):
        # second derivative: s - 3s^2 + 2s^3
        s = cls.f(x)
        return s - 3.0 * (s**2) + 2.0 * (s**3)

    @classmethod
    def dg(cls, y):
        # derivative of inverse => 1 / (y - y^2)
        return 1.0 / (y - y**2)


class SoftSign(object):
    """SoftSign function: x/(1+|x|)."""

    @classmethod
    def f(cls, x):
        if np.isscalar(x):
            return x / (1.0 + abs(x))
        x_t = _ensure_torch_double(x)
        out_t = x_t / (1.0 + torch.abs(x_t))
        return _return_nparray_or_scalar(x, out_t)

    @classmethod
    def df(cls, x):
        # derivative => 1 / (1+|x|)^2
        if np.isscalar(x):
            return 1.0 / ((1.0 + abs(x)) ** 2)
        x_t = _ensure_torch_double(x)
        out_t = 1.0 / torch.pow(1.0 + torch.abs(x_t), 2)
        return _return_nparray_or_scalar(x, out_t)

    @classmethod
    def ddf(cls, x):
        # second derivative => -(2*x)/(|x|*(1+|x|)^3)
        if np.isscalar(x):
            ax = abs(x)
            return -(2.0 * x) / (ax * ((1.0 + ax) ** 3)) if ax > 1e-40 else 0.0
        x_t = _ensure_torch_double(x)
        ax_t = torch.abs(x_t)
        # avoid tiny-division
        mask = ax_t > 1e-40
        # expression
        num_t = -2.0 * x_t
        denom_t = ax_t * ((1.0 + ax_t) ** 3)
        out_t = torch.zeros_like(x_t)
        out_t[mask] = num_t[mask] / denom_t[mask]
        return _return_nparray_or_scalar(x, out_t)


class HyperbolicTangent(object):
    """HyperbolicTangent function: tanh(x)."""

    @classmethod
    def f(cls, x):
        if np.isscalar(x):
            return np.tanh(x)
        x_t = _ensure_torch_double(x)
        out_t = torch.tanh(x_t)
        return _return_nparray_or_scalar(x, out_t)

    @classmethod
    def g(cls, y):
        # inverse => 0.5*(log(1+y) - log(1-y)) = atanh(y)
        if np.isscalar(y):
            return 0.5 * (np.log(1.0 + y) - np.log(1.0 - y))
        y_t = _ensure_torch_double(y)
        out_t = 0.5 * (torch.log(1.0 + y_t) - torch.log(1.0 - y_t))
        return _return_nparray_or_scalar(y, out_t)

    @classmethod
    def df(cls, x):
        # derivative => 1 - tanh^2(x)
        t = cls.f(x)
        return 1.0 - t * t

    @classmethod
    def ddf(cls, x):
        # second derivative => -2*tanh(x)*(1 - tanh^2(x))
        t = cls.f(x)
        return -2.0 * t * (1.0 - t * t)

    @classmethod
    def dg(cls, y):
        # derivative of inverse => exp(-log(1-y^2)) = 1/(1-y^2)
        # but original code => np.exp(-np.log((1.0 - y**2)))
        # that's 1/(1-y^2)
        return 1.0 / (1.0 - y**2)


class SoftMax(object):
    """SoftMax function."""

    @classmethod
    def f(cls, x):
        """
        For a batch of vectors x shape = [batch, dim],
        softmax across the last dimension (dim=1).
        result[i] = exp(x[i]) / sum(exp(x[i]), axis=1).
        Uses log_sum_exp from numpyextension (Torch-based).
        """
        # replicate old:
        # return np.exp(x - log_sum_exp(x, axis=1).reshape(x.shape[0],1))
        # This implies x must be shape (N, M)
        lse = log_sum_exp(x, axis=1)
        # lse has shape (N,). We broadcast
        return np.exp(x - lse.reshape(x.shape[0], 1))

    @classmethod
    def df(cls, x):
        """
        The derivative of softmax is a batch of (dim x dim) Jacobians.
        original code:
          result = x[0]*I - x[0].T x[0]
          stacked for each row in x.
          shape => [batch, dim, dim]
        """
        # x shape => (batch, dim)
        # result shape => (batch, dim, dim)
        # The code does: x[i]*I - x[i]^T x[i]
        # We replicate that in Torch, then return as np
        if x.ndim != 2:
            raise ValueError("SoftMax.df expects x to be 2D [batch, dim].")
        batch, dim = x.shape
        # We'll do a for-loop exactly like the original
        first = x[0] * np.eye(dim) - np.outer(x[0], x[0])
        first = first.reshape((1, dim, dim))
        if batch == 1:
            return first
        out = [first]
        for i in range(1, batch):
            mat = x[i] * np.eye(dim) - np.outer(x[i], x[i])
            mat = mat.reshape((1, dim, dim))
            out.append(mat)
        return np.concatenate(out, axis=0)


##############################################################################
# Symmetric, periodic
##############################################################################


class RadialBasis(object):
    """Radial Basis function: exp(-(x-mean)^2 / variance)."""

    def __init__(self, mean=0.0, variance=1.0):
        self.mean = mean
        self.variance = variance

    def f(self, x):
        """
        out = exp(-((x-mean)^2)/variance).
        """
        if np.isscalar(x):
            act = x - self.mean
            return np.exp(-(act**2) / self.variance)
        x_t = _ensure_torch_double(x)
        mean_t = _ensure_torch_double(self.mean)
        act_t = x_t - mean_t
        out_t = torch.exp(-(act_t * act_t) / self.variance)
        return _return_nparray_or_scalar(x, out_t)

    def df(self, x):
        """
        derivative => f(x)*2*(mean - x)/variance
        """
        val = self.f(x)  # returns either scalar or np array
        if np.isscalar(x):
            return val * 2.0 * (self.mean - x) / self.variance
        x_np = np.array(x, copy=False)
        return val * 2.0 * (self.mean - x_np) / self.variance

    def ddf(self, x):
        """
        second derivative => 2/variance * exp(-((x-mean)^2)/variance)*(2((x-mean)^2)/variance - 1)
        """
        if np.isscalar(x):
            activation = ((x - self.mean) ** 2) / self.variance
            return 2.0 / self.variance * np.exp(-activation) * (2.0 * activation - 1.0)
        x_t = _ensure_torch_double(x)
        mean_t = torch.tensor(self.mean, dtype=torch.float64)
        activation_t = ((x_t - mean_t) ** 2) / self.variance
        out_t = (
            2.0 / self.variance * torch.exp(-activation_t) * (2.0 * activation_t - 1.0)
        )
        return _return_nparray_or_scalar(x, out_t)


class Sinus(object):
    """Sinus function: sin(x)."""

    @classmethod
    def f(cls, x):
        if np.isscalar(x):
            return np.sin(x)
        x_t = _ensure_torch_double(x)
        out_t = torch.sin(x_t)
        return _return_nparray_or_scalar(x, out_t)

    @classmethod
    def df(cls, x):
        # derivative => cos(x)
        if np.isscalar(x):
            return np.cos(x)
        x_t = _ensure_torch_double(x)
        out_t = torch.cos(x_t)
        return _return_nparray_or_scalar(x, out_t)

    @classmethod
    def ddf(cls, x):
        # second derivative => -sin(x)
        if np.isscalar(x):
            return -np.sin(x)
        x_t = _ensure_torch_double(x)
        out_t = -torch.sin(x_t)
        return _return_nparray_or_scalar(x, out_t)


##############################################################################
# K-Winner-Take-All
##############################################################################


class KWinnerTakeAll(object):
    """
    K Winner take all activation function.

    :WARNING: The derivative is computed during forward pass. So the call
              order must always be forward pass, then backward pass on
              the same data if you rely on that stored derivative.
    """

    def __init__(self, k, axis=1, activation_function=Identity()):
        self.k = k
        self.axis = axis
        self.activation_function = activation_function
        self._temp_derivative = None

    def f(self, x):
        """
        1) Apply the underlying activation_function f(x).
        2) Keep only the top k elements along 'axis', zero out the rest.
        3) Store derivative for backward pass.
        """
        # Evaluate underlying activation
        act = self.activation_function.f(np.atleast_2d(x))  # returns np array
        act_t = torch.from_numpy(np.array(act, copy=False)).double()

        # We replicate the old code, which sorts along self.axis, picks threshold:
        # if axis=0 => threshold = sort(act, axis=0)[-k,:]
        # else => axis=1 => threshold = sort(act, axis=1)[:,-k]
        # Then compare act >= threshold, produce winner mask
        if self.axis == 0:
            # shape of act => [rows, cols]; sort => [rows, cols] sorted by row
            # the row index -k => k-th from top
            sorted_t, _ = torch.sort(act_t, dim=0)
            # row -k => sorted_t[rows-k, :], note that negative indexing is reversed
            # but let's replicate the old code exactly
            # For small shapes, we can do that.
            thresh_t = sorted_t[sorted_t.shape[0] - self.k, :]
            # winner => act_t >= thresh_t (broadcast on rows)
            winner_t = act_t >= thresh_t  # broadcast across rows
        else:
            # axis=1 => we do .T for sorting, then the same indexing
            act_trans_t = act_t.transpose(0, 1)  # shape => [cols, rows]
            sorted_t, _ = torch.sort(act_trans_t, dim=0)  # sort each col
            thresh_t = sorted_t[sorted_t.shape[0] - self.k, :]
            # compare act_trans_t >= thresh_t => shape => [cols, rows]
            winner_trans_t = act_trans_t >= thresh_t
            winner_t = winner_trans_t.transpose(0, 1)

        winner_np = winner_t.to(torch.float64).cpu().numpy()
        # Now the final activation => act * winner
        out_np = act * winner_np

        # store derivative => winner * activation_function.df(x)
        # we must get the derivative from the underlying function
        df_underlying = self.activation_function.df(x)  # also shape => [rows, cols]
        # multiply by winner
        self._temp_derivative = df_underlying * winner_np
        return out_np

    def df(self, x):
        """
        Returns the derivative stored from the last forward pass.
        The param x is unused by default, but we keep the signature.
        """
        return self._temp_derivative
