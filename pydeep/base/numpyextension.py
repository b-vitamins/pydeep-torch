"""
This module provides different math functions that extend the numpy library,
now re-implemented internally using PyTorch while preserving the same external
interface and invariants.

:Implemented:
   - log_sum_exp
   - log_diff_exp
   - get_norms
   - multinominal_batch_sampling
   - restrict_norms
   - resize_norms
   - angle_between_vectors
   - get_2D_gauss_kernel
   - generate_binary_code
   - get_binary_label
   - compare_index_of_max
   - shuffle_dataset
   - rotation_sequence
   - generate_2D_connection_matrix

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

# We keep SciPy's rotate to preserve invariance in the rotation tests
from scipy.ndimage.interpolation import rotate

################################################################################
# For stable log-sum-exp & log-diff-exp, the old code used EXACTLY this shift:
#   alpha = x.max(axis) - numx.log(numx.finfo(numx.float64).max)/2.0
# The test suite can be sensitive to even tiny numeric differences, so we
# define this constant from the exact NumPy expression:
################################################################################
SHIFT_DOUBLE = float(numx.log(numx.finfo(numx.float64).max) / 2.0)


def log_sum_exp(x, axis=0):
    """Replicates the old stable log-sum-exp logic."""
    x_t = torch.as_tensor(x, dtype=torch.float64)
    if x_t.ndim == 0:
        return float(x_t.item())

    alpha_t = x_t.max(dim=axis)[0] - SHIFT_DOUBLE

    if axis == 1:
        tmp = x_t.transpose(0, 1) - alpha_t
        s = torch.sum(torch.exp(tmp), dim=0)
        out_t = alpha_t + torch.log(s)
    else:
        tmp = x_t - alpha_t
        s = torch.sum(torch.exp(tmp), dim=0)
        out_t = alpha_t + torch.log(s)

    return out_t.squeeze().cpu().numpy()


def log_diff_exp(x, axis=0):
    """Replicates the old stable log-diff-exp logic."""
    x_t = torch.as_tensor(x, dtype=torch.float64)
    if x_t.ndim == 0:
        return float(x_t.item())

    alpha_t = x_t.max(dim=axis)[0] - SHIFT_DOUBLE

    if axis == 1:
        tmp = x_t.transpose(0, 1) - alpha_t
        diff_vals = torch.diff(torch.exp(tmp), dim=0)
        out_t = alpha_t + torch.log(diff_vals)
    else:
        tmp = x_t - alpha_t
        diff_vals = torch.diff(torch.exp(tmp), dim=0)
        out_t = alpha_t + torch.log(diff_vals)

    return out_t.squeeze().cpu().numpy()


def multinominal_batch_sampling(probabilties, isnormalized=True):
    """Sample states where only one entry is 1 and the rest are 0, from given row-wise probabilities."""
    probs = numx.float64(probabilties)
    if not isnormalized:
        row_sums = probs.sum(axis=1).reshape(-1, 1)
        probs = probs / row_sums

    probs_t = torch.from_numpy(probs)
    cdf_t = torch.cumsum(probs_t, dim=1)
    cdf_minus_t = cdf_t - probs_t

    sample = numx.random.random((probs.shape[0], 1))
    sample_t = torch.from_numpy(sample)

    result_t = (cdf_t > sample_t) * (sample_t >= cdf_minus_t)
    return result_t.numpy()


def get_norms(matrix, axis=0):
    mat_t = torch.as_tensor(matrix, dtype=torch.float64)
    if axis is None:
        val = torch.sqrt(torch.sum(mat_t * mat_t))
        return float(val.item())
    else:
        val = torch.sqrt(torch.sum(mat_t * mat_t, dim=axis))
        return val.cpu().numpy()


def restrict_norms(matrix, max_norm, axis=0):
    res_t = torch.tensor(matrix, dtype=torch.float64)
    if axis is None:
        norm_val = torch.sqrt(torch.sum(res_t * res_t))
        if norm_val > max_norm:
            res_t *= max_norm / norm_val
    else:
        threshold = max_norm / numx.sqrt(res_t.shape[abs(1 - axis)])
        if res_t.max().item() > threshold:
            norms_t = torch.sqrt(torch.sum(res_t * res_t, dim=axis))
            for r in range(norms_t.shape[0]):
                if norms_t[r] > max_norm:
                    if axis == 0:
                        res_t[:, r] *= max_norm / norms_t[r]
                    else:
                        res_t[r, :] *= max_norm / norms_t[r]
    return res_t.cpu().numpy()


def resize_norms(matrix, norm, axis=0):
    res_t = torch.tensor(matrix, dtype=torch.float64)
    if axis is None:
        total_norm = torch.sqrt(torch.sum(res_t * res_t))
        if total_norm > 1e-40:
            res_t *= norm / total_norm
    else:
        norms_t = torch.sqrt(torch.sum(res_t * res_t, dim=axis))
        for r in range(norms_t.shape[0]):
            if norms_t[r] > 1e-40:
                if axis == 0:
                    res_t[:, r] *= norm / norms_t[r]
                else:
                    res_t[r, :] *= norm / norms_t[r]
    return res_t.cpu().numpy()


def angle_between_vectors(v1, v2, degree=True):
    """Computes angle(s) between rows of v1 and v2, replicating the old code's multi-row logic."""
    v1_np = numx.atleast_2d(v1)
    v2_np = numx.atleast_2d(v2)

    v1_t = torch.from_numpy(v1_np).double()  # NxD
    v2_t = torch.from_numpy(v2_np).double()  # MxD

    dot_t = v1_t @ v2_t.transpose(0, 1)  # NxM
    norm1_t = torch.sqrt(torch.sum(v1_t * v1_t, dim=1, keepdim=True)) + 1e-40
    norm2_t = torch.sqrt(torch.sum(v2_t * v2_t, dim=1, keepdim=True)) + 1e-40

    denom_t = norm1_t * norm2_t.transpose(0, 1)  # NxM
    cos_t = dot_t / denom_t
    cos_t = torch.clamp(cos_t, -1.0, 1.0)
    angle_t = torch.acos(cos_t)

    if degree:
        angle_t = angle_t * (180.0 / 3.141592653589793)

    return angle_t.cpu().numpy()


def get_2d_gauss_kernel(width, height, shift=0, var=[1.0, 1.0]):
    """Creates a 2D Gauss kernel of size (width x height) with the specified variance and shift."""

    def gauss_pt(xy, mean, covariance):
        det_c = torch.linalg.det(covariance)
        inv_c = torch.linalg.inv(covariance)
        fac = 1.0 / (2.0 * 3.141592653589793 * torch.sqrt(det_c))
        d = xy - mean
        exponent = -0.5 * torch.matmul(
            torch.matmul(d.unsqueeze(0), inv_c), d.unsqueeze(1)
        )
        return fac * torch.exp(exponent.squeeze(0))

    if width % 2 == 0:
        print("N needs to be odd!")
    if height % 2 == 0:
        print("M needs to be odd!")

    if numx.isscalar(shift):
        m = torch.tensor([shift, shift], dtype=torch.float64)
    else:
        m = torch.tensor(shift, dtype=torch.float64)

    if numx.isscalar(var):
        covar = torch.tensor([[var, 0], [0, var]], dtype=torch.float64)
    else:
        var_t = torch.as_tensor(var, dtype=torch.float64)
        if var_t.ndim == 1:
            covar = torch.tensor([[var_t[0], 0], [0, var_t[1]]], dtype=torch.float64)
        else:
            covar = var_t

    lowern = (width - 1) // 2
    lowerm = (height - 1) // 2

    mat_t = torch.zeros((width, height), dtype=torch.float64)
    for x in range(width):
        for y in range(height):
            xy_t = torch.tensor([x - lowern, y - lowerm], dtype=torch.float64)
            mat_t[x, y] = gauss_pt(xy_t, m, covar)

    return mat_t.numpy()


def generate_binary_code(bit_length, batch_size_exp=None, batch_number=0):
    """Generate all possible binary vectors of length 'bit_length', or a particular batch of them."""
    if batch_size_exp is None:
        batch_size_exp = bit_length
    batch_size = 2**batch_size_exp

    bit_combinations_t = torch.zeros((batch_size, bit_length), dtype=torch.float64)
    for number in range(batch_size):
        dividend = number + batch_number * batch_size
        bit_index = 0
        while dividend != 0:
            bit_combinations_t[number, bit_index] = float(dividend % 2)
            dividend //= 2
            bit_index += 1

    return bit_combinations_t.numpy()


def get_binary_label(int_array):
    """Converts a 1D array of integer labels into a 2D array of binary indicators (one-hot)."""
    int_tensor = torch.as_tensor(int_array, dtype=torch.long)
    max_label = torch.max(int_tensor).item() + 1
    out = torch.zeros((int_tensor.shape[0], max_label), dtype=torch.float64)
    for i in range(int_tensor.shape[0]):
        out[i, int_tensor[i]] = 1.0
    return out.numpy()


def compare_index_of_max(output, target):
    """Compares data rows by the index of the maximal value, returning 0 if they match, 1 otherwise."""
    out_t = torch.from_numpy(numx.array(output, dtype=numx.float64))
    tgt_t = torch.from_numpy(numx.array(target, dtype=numx.float64))
    out_idx = torch.argmax(out_t, dim=1)
    tgt_idx = torch.argmax(tgt_t, dim=1)
    diff = (out_idx != tgt_idx).int()
    return diff.numpy()


def shuffle_dataset(data, label):
    """Shuffles data points and labels correspondingly using the same random permutation."""
    idx = numx.arange(data.shape[0])
    idx = numx.random.permutation(idx)
    return data[idx], label[idx]


def rotation_sequence(image, width, height, steps):
    """
    Rotates a 2D image (provided as a 1D vector) in 'steps' increments. Each step rotates by (360/steps) degrees.

    :param image: Image as 1D vector of length (width*height).
    :param width: Image width.
    :param height: Image height.
    :param steps: Number of rotation steps.
    :return: [steps, width*height] array of rotated images.
    """
    results = numx.zeros((steps, image.shape[0]))
    results[0] = image
    for i in range(1, steps):
        angle = i * 360.0 / steps
        sample = rotate(image.reshape(width, height), angle)
        sample = sample[
            (sample.shape[0] - width) // 2 : (sample.shape[0] + width) // 2,
            (sample.shape[1] - height) // 2 : (sample.shape[1] + height) // 2,
        ]
        results[i] = sample.reshape(1, image.shape[0])
    return results


def generate_2d_connection_matrix(
    input_x_dim,
    input_y_dim,
    field_x_dim,
    field_y_dim,
    overlap_x_dim,
    overlap_y_dim,
    wrap_around=True,
):
    """Constructs a binary connection matrix for local receptive fields."""
    if field_x_dim > input_x_dim:
        raise NotImplementedError("field_x_dim > input_x_dim is invalid!")
    if field_y_dim > input_y_dim:
        raise NotImplementedError("field_y_dim > input_y_dim is invalid!")
    if overlap_x_dim >= field_x_dim:
        raise NotImplementedError("overlap_x_dim >= field_x_dim is invalid!")
    if overlap_y_dim >= field_y_dim:
        raise NotImplementedError("overlap_y_dim >= field_y_dim is invalid!")

    matrix = None
    start_x = 0
    start_y = 0
    end_x = input_x_dim
    end_y = input_y_dim
    if not wrap_around:
        end_x = input_x_dim - (field_x_dim - 1)
        end_y = input_y_dim - (field_y_dim - 1)

    step_x = field_x_dim - overlap_x_dim
    step_y = field_y_dim - overlap_y_dim

    for x in range(start_x, end_x, step_x):
        for y in range(start_y, end_y, step_y):
            column_t = torch.zeros((input_x_dim, input_y_dim), dtype=torch.float64)
            for i in range(x, x + field_x_dim):
                for j in range(y, y + field_y_dim):
                    column_t[i % input_x_dim, j % input_y_dim] = 1.0
            column_t = column_t.reshape(input_x_dim * input_y_dim)
            if matrix is None:
                matrix = column_t
            else:
                matrix = torch.vstack((matrix, column_t))

    return matrix.T.numpy()
