"""Test module for numpy extensions.

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

import unittest
import sys
from pydeep.base.numpyextension import *

print("\n... pydeep.base.numpyextension.py")

epsilon = 0.00001


class TestNumpyExtension(unittest.TestCase):
    def test_log_sum_exp(self):
        sys.stdout.write("NumpyExtension -> Performing log_sum_exp test ...")
        sys.stdout.flush()
        x = np.array(
            [
                [1.0, 0.0, 9.0, 1.0, 6.0, 6.0],
                [0.0, 1.0, 3.0, 1.0, 1.0, 0.0],
                [0.0, 2.0, 1.0, 1.0, 0.0, 1.0],
                [1.0, 5.0, 1.0, 1.0, 1.0, 4.0],
            ]
        )
        res = np.array(
            [2.00640887, 5.07217242, 9.00314473, 2.38629436, 6.01582871, 6.13501328]
        )
        assert np.all(np.abs(log_sum_exp(x, 0) - res) < epsilon)
        res = np.array([9.0956451, 3.4091782, 2.8647064, 5.36543585])
        assert np.all(np.abs(log_sum_exp(x, 1) - res) < epsilon)
        print(" successfully passed!")
        sys.stdout.flush()

    def test_log_diff_exp(self):
        sys.stdout.write("NumpyExtension -> Performing log_diff_exp test ...")
        sys.stdout.flush()
        x = np.array([[1, 2, 3, 4, 5], [3, 4, 5, 6, 7]])
        res = np.array([2.85458654, 3.85458654, 4.85458654, 5.85458654, 6.85458654])
        assert np.all(np.abs(log_diff_exp(x, 0) - res) < epsilon)
        res = np.array(
            [
                [1.54132485, 3.54132485],
                [2.54132485, 4.54132485],
                [3.54132485, 5.54132485],
                [4.5413248, 6.54132485],
            ]
        )
        assert np.all(np.abs(log_diff_exp(x, 1) - res) < epsilon)
        print(" successfully passed!")
        sys.stdout.flush()

    def test_multinominal_batch_sampling(self):
        sys.stdout.write(
            "NumpyExtension -> Performing multinominal_batch_sampling test ..."
        )
        sys.stdout.flush()
        np.random.seed(42)
        p = np.tile(np.arange(5).reshape(5, 1), 100000).T
        res = np.mean(multinominal_batch_sampling(p, False), axis=0)
        res /= np.sum(res)
        target = np.arange(5) / np.float64(np.sum(np.arange(5)))
        assert np.all(np.abs(target - res) < 0.01)
        print(" successfully passed!")
        sys.stdout.flush()

    def test_get_norms(self):
        sys.stdout.write("NumpyExtension -> Performing get_norms test ...")
        sys.stdout.flush()
        x = np.arange(9).reshape(3, 3) + 1.0
        target0 = np.array([np.sqrt(66.0), np.sqrt(93.0), np.sqrt(126.0)])
        target1 = np.array([np.sqrt(14.0), np.sqrt(77.0), np.sqrt(194.0)])
        targetNone = np.sqrt(285.0)
        assert np.all(np.abs(target0 - get_norms(x, 0)) < epsilon)
        assert np.all(np.abs(target1 - get_norms(x, 1)) < epsilon)
        assert np.all(np.abs(targetNone - get_norms(x, None)) < epsilon)
        print(" successfully passed!")
        sys.stdout.flush()

    def test_restrict_norms(self):
        sys.stdout.write("NumpyExtension -> Performing restrict_norms test ...")
        sys.stdout.flush()
        x = np.arange(9).reshape(3, 3) + 1.0
        res = restrict_norms(x, 1.0, axis=0)
        assert np.all(restrict_norms(x, 1.0, axis=0) <= 1.0)
        assert np.all(restrict_norms(x, 1.0, axis=1) <= 1.0)
        assert np.all(restrict_norms(x, 1.0, axis=None) <= 1.0)
        x *= 0.0
        assert np.all(restrict_norms(x, 1.0, axis=0) <= 0.000001)
        assert np.all(restrict_norms(x, 1.0, axis=1) <= 0.000001)
        assert np.all(restrict_norms(x, 1.0, axis=None) <= 0.000001)
        print(" successfully passed!")
        sys.stdout.flush()

    def test_resize_norms(self):
        sys.stdout.write("NumpyExtension -> Performing resize_norms test ...")
        sys.stdout.flush()
        x = np.arange(9).reshape(3, 3) + 1.0
        res = resize_norms(x, 1.0, axis=0)
        assert np.all(np.abs(get_norms(res, 0) - 1.0) < epsilon)
        res = resize_norms(x, 1.0, axis=1)
        assert np.all(np.abs(get_norms(res, 1) - 1.0) < epsilon)
        res = resize_norms(x, 1.0, axis=None)
        assert np.all(np.abs(get_norms(res, None) - 1.0) < epsilon)
        print(" successfully passed!")
        sys.stdout.flush()

    def test_angle_between_vectors(self):
        sys.stdout.write("NumpyExtension -> Performing angle_between_vectors test ...")
        sys.stdout.flush()
        v1 = np.array([[1.0, 0.0]])
        v2 = np.array([[1.0, 1.0]])
        target1 = 45.0
        target2 = np.pi / 4.0
        assert (np.abs(angle_between_vectors(v1, v2, True) - target1)) < epsilon
        assert (np.abs(angle_between_vectors(v1, v2, False) - target2)) < epsilon

        v1 = np.array([[1.0, 0.0]])
        v2 = np.array([[0.0, 1.0]])
        target1 = 90.0
        target2 = np.pi / 2.0
        assert (np.abs(angle_between_vectors(v1, v2, True) - target1)) < epsilon
        assert (np.abs(angle_between_vectors(v1, v2, False) - target2)) < epsilon

        v1 = np.array([[-1.0, -1.0]])
        v2 = np.array([[1.0, 1.0]])
        target1 = 180.0
        target2 = np.pi
        assert (np.abs(angle_between_vectors(v1, v2, True) - target1)) < epsilon
        assert (np.abs(angle_between_vectors(v1, v2, False) - target2)) < epsilon
        print(" successfully passed!")
        sys.stdout.flush()

    def test_get_2D_gauss_kernel(self):
        sys.stdout.write("NumpyExtension -> Performing get_2D_gauss_kernel test ...")
        sys.stdout.flush()
        x = get_2d_gauss_kernel(
            3, 3, np.array([0.0, 0.0]), np.array([[1.0, 0.0], [0.0, 1.0]])
        )
        target = np.array(
            [
                [0.05854983, 0.09653235, 0.05854983],
                [0.09653235, 0.15915494, 0.09653235],
                [0.05854983, 0.09653235, 0.05854983],
            ]
        )
        assert np.all(np.abs(x - target) < 0.00001)
        x = get_2d_gauss_kernel(3, 3, np.array([0.0, 0.0]), np.array([1.0, 1.0]))
        target = np.array(
            [
                [0.05854983, 0.09653235, 0.05854983],
                [0.09653235, 0.15915494, 0.09653235],
                [0.05854983, 0.09653235, 0.05854983],
            ]
        )
        assert np.all(np.abs(x - target) < 0.00001)
        x = get_2d_gauss_kernel(3, 3, 0, 1)
        target = np.array(
            [
                [0.05854983, 0.09653235, 0.05854983],
                [0.09653235, 0.15915494, 0.09653235],
                [0.05854983, 0.09653235, 0.05854983],
            ]
        )
        assert np.all(np.abs(x - target) < 0.00001)
        x = get_2d_gauss_kernel(
            3, 3, np.array([1.0, 1.0]), np.array([[1.0, 0.0], [0.0, 1.0]])
        )
        target = np.array(
            [
                [0.00291502, 0.01306423, 0.02153928],
                [0.01306423, 0.05854983, 0.09653235],
                [0.02153928, 0.09653235, 0.15915494],
            ]
        )
        assert np.all(np.abs(x - target) < 0.00001)
        x = get_2d_gauss_kernel(
            3, 3, np.array([0.0, 0.0]), np.array([[1.0, 2.0], [0.0, 1.0]])
        )
        target = np.array(
            [
                [0.15915494, 0.09653235, 0.02153928],
                [0.09653235, 0.15915494, 0.09653235],
                [0.02153928, 0.09653235, 0.15915494],
            ]
        )
        assert np.all(np.abs(x - target) < 0.00001)
        print(" successfully passed!")
        sys.stdout.flush()

    def test_generate_binary_code(self):
        sys.stdout.write("NumpyExtension -> Performing generate_binary_code test ...")
        sys.stdout.flush()
        x = generate_binary_code(3, batch_size_exp=None, batch_number=0)
        res = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
            ]
        )
        assert np.all(np.abs(res == x))
        x = generate_binary_code(3, batch_size_exp=2, batch_number=0)
        res = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]
        )
        assert np.all(np.abs(res == x))
        x = generate_binary_code(3, batch_size_exp=2, batch_number=1)
        res = np.array(
            [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
        )
        assert np.all(np.abs(res == x))
        print(" successfully passed!")
        sys.stdout.flush()

    def test_get_binary_label(self):
        sys.stdout.write("NumpyExtension -> Performing get_binary_label test ...")
        sys.stdout.flush()
        labels = np.arange(5)
        target = np.eye(5, 5)
        res_test = get_binary_label(labels)
        assert np.all(np.abs(res_test == target))
        print(" successfully passed!")
        sys.stdout.flush()

    def test_compare_index_of_max(self):
        sys.stdout.write("NumpyExtension -> Performing compare_index_of_max test ...")
        sys.stdout.flush()
        a = np.array(
            [[3, 5, 9, 1, 2], [11, 3, 8, 2, 1], [4, 7, 3, 2, 10], [2, 3, 4, 5, 6]]
        )
        b = np.array(
            [
                [24, 43, 55, 32, 22],
                [0, 5, 6, 7, 5],
                [9, 7, 18, 22, 21],
                [32, 21, 44, 50, 60],
            ]
        )
        target = [0, 1, 1, 0]
        res_test = compare_index_of_max(a, b)
        assert np.all(np.abs(res_test == target))
        print(" successfully passed!")
        sys.stdout.flush()

    def test_shuffle_dataset(self):
        sys.stdout.write("NumpyExtension -> Performing shuffle_dataset test ...")
        sys.stdout.flush()
        np.random.seed(42)
        data = np.array(
            [[3, 5, 9, 1, 2], [11, 3, 8, 2, 1], [4, 7, 3, 2, 10], [2, 3, 4, 5, 6]]
        )
        label = np.array([[1], [2], [3], [4]])
        data_target = np.array(
            [[11, 3, 8, 2, 1], [2, 3, 4, 5, 6], [3, 5, 9, 1, 2], [4, 7, 3, 2, 10]]
        )
        label_target = np.array([[2], [4], [1], [3]])
        res = shuffle_dataset(data, label)
        assert np.all(res[0] == data_target)
        assert np.all(res[1] == label_target)
        print(" successfully passed!")
        sys.stdout.flush()

    def test_rotation_sequence(self):
        sys.stdout.write("NumpyExtension -> Performing rotationSequence test ...")
        sys.stdout.flush()
        x = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        target = np.array(
            [
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        )
        res_test = np.int32(rotation_sequence(x, 4, 4, 4))
        assert np.all(np.abs(res_test == target))
        print(" successfully passed!")
        sys.stdout.flush()

    def test_generate_2d_connection_matrix(self):
        sys.stdout.write(
            "NumpyExtension -> Performing generate_2D_connection_matrix test ..."
        )
        sys.stdout.flush()
        x = generate_2d_connection_matrix(
            input_x_dim=3,
            input_y_dim=3,
            field_x_dim=2,
            field_y_dim=2,
            overlap_x_dim=1,
            overlap_y_dim=1,
            wrap_around=False,
        )
        target = np.array(
            [
                [1, 1, 0, 1, 1, 0, 0, 0, 0],
                [0, 1, 1, 0, 1, 1, 0, 0, 0],
                [0, 0, 0, 1, 1, 0, 1, 1, 0],
                [0, 0, 0, 0, 1, 1, 0, 1, 1],
            ]
        ).T
        assert np.all(np.abs(x == target))
        x = generate_2d_connection_matrix(
            input_x_dim=3,
            input_y_dim=3,
            field_x_dim=2,
            field_y_dim=2,
            overlap_x_dim=1,
            overlap_y_dim=1,
            wrap_around=True,
        )
        target = np.array(
            [
                [1, 1, 0, 1, 1, 0, 0, 0, 0],
                [0, 1, 1, 0, 1, 1, 0, 0, 0],
                [1, 0, 1, 1, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 1, 0, 1, 1, 0],
                [0, 0, 0, 0, 1, 1, 0, 1, 1],
                [0, 0, 0, 1, 0, 1, 1, 0, 1],
                [1, 1, 0, 0, 0, 0, 1, 1, 0],
                [0, 1, 1, 0, 0, 0, 0, 1, 1],
                [1, 0, 1, 0, 0, 0, 1, 0, 1],
            ]
        ).T
        assert np.all(np.abs(x == target))
        print(" successfully passed!")
        sys.stdout.flush()


if __name__ is "__main__":
    unittest.main()
