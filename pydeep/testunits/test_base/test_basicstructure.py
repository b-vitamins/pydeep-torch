"""Test module for basic structures.

:Version:
    1.1.0

:Date:
    13.03.2017

:Author:
    Jan Melchior, Nan Wang

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

import copy
import sys
import unittest
from collections.abc import Iterable

import numpy as np

from pydeep.base.activationfunction import Sigmoid
from pydeep.base.basicstructure import BipartiteGraph, StackOfBipartiteGraphs

print("\n... pydeep.base.basicstructure.py")


def compare_BipartiteGraph_after_removing(new_model, old_model, indices, unit_type):
    """Compare the two bipartiteGraph instances after removing.
    All the parameters of the two models are arranged as
    (changed_dimension, unchanged_dimension) for the easiness of comparison.
    """
    if unit_type == "vis":
        new_bias = new_model.bv.T
        new_offset = new_model.ov.T
        new_weights = new_model.w
        new_changed_dim = new_model.input_dim
        new_unchanged_dim = new_model.output_dim
        old_bias = old_model.bv.T
        old_offset = old_model.ov.T
        old_weights = old_model.w
        old_changed_dim = old_model.input_dim
        old_unchanged_dim = old_model.output_dim
    elif unit_type == "hid":
        new_bias = new_model.bh.T
        new_offset = new_model.oh.T
        new_weights = new_model.w.T
        new_changed_dim = new_model.output_dim
        new_unchanged_dim = new_model.input_dim
        old_bias = old_model.bh.T
        old_offset = old_model.oh.T
        old_weights = old_model.w.T
        old_changed_dim = old_model.output_dim
        old_unchanged_dim = old_model.input_dim
    else:
        raise NotImplementedError
    if np.isscalar(indices):
        idx_list = (indices,)
    else:
        idx_list = np.array(indices)
        idx_list.sort()
    # check the dimension
    assert new_changed_dim == old_changed_dim - len(idx_list)
    assert new_unchanged_dim == old_unchanged_dim
    # check the rest part
    for pos_idx, each_pos in enumerate(idx_list):
        if pos_idx == 0:
            old_start_pos = 0
            old_end_pos = each_pos
            new_start_pos = 0
            new_end_pos = each_pos
        elif pos_idx == len(idx_list) - 1:
            old_start_pos = each_pos + 1
            old_end_pos = None
            new_start_pos = new_end_pos
            new_end_pos = None
        else:
            old_start_pos = idx_list[pos_idx - 1] + 1
            old_end_pos = each_pos
            new_start_pos = new_end_pos
            new_end_pos += old_end_pos - old_start_pos
        assert np.all(
            new_weights[new_start_pos:new_end_pos]
            == old_weights[old_start_pos:old_end_pos]
        )
        assert np.all(
            new_bias[new_start_pos:new_end_pos] == old_bias[old_start_pos:old_end_pos]
        )
        assert np.all(
            new_offset[new_start_pos:new_end_pos]
            == old_offset[old_start_pos:old_end_pos]
        )


def compare_BipartiteGraph_after_adding(
    new_model,
    old_model,
    position,
    num_new_units,
    unit_type,
    check_newpart=False,
    check_oldpart=True,
    check_dim=True,
):
    """Compare the two bipartiteGraph instances
    All the parameters of the two models are arranged as
    (changed_dimension, unchanged_dimension) for the easiness of comparison.
    """
    pos = position
    if unit_type == "vis":
        new_bias = new_model.bv.T
        new_offset = new_model.ov.T
        new_weights = new_model.w
        new_changed_dim = new_model.input_dim
        new_unchanged_dim = new_model.output_dim
        old_bias = old_model.bv.T
        old_offset = old_model.ov.T
        old_weights = old_model.w
        old_changed_dim = old_model.input_dim
        old_unchanged_dim = old_model.output_dim
    elif unit_type == "hid":
        new_bias = new_model.bh.T
        new_offset = new_model.oh.T
        new_weights = new_model.w.T
        new_changed_dim = new_model.output_dim
        new_unchanged_dim = new_model.input_dim
        old_bias = old_model.bh.T
        old_offset = old_model.oh.T
        old_weights = old_model.w.T
        old_changed_dim = old_model.output_dim
        old_unchanged_dim = old_model.input_dim
    else:
        raise NotImplementedError
    if check_newpart:
        # check the middle part
        assert np.all(new_weights[pos : pos + num_new_units] == old_weights)
        assert np.all(new_bias[pos : pos + num_new_units] == old_bias)
        assert np.all(new_offset[pos : pos + num_new_units] == old_offset)
    if check_dim:
        # check the dimension
        assert new_changed_dim == 2 * old_changed_dim
        assert new_unchanged_dim == old_unchanged_dim
    if check_oldpart:
        # check the left part
        assert np.all(new_weights[:pos] == old_weights[:pos])
        assert np.all(new_bias[:pos] == old_bias[:pos])
        assert np.all(new_offset[:pos] == old_offset[:pos])
        # check the right part
        assert np.all(new_weights[pos + num_new_units :] == old_weights[pos:])
        assert np.all(new_bias[pos + num_new_units :] == old_bias[pos:])
        assert np.all(new_offset[pos + num_new_units :] == old_offset[pos:])


class Test_BipartiteGraph(unittest.TestCase):
    def test___init__(self):
        sys.stdout.write(
            "BipartiteGraph -> Performing BipartiteGraph initialzation test ..."
        )
        sys.stdout.flush()

        # Check init scalar
        number_visibles = 3
        number_hiddens = 2

        np.random.seed(42)
        model = BipartiteGraph(
            number_visibles=number_visibles,
            number_hiddens=number_hiddens,
            data=None,
            initial_weights=np.random.randn(),
            initial_visible_bias=np.random.randn(),
            initial_hidden_bias=np.random.randn(),
            initial_visible_offsets=np.random.randn(),
            initial_hidden_offsets=np.random.randn(),
        )

        np.random.seed(42)
        initial_weights = np.random.randn()
        initial_visible_bias = np.random.randn() * np.ones((1, number_visibles))
        initial_hidden_bias = np.random.randn() * np.ones((1, number_hiddens))
        initial_visible_offsets = np.random.randn() * np.ones((1, number_visibles))
        initial_hidden_offsets = np.random.randn() * np.ones((1, number_hiddens))
        initial_weights = (
            np.random.randn(number_visibles, number_hiddens) * initial_weights
        )

        assert np.all(model.input_dim == number_visibles)
        assert np.all(model.output_dim == number_hiddens)
        assert np.all(model.w == initial_weights)
        assert np.all(model.bv == initial_visible_bias)
        assert np.all(model.bh == initial_hidden_bias)
        assert np.all(model.ov == initial_visible_offsets)
        assert np.all(model.oh == initial_hidden_offsets)

        # Check init arrays
        np.random.seed(42)
        initial_weights = np.random.randn(number_visibles, number_hiddens)
        initial_visible_bias = np.random.randn(1, number_visibles)
        initial_hidden_bias = np.random.randn(1, number_hiddens)
        initial_visible_offsets = np.random.randn(1, number_visibles)
        initial_hidden_offsets = np.random.randn(1, number_hiddens)

        np.random.seed(42)
        model = BipartiteGraph(
            number_visibles=number_visibles,
            number_hiddens=number_hiddens,
            data=None,
            initial_weights=initial_weights,
            initial_visible_bias=initial_visible_bias,
            initial_hidden_bias=initial_hidden_bias,
            initial_visible_offsets=initial_visible_offsets,
            initial_hidden_offsets=initial_hidden_offsets,
        )

        np.random.seed(42)
        assert np.all(model.w == initial_weights)
        assert np.all(model.bv == initial_visible_bias)
        assert np.all(model.bh == initial_hidden_bias)
        assert np.all(model.ov == initial_visible_offsets)
        assert np.all(model.oh == initial_hidden_offsets)

        # Check AUTO init without data
        np.random.seed(42)
        initial_weights = (
            2.0 * np.random.rand(number_visibles, number_hiddens) - 1.0
        ) * (4.0 * np.sqrt(6.0 / (number_visibles + number_hiddens)))
        initial_visible_bias = "AUTO"
        initial_hidden_bias = "AUTO"
        initial_visible_offsets = "AUTO"
        initial_hidden_offsets = "AUTO"

        np.random.seed(42)
        model = BipartiteGraph(
            number_visibles=number_visibles,
            number_hiddens=number_hiddens,
            data=None,
            initial_weights="AUTO",
            initial_visible_bias="AUTO",
            initial_hidden_bias="AUTO",
            initial_visible_offsets="AUTO",
            initial_hidden_offsets="AUTO",
        )

        assert np.all(model.w == initial_weights)
        assert np.all(model.bv == 0.0)
        assert np.all(model.bh == 0.0)
        assert np.all(model.ov == 0.5)
        assert np.all(model.oh == 0.5)

        # Check AUTO init with data
        test_data = np.random.randn(100, number_visibles)
        test_data_mean = test_data.mean(axis=0).reshape(1, test_data.shape[1])

        np.random.seed(42)
        # All weight combination checked already
        initial_visible_bias = (
            Sigmoid().g(np.clip(test_data_mean, 0.001, 0.9999)).reshape(model.ov.shape)
        )
        initial_hidden_bias = 0.0
        initial_visible_offsets = test_data_mean
        initial_hidden_offsets = 0.5

        np.random.seed(42)
        model = BipartiteGraph(
            number_visibles=number_visibles,
            number_hiddens=number_hiddens,
            data=test_data,
            initial_weights="AUTO",
            initial_visible_bias="AUTO",
            initial_hidden_bias="AUTO",
            initial_visible_offsets="AUTO",
            initial_hidden_offsets="AUTO",
        )

        assert np.all(model.bv == initial_visible_bias)
        assert np.all(model.bh == initial_hidden_bias)
        assert np.all(model.ov == initial_visible_offsets)
        assert np.all(model.oh == initial_hidden_offsets)

        # Check AUTO init with INVERSE SIGMOID
        test_data = np.random.randn(100, number_visibles)
        test_data_mean = test_data.mean(axis=0).reshape(1, test_data.shape[1])

        np.random.seed(42)
        # All weight combination checked already
        initial_visible_offsets = np.random.randn() * np.ones((1, number_visibles))
        initial_hidden_offsets = np.random.randn() * np.ones((1, number_hiddens))
        initial_visible_bias = np.array(
            Sigmoid().g(np.clip(initial_visible_offsets, 0.001, 0.9999))
        ).reshape(1, number_visibles)
        initial_hidden_bias = np.array(
            Sigmoid().g(np.clip(initial_hidden_offsets, 0.001, 0.9999))
        ).reshape(1, number_hiddens)

        np.random.seed(42)
        model = BipartiteGraph(
            number_visibles=number_visibles,
            number_hiddens=number_hiddens,
            data=test_data,
            initial_weights="AUTO",
            initial_visible_bias="INVERSE_SIGMOID",
            initial_hidden_bias="INVERSE_SIGMOID",
            initial_visible_offsets=np.random.randn() * np.ones((1, number_visibles)),
            initial_hidden_offsets=np.random.randn() * np.ones((1, number_hiddens)),
        )

        assert np.all(model.bv == initial_visible_bias)
        assert np.all(model.bh == initial_hidden_bias)
        assert np.all(model.ov == initial_visible_offsets)
        assert np.all(model.oh == initial_hidden_offsets)
        print(" successfully passed!")
        sys.stdout.flush()

    def test__add_hidden_units(self):
        sys.stdout.write("BipartiteGraph -> Performing add_hidden_units test ...")
        sys.stdout.flush()
        num_hid = 4
        pos = num_hid // 2
        init_value = 3.0
        check_list = ["AUTO", init_value, "non-scalar"]
        for each_case in check_list:
            bpgraph = BipartiteGraph(number_visibles=2, number_hiddens=num_hid)
            bpgraph_old = copy.deepcopy(bpgraph)
            if each_case == "non-scalar":
                bpgraph._add_hidden_units(
                    num_new_hiddens=num_hid,
                    position=pos,
                    initial_weights=bpgraph_old.w,
                    initial_bias=bpgraph_old.bh,
                    initial_offsets=bpgraph_old.oh,
                )
                check_newpart = True
            else:
                bpgraph._add_hidden_units(
                    num_new_hiddens=num_hid,
                    position=pos,
                    initial_weights=each_case,
                    initial_bias=each_case,
                    initial_offsets=each_case,
                )
                check_newpart = False
            compare_BipartiteGraph_after_adding(
                bpgraph, bpgraph_old, pos, num_hid, "hid", check_newpart
            )
        print(" successfully passed!")
        sys.stdout.flush()

    def test__remove_hidden_units(self):
        sys.stdout.write("BipartiteGraph -> Performing remove_hidden_units test ...")
        sys.stdout.flush()
        bpgraph = BipartiteGraph(number_visibles=2, number_hiddens=4)
        bpgraph_old = copy.deepcopy(bpgraph)
        indices = (0, 1, 3)
        bpgraph._remove_hidden_units(indices)
        compare_BipartiteGraph_after_removing(
            bpgraph, bpgraph_old, indices, unit_type="hid"
        )
        print(" successfully passed!")
        sys.stdout.flush()

    def test__add_visible_units(self):
        sys.stdout.write("BipartiteGraph -> Performing add_visible_units test ...")
        sys.stdout.flush()
        num_vis = 4
        pos = num_vis // 2
        init_value = 3.0
        check_list = ["AUTO", init_value, "non-scalar"]
        for each_case in check_list:
            bpgraph = BipartiteGraph(number_visibles=num_vis, number_hiddens=3)
            bpgraph_old = copy.deepcopy(bpgraph)
            if each_case == "non-scalar":
                bpgraph._add_visible_units(
                    num_new_visibles=num_vis,
                    position=pos,
                    initial_weights=bpgraph_old.w,
                    initial_bias=bpgraph_old.bv,
                    initial_offsets=bpgraph_old.ov,
                )
                check_newpart = True
            else:
                bpgraph._add_visible_units(
                    num_new_visibles=num_vis,
                    position=pos,
                    initial_weights=each_case,
                    initial_bias=each_case,
                    initial_offsets=each_case,
                )
                check_newpart = False
            compare_BipartiteGraph_after_adding(
                bpgraph, bpgraph_old, pos, num_vis, "vis", check_newpart
            )
        print(" successfully passed!")
        sys.stdout.flush()

    def test__remove_visible_units(self):
        sys.stdout.write("BipartiteGraph -> Performing remove_visible_units test ...")
        sys.stdout.flush()
        bpgraph = BipartiteGraph(number_visibles=4, number_hiddens=3)
        bpgraph_old = copy.deepcopy(bpgraph)
        indices = (0, 1, 3)
        bpgraph._remove_visible_units(indices)
        compare_BipartiteGraph_after_removing(
            bpgraph, bpgraph_old, indices, unit_type="vis"
        )
        print(" successfully passed!")
        sys.stdout.flush()

    def test_get_parameters(self):
        """check whether the return of the function is iterable"""
        sys.stdout.write("BipartiteGraph -> Performing get_parameters test ...")
        sys.stdout.flush()
        bpgraph = BipartiteGraph(number_visibles=2, number_hiddens=3)
        parameter_list = bpgraph.get_parameters()
        # provide more error information here
        # check whether the parameter actually exisit in the instance?
        assert isinstance(parameter_list, Iterable)
        print(" successfully passed!")
        sys.stdout.flush()

    def test_update_parameters(self):
        """use the old parameters to check the updated parameters"""
        sys.stdout.write("BipartiteGraph -> Performing update_parameters test ...")
        sys.stdout.flush()
        bpgraph = BipartiteGraph(number_visibles=2, number_hiddens=3)
        old_param = copy.deepcopy(bpgraph.get_parameters())
        bpgraph.update_parameters(old_param)
        for each_new_param, each_old_param in zip(bpgraph.get_parameters(), old_param):
            assert np.all(each_new_param == 2 * each_old_param)
        print(" successfully passed!")
        sys.stdout.flush()

    def test_update_offsets(self):
        """use the double of the old offsets to check the updated offsets."""
        sys.stdout.write("BipartiteGraph -> Performing update_offsets test ...")
        sys.stdout.flush()
        bpgraph = BipartiteGraph(
            number_visibles=2,
            number_hiddens=3,
            initial_visible_offsets=1.0,
            initial_hidden_offsets=1.0,
        )
        old_ov = copy.deepcopy(bpgraph.ov)
        old_oh = copy.deepcopy(bpgraph.oh)
        old_bv = copy.deepcopy(bpgraph.bv)
        old_bh = copy.deepcopy(bpgraph.bh)
        offset_ratio = 0.1
        bpgraph.update_offsets(
            new_visible_offsets=2 * old_ov,
            new_hidden_offsets=2 * old_oh,
            update_visible_offsets=offset_ratio,
            update_hidden_offsets=offset_ratio,
        )
        assert np.all(bpgraph.ov == (1 + offset_ratio) * old_ov)
        assert np.all(bpgraph.oh == (1 + offset_ratio) * old_oh)
        assert np.all(bpgraph.bv == old_bv + np.dot(old_oh, bpgraph.w.T) * offset_ratio)
        assert np.all(bpgraph.bh == old_bh + np.dot(old_ov, bpgraph.w) * offset_ratio)
        print(" successfully passed!")
        sys.stdout.flush()

    def test_visible_activation(self):
        """use the double of the old offsets to check the updated offsets."""
        sys.stdout.write("BipartiteGraph -> Performing visible_activation test ...")
        sys.stdout.flush()
        bpgraph = BipartiteGraph(
            number_visibles=2,
            number_hiddens=3,
            initial_visible_offsets=1.0,
            initial_hidden_offsets=1.0,
        )
        visact = bpgraph.visible_activation_function.f(
            np.dot(np.ones((1, 3)) - bpgraph.oh, bpgraph.w.T) + bpgraph.bv
        )
        visact_test = bpgraph.visible_activation(np.ones((1, 3)))
        assert np.all(visact == visact_test)
        print(" successfully passed!")
        sys.stdout.flush()

    def test_hidden_activation(self):
        """use the double of the old offsets to check the updated offsets."""
        sys.stdout.write("BipartiteGraph -> Performing hidden_activation test ...")
        sys.stdout.flush()
        bpgraph = BipartiteGraph(
            number_visibles=2,
            number_hiddens=3,
            initial_visible_offsets=1.0,
            initial_hidden_offsets=1.0,
        )
        hidact = bpgraph.hidden_activation_function.f(
            np.dot(np.ones((1, 2)) - bpgraph.ov, bpgraph.w) + bpgraph.bh
        )
        hidact_test = bpgraph.hidden_activation(np.ones((1, 2)))
        assert np.all(hidact == hidact_test)
        print(" successfully passed!")
        sys.stdout.flush()


class Test_StackOfBipartiteGraphs(unittest.TestCase):
    def test___init__(self):
        sys.stdout.write(
            "StackOfBipartiteGraphs -> Performing initialzation and property test ..."
        )
        sys.stdout.flush()

        # Check init scalar
        number_visibles1 = 2
        number_hiddens1 = 4
        number_visibles2 = 4
        number_hiddens2 = 3
        number_visibles3 = 3
        number_hiddens3 = 2

        np.random.seed(42)
        model1 = BipartiteGraph(
            number_visibles=number_visibles1,
            number_hiddens=number_hiddens1,
            data=None,
            initial_weights=1,
            initial_visible_bias=1,
            initial_hidden_bias=1,
            initial_visible_offsets=0,
            initial_hidden_offsets=0,
            dtype=np.float64,
        )

        model2 = BipartiteGraph(
            number_visibles=number_visibles2,
            number_hiddens=number_hiddens2,
            data=None,
            initial_weights=2,
            initial_visible_bias=2,
            initial_hidden_bias=2,
            initial_visible_offsets=0,
            initial_hidden_offsets=0,
            dtype=np.float64,
        )

        model3 = BipartiteGraph(
            number_visibles=number_visibles3,
            number_hiddens=number_hiddens3,
            data=None,
            initial_weights=3,
            initial_visible_bias=3,
            initial_hidden_bias=3,
            initial_visible_offsets=0,
            initial_hidden_offsets=0,
            dtype=np.float64,
        )

        stack = StackOfBipartiteGraphs([model1, model2, model3])

        assert np.all(stack.input_dim == number_visibles1)
        assert np.all(stack.output_dim == number_hiddens3)
        assert np.all(stack.num_layers == 3)
        assert np.all(stack.depth == 4)
        assert np.all(stack[1].dtype == np.float64)

        print(" successfully passed!")
        sys.stdout.flush()

    def test_append_pop(self):
        sys.stdout.write("StackOfBipartiteGraphs -> Performing pop and append test ...")
        sys.stdout.flush()

        # Check init scalar
        number_visibles1 = 2
        number_hiddens1 = 4
        number_visibles2 = 4
        number_hiddens2 = 3
        number_visibles3 = 3
        number_hiddens3 = 2

        np.random.seed(42)
        model1 = BipartiteGraph(
            number_visibles=number_visibles1,
            number_hiddens=number_hiddens1,
            data=None,
            initial_weights=1,
            initial_visible_bias=1,
            initial_hidden_bias=1,
            initial_visible_offsets=0,
            initial_hidden_offsets=0,
            dtype=np.float64,
        )

        model2 = BipartiteGraph(
            number_visibles=number_visibles2,
            number_hiddens=number_hiddens2,
            data=None,
            initial_weights=2,
            initial_visible_bias=2,
            initial_hidden_bias=2,
            initial_visible_offsets=0,
            initial_hidden_offsets=0,
            dtype=np.float64,
        )

        model3 = BipartiteGraph(
            number_visibles=number_visibles3,
            number_hiddens=number_hiddens3,
            data=None,
            initial_weights=3,
            initial_visible_bias=3,
            initial_hidden_bias=3,
            initial_visible_offsets=0,
            initial_hidden_offsets=0,
            dtype=np.float64,
        )

        stack = StackOfBipartiteGraphs([model1, model2])
        assert np.all(stack.input_dim == number_visibles1)
        assert np.all(stack.output_dim == number_hiddens2)
        assert np.all(stack.num_layers == 2)
        assert np.all(stack.depth == 3)

        stack.append_layer(model3)
        assert np.all(stack.input_dim == number_visibles1)
        assert np.all(stack.output_dim == number_hiddens3)
        assert np.all(stack.num_layers == 3)
        assert np.all(stack.depth == 4)
        assert np.all(stack[1].dtype == np.float64)

        model2_new = BipartiteGraph(
            number_visibles=number_visibles2,
            number_hiddens=number_hiddens2,
            data=None,
            initial_weights=2,
            initial_visible_bias=2,
            initial_hidden_bias=2,
            initial_visible_offsets=0,
            initial_hidden_offsets=0,
            dtype=np.int32,
        )
        stack[1] = model2_new
        assert np.all(stack[1].dtype == np.int32)

        stack.pop_last_layer()
        assert np.all(stack.input_dim == number_visibles1)
        assert np.all(stack.output_dim == number_hiddens2)
        assert np.all(stack.num_layers == 2)
        assert np.all(stack.depth == 3)

        stack.append_layer(model3)
        assert np.all(stack.depth == 4)
        assert np.all(stack.output_dim == number_hiddens3)

        stack.pop_last_layer()
        assert np.all(stack.input_dim == number_visibles1)
        assert np.all(stack.output_dim == number_hiddens2)
        assert np.all(stack.num_layers == 2)
        assert np.all(stack.depth == 3)

        stack.pop_last_layer()
        assert np.all(stack.input_dim == number_visibles1)
        assert np.all(stack.output_dim == number_hiddens1)
        assert np.all(stack.num_layers == 1)
        assert np.all(stack.depth == 2)

        stack.pop_last_layer()
        assert stack.input_dim is None
        assert stack.output_dim is None
        assert np.all(stack.num_layers == 0)
        assert np.all(stack.depth == 1)

        stack.pop_last_layer()
        assert stack.input_dim is None
        assert stack.output_dim is None

        assert np.all(stack.num_layers == 0)
        assert np.all(stack.depth == 1)

        print(" successfully passed!")
        sys.stdout.flush()

    def test_forward_backward_reconstruct(self):
        sys.stdout.write(
            "StackOfBipartiteGraphs -> Performing forward, backward, and reconstruct test ..."
        )
        sys.stdout.flush()

        # Check init scalar
        number_visibles1 = 2
        number_hiddens1 = 4
        number_visibles2 = 4
        number_hiddens2 = 3
        number_visibles3 = 3
        number_hiddens3 = 2

        np.random.seed(42)
        model1 = BipartiteGraph(
            number_visibles=number_visibles1,
            number_hiddens=number_hiddens1,
            data=None,
            initial_weights=1,
            initial_visible_bias=1,
            initial_hidden_bias=1,
            initial_visible_offsets=0,
            initial_hidden_offsets=0,
            dtype=np.float64,
        )

        model2 = BipartiteGraph(
            number_visibles=number_visibles2,
            number_hiddens=number_hiddens2,
            data=None,
            initial_weights=2,
            initial_visible_bias=2,
            initial_hidden_bias=2,
            initial_visible_offsets=0,
            initial_hidden_offsets=0,
            dtype=np.float64,
        )

        model3 = BipartiteGraph(
            number_visibles=number_visibles3,
            number_hiddens=number_hiddens3,
            data=None,
            initial_weights=3,
            initial_visible_bias=3,
            initial_hidden_bias=3,
            initial_visible_offsets=0,
            initial_hidden_offsets=0,
            dtype=np.float64,
        )

        stack = StackOfBipartiteGraphs([model1, model2, model3])

        forward_target = np.array([[0.97356463, 0.64265444], [0.974248, 0.64951497]])
        backward_target = np.array([[0.89076868, 0.78815199], [0.93154994, 0.82386556]])
        rec_target = np.array([[0.85960142, 0.74341554], [0.86000094, 0.74392953]])

        assert np.sum(
            np.abs(stack.forward_propagate(np.array([[1, 2], [3, 4]])) - forward_target)
            < 0.000001
        )
        assert np.sum(
            np.abs(
                stack.backward_propagate(np.array([[1, 2], [3, 4]])) - backward_target
            )
            < 0.000001
        )
        assert np.sum(
            np.abs(
                stack.backward_propagate(
                    stack.forward_propagate(np.array([[1, 2], [3, 4]]))
                )
                - rec_target
            )
            < 0.000001
        )
        assert np.sum(
            np.abs(stack.reconstruct(np.array([[1, 2], [3, 4]])) - rec_target)
            < 0.000001
        )

        print(" successfully passed!")
        sys.stdout.flush()


if __name__ == "__main__":
    unittest.main()
