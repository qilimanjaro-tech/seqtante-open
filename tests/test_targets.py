# Copyright 2026 Qilimanjaro Quantum Tech
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for experiment targets."""

import operator
from dataclasses import FrozenInstanceError, asdict

import pytest

from seqtante_open.targets import CouplerTarget, ExperimentTarget, QubitTarget


class TestNames:
    def test_qubit(self):
        assert str(QubitTarget(14)) == "q14"

    def test_coupler(self):
        assert str(CouplerTarget(2, 3)) == "c2_3"

    def test_index_zero(self):
        assert str(QubitTarget(0)) == "q0"
        assert str(CouplerTarget(0, 1)) == "c0_1"

    def test_repr_hides_kind(self):
        assert repr(QubitTarget(14)) == "QubitTarget(qubit_id=14)"
        assert repr(CouplerTarget(2, 3)) == "CouplerTarget(qubit_id_1=2, qubit_id_2=3)"


class TestKind:
    def test_kind(self):
        assert QubitTarget(1).kind == "qubit"
        assert CouplerTarget(1, 2).kind == "coupler"

    def test_kind_is_in_asdict(self):
        assert asdict(QubitTarget(14)) == {"kind": "qubit", "qubit_id": 14}
        assert asdict(CouplerTarget(2, 3)) == {"kind": "coupler", "qubit_id_1": 2, "qubit_id_2": 3}

    def test_kind_cannot_be_passed(self):
        with pytest.raises(TypeError):
            QubitTarget(1, kind="coupler")  # type: ignore[call-arg]

    def test_frozen(self):
        target = QubitTarget(1)
        with pytest.raises(FrozenInstanceError):
            target.qubit_id = 2  # type: ignore[misc]

    def test_subclasses_are_experiment_targets(self):
        assert isinstance(QubitTarget(1), ExperimentTarget)
        assert isinstance(CouplerTarget(1, 2), ExperimentTarget)


class TestEquality:
    def test_same_target(self):
        qubit_a, qubit_b = QubitTarget(1), QubitTarget(1)
        coupler_a, coupler_b = CouplerTarget(1, 2), CouplerTarget(1, 2)
        assert qubit_a is not qubit_b
        assert qubit_a == qubit_b
        assert coupler_a is not coupler_b
        assert coupler_a == coupler_b

    def test_different_targets(self):
        assert QubitTarget(1) != QubitTarget(2)
        assert CouplerTarget(1, 2) != CouplerTarget(1, 3)

    def test_qubit_is_not_a_coupler(self):
        assert QubitTarget(1) != CouplerTarget(1, 2)

    def test_equal_to_its_name_both_ways(self):
        assert QubitTarget(14) == "q14"
        assert operator.eq("q14", QubitTarget(14))
        assert CouplerTarget(2, 3) == "c2_3"
        assert operator.eq("c2_3", CouplerTarget(2, 3))

    @pytest.mark.parametrize("other", ["q15", "q014", "c1_14", "14"])
    def test_not_equal_to_another_name(self, other):
        assert QubitTarget(14) != other

    @pytest.mark.parametrize("other", [14, None, ["q14"]])
    def test_not_equal_to_other_types(self, other):
        assert QubitTarget(14) != other


class TestHash:
    def test_name_finds_a_target_key(self):
        parameters = {QubitTarget(1): "a", CouplerTarget(1, 2): "b"}
        assert parameters["q1"] == "a"
        assert parameters["c1_2"] == "b"
        assert "q1" in parameters

    def test_target_finds_a_name_key(self):
        assert {"q1": "a"}[QubitTarget(1)] == "a"

    def test_set_holds_one_of_each(self):
        assert len({QubitTarget(1), QubitTarget(1), "q1", CouplerTarget(1, 2), "c1_2"}) == 2


class TestFromStr:
    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("q0", QubitTarget(0)),
            ("q14", QubitTarget(14)),
            ("q100", QubitTarget(100)),
            ("c0_1", CouplerTarget(0, 1)),
            ("c2_13", CouplerTarget(2, 13)),
        ],
    )
    def test_valid(self, name, expected):
        target = ExperimentTarget.from_str(name)
        assert type(target) is type(expected)
        assert target == expected
        assert str(target) == name

    def test_narrows_with_isinstance(self):
        target = ExperimentTarget.from_str("c2_3")
        assert isinstance(target, CouplerTarget)
        assert (target.qubit_id_1, target.qubit_id_2) == (2, 3)

    @pytest.mark.parametrize(
        "name",
        [
            "q01",
            "q00",
            "c01_2",
            "c1_02",
            "c2_1",
            "c1_1",
            "q",
            "c1",
            "c1_",
            "c_1",
            "c1_2_3",
            "q-1",
            "Q1",
            "C1_2",
            " q1",
            "q1 ",
            "x1",
            "",
            "all_q",
            "all_c",
            "q1٣",
            "c1_2٣",
        ],
    )
    def test_invalid_name(self, name):
        with pytest.raises(ValueError, match="Invalid target"):
            ExperimentTarget.from_str(name)

    @pytest.mark.parametrize("name", [1, None, 1.0, ["q1"]])
    def test_not_a_string(self, name):
        with pytest.raises(ValueError, match="Invalid target"):
            ExperimentTarget.from_str(name)  # type: ignore[arg-type]


class TestConstruction:
    @pytest.mark.parametrize("qubit_id", [-1, True, 1.0, "1", None])
    def test_invalid_qubit_id(self, qubit_id):
        with pytest.raises(ValueError, match="Invalid qubit_id"):
            QubitTarget(qubit_id)

    @pytest.mark.parametrize(("qubit_id_1", "qubit_id_2"), [(-1, 2), (1, "2"), (False, 1)])
    def test_invalid_coupler_index(self, qubit_id_1, qubit_id_2):
        with pytest.raises(ValueError, match="Invalid qubit_id_"):
            CouplerTarget(qubit_id_1, qubit_id_2)

    @pytest.mark.parametrize(("qubit_id_1", "qubit_id_2"), [(2, 1), (1, 1)])
    def test_coupler_must_be_ordered(self, qubit_id_1, qubit_id_2):
        with pytest.raises(ValueError, match="qubit_id_1 must be lower than qubit_id_2"):
            CouplerTarget(qubit_id_1, qubit_id_2)


class TestBetween:
    @pytest.mark.parametrize(("qubit_a", "qubit_b"), [(1, 2), (2, 1)])
    def test_orders_the_pair(self, qubit_a, qubit_b):
        assert CouplerTarget.between(qubit_a, qubit_b) == CouplerTarget(1, 2)

    def test_same_qubit(self):
        with pytest.raises(ValueError, match="qubit_id_1 must be lower than qubit_id_2"):
            CouplerTarget.between(1, 1)
