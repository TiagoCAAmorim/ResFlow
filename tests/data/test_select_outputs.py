"""Unit tests of the output selection (no data needed)."""
import pytest

from resflow.preprocess.model import select_outputs


def test_docstring_examples():
    assert select_outputs([1.0, 2.0, 3.0, 5.0, 6.0, 7.0, 10.0]) == ([0, 1, 2, 3, 4, 5], [[0, 2], [3, 5]])
    assert select_outputs([0.0, 0.5, 1.0, 1.5, 2.0]) == ([0, 2, 4], [[0, 2]])
    assert select_outputs([0.0, 1.0, 1.4, 2.4, 3.4]) == ([0, 1, 2, 3, 4], [[0, 1], [2, 4]])


def test_ten_day_outputs():
    days = [0.0, 10.0, 20.0, 30.0]
    assert select_outputs(days, step=10.0) == ([0, 1, 2, 3], [[0, 3]])
    # step 1 keeps nothing on 10-day outputs
    assert select_outputs(days, step=1.0) == ([], [])


def test_single_output_series_are_dropped():
    assert select_outputs([0.0, 5.0], step=1.0) == ([], [])


@pytest.mark.parametrize('values', [[], [3.0]])
def test_degenerate(values):
    assert select_outputs(values) == ([], [])
