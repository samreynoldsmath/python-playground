"""
python3 -m pytest symbolic/tests -s
"""

import pytest

from symbolic import Edge

GOOD_STR = "t"


def _test_good(good_str: str) -> None:
    e = Edge(x=good_str, y=GOOD_STR)
    e.sample(num_pts=8)
    for x, y in zip(e.x, e.y):
        assert isinstance(x, (float, int))
        assert isinstance(y, (float, int))


def _test_bad(bad_str: str) -> None:
    with pytest.raises(ValueError):
        Edge(x=bad_str, y=GOOD_STR)


def test_good_str_cos() -> None:
    _test_good(good_str="cos(t)")


def test_good_str_gaussian() -> None:
    _test_good(good_str="exp(-t**2)")


def test_good_str_constant_literal() -> None:
    _test_good(good_str="6.28")


def test_good_str_constant_symbolic() -> None:
    _test_good(good_str="2 * pi")


def test_bad_str_x() -> None:
    _test_bad(bad_str="x")


def test_bad_str_sin_T() -> None:
    _test_bad(bad_str="sin(T)")


def test_bad_str_sim_t() -> None:
    # _test_bad(bad_str="sim(t)")
    #
    # Unfortunately, it seems very difficult for sympy to recognize that "sim"
    # is not a valid function. At least for now, we will have to settle for a
    # TypeError being raised when the expression is evaluated.
    #
    # See symbolic/tests/traceback.txt for the full traceback.
    edge = Edge(x="sim(t)", y=GOOD_STR)
    with pytest.raises(TypeError) as e:
        print(e)
        edge.sample(num_pts=8)
