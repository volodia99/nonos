import re

import numpy as np
import pytest

from nonos.parsing import (
    parse_image_format,
    parse_output_number_range,
    parse_range,
    parse_vmin_vmax,
)


@pytest.mark.parametrize(
    "received, expected",
    [
        (1, [1]),
        (0, [0]),
        ([0, 1], [0, 1]),
        ([0, 2], [0, 1, 2]),
        ([0, 5], [0, 1, 2, 3, 4, 5]),
        ([10, 20, 5], [10, 15, 20]),
        ([0, 299, 50], [0, 50, 100, 150, 200, 250]),
    ],
)
def test_range_outputs(received, expected):
    assert parse_output_number_range(received) == expected


def test_unparseable_data():
    with pytest.raises(
        ValueError, match="Can't parse a range from unset values without a max."
    ):
        parse_output_number_range("unset")


def test_invalid_request():
    with pytest.raises(
        ValueError, match="No output beyond 5 is available, but 10 was requested."
    ):
        parse_output_number_range([1, 10], maxval=5)


def test_invalid_nargs():
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Can't parse a range from sequence [1, 2, 3, 4] with more than 3 values."
        ),
    ):
        parse_output_number_range([1, 2, 3, 4])


@pytest.mark.parametrize(
    "abscissa, ordinate, dim, received",
    [
        (
            np.linspace(0.2, 10, 100),
            np.linspace(-0.4, 0.4, 100),
            2,
            ("0.4", "8", "0", "-0.2", "0.2"),
        ),
        (np.linspace(0.2, 10, 100), np.linspace(-0.4, 0.4, 100), 1, (0.4, 8, 0.2)),
    ],
)
def test_invalid_nargs_parse_range(abscissa, ordinate, dim, received):
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"Need to parse a range from sequence {received} with exactly {2*dim} values."
        ),
    ):
        parse_range(received, abscissa=abscissa, ordinate=ordinate, dim=dim)


@pytest.mark.parametrize(
    "received",
    [
        [1, 0],
        [1, 0, 1],
        [1, 0, 2],
    ],
)
def test_invalid_range(received):
    with pytest.raises(ValueError, match="Can't parse a range with max < min."):
        parse_output_number_range(received)


@pytest.mark.parametrize(
    "data, expected",
    [
        (np.array([0, 0, 1]), (0, 1)),
        (np.array([0, 1, 1]), (0, 1)),
        (np.array([-1, 0, 1]), (-1, 1)),
    ],
)
def test_nodiff_parse_vmin_vmax(data, expected):
    assert parse_vmin_vmax("unset", "unset", diff=False, data=data) == expected


@pytest.mark.parametrize(
    "data, expected",
    [
        (np.array([0, -2, 1]), (-2, 2)),
        (np.array([0, -2, 2]), (-2, 2)),
        (np.array([0, 0, 0]), (0, 0)),
    ],
)
def test_diff_parse_vmin_vmax(data, expected):
    assert parse_vmin_vmax("unset", "unset", diff=True, data=data) == expected


@pytest.mark.parametrize(
    "abscissa, ordinate, dim, expected",
    [
        (
            np.linspace(0.2, 10, 100),
            np.linspace(-np.pi, np.pi, 100),
            2,
            (0.2, 10.0, -np.pi, np.pi),
        ),
        (np.linspace(0.2, 10, 100), np.zeros(2), 1, (0.2, 10.0)),
    ],
)
def test_parse_range(abscissa, ordinate, dim, expected):
    assert (
        parse_range("unset", abscissa=abscissa, ordinate=ordinate, dim=dim) == expected
    )
    assert (
        parse_range(
            ("0.5", "5", "-0.2", "0.2"),
            abscissa=np.linspace(0.2, 10, 100),
            ordinate=np.linspace(-0.4, 0.4, 100),
            dim=2,
        )
        == (0.5, 5, -0.2, 0.2)
    )
    assert parse_range(
        ("0.4", "9.5"), abscissa=np.linspace(0.2, 10, 100), ordinate=np.zeros(2), dim=1
    ) == (0.4, 9.5)
    assert (
        parse_range(
            ("0.5", "x", "-0.2", "x"),
            abscissa=np.linspace(0.2, 10, 100),
            ordinate=np.linspace(-0.4, 0.4, 100),
            dim=2,
        )
        == (0.5, 10.0, -0.2, 0.4)
    )


@pytest.mark.parametrize(
    "received, expected",
    [
        (".png", "png"),
        (".pdf", "pdf"),
        ("png", "png"),
        ("pdf", "pdf"),
    ],
)
def test_image_format(received, expected):
    assert parse_image_format(received) == expected


def test_invalid_image_format():
    fake_ext = ".pnd"
    with pytest.raises(
        ValueError,
        match=f"^(Received unknown file format '{fake_ext}'. Available formated are)",
    ):
        parse_image_format(fake_ext)
