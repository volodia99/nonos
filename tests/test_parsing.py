import re
import pytest
from nonos.parsing import parse_output_number_range

@pytest.mark.parametrize(
    "received, expected", [
        (1, [1]),
        (0, [0]),
        ([0, 1], [0, 1]),
        ([0, 2], [0, 1, 2]),
        ([0, 5], [0, 1, 2, 3, 4, 5]),
        ([10, 20, 5], [10, 15, 20]),
        ([0, 299, 50], [0, 50, 100, 150, 200, 250]),
    ]
)
def test_range_outputs(received, expected):
    assert parse_output_number_range(received) == expected

def test_unparseable_data():
    with pytest.raises(ValueError, match="Can't parse a range from unset values without a max."):
        parse_output_number_range('unset')

def test_invalid_request():
    with pytest.raises(ValueError, match="No output beyond 5 is available, but 10 was requested."):
        parse_output_number_range([1, 10], maxval=5)

def test_invalid_nargs():
    with pytest.raises(ValueError, match=re.escape("Can't parse a range from sequence [1, 2, 3, 4] with more than 3 values.")):
        parse_output_number_range([1, 2, 3, 4])

@pytest.mark.parametrize(
    "received", [
        [1, 0],
        [1, 0, 1],
        [1, 0, 2],
    ]
)
def test_invalid_range(received):
    with pytest.raises(ValueError, match="Can't parse a range with max < min."):
        parse_output_number_range(received)