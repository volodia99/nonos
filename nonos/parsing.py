from typing import Any, List, Optional, Union

import numpy as np


def is_set(x: Any) -> bool:
    return x not in (None, "unset")


def parse_output_number_range(
    on: Optional[Union[List[int], int, str]], maxval: Optional[int] = None
) -> List[int]:
    if not is_set(on):
        if maxval is None:
            raise ValueError("Can't parse a range from unset values without a max.")
        return [maxval]

    if isinstance(on, int):
        return [on]

    assert isinstance(on, list) and all(isinstance(o, int) for o in on)

    if len(on) > 3:
        raise ValueError(
            f"Can't parse a range from sequence {on} with more than 3 values."
        )
    if len(on) == 1:
        return on

    if on[1] < on[0]:
        raise ValueError("Can't parse a range with max < min.")

    # make the upper boundary inclusive
    on[1] += 1
    ret = list(range(*on))
    if maxval is not None and (max_requested := ret[-1]) > maxval:
        raise ValueError(
            f"No output beyond {maxval} is available, but {max_requested} was requested."
        )
    return ret


def parse_range(extent, dim: int):
    if not is_set(extent):
        if dim == 2:
            return (None, None, None, None)
        elif dim == 1:
            return (None, None)
        else:
            raise ValueError("dim has to be 1 or 2.")

    if len(extent) != 2 * dim:
        raise ValueError(
            f"Received sequence `extent` with incorrect size {len(extent)}. Expected exactly {2*dim=} values."
        )
    return tuple(float(i) if i != "x" else None for i in extent)


def range_converter(extent, abscissa: np.ndarray, ordinate: np.ndarray):
    trueextent = [abscissa.min(), abscissa.max(), ordinate.min(), ordinate.max()]
    return tuple(i if i is not None else j for (i, j) in zip(extent, trueextent))


def parse_image_format(s: Optional[str]) -> str:
    from matplotlib.backend_bases import FigureCanvasBase

    if not is_set(s):
        return FigureCanvasBase.get_default_filetype()

    assert isinstance(s, str)
    _, _, ext = s.rpartition(".")
    if ext not in (
        available := list(FigureCanvasBase.get_supported_filetypes().keys())
    ):
        raise ValueError(
            f"Received unknown file format '{s}'. "
            f"Available formated are {available}."
        )
    return ext
