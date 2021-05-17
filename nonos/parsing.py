from typing import Any, List, Optional, Tuple, Union

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

    if not isinstance(on, list):
        if isinstance(on, int):
            return [on]

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


def parse_vmin_vmax(vmin, vmax, diff: bool, data: np.ndarray) -> Tuple[float, float]:
    if not is_set(vmin):
        vmin = data.min()
    if not is_set(vmax):
        vmax = -data.min() if diff else data.max()
    return vmin, vmax


def parse_rmin_rmax(rmin, rmax, array) -> Tuple[float, float]:
    if not is_set(rmin):
        rmin = array.min()
    if not is_set(rmax):
        rmax = array.max()
    return rmin, rmax


def parse_image_format(s: Optional[str]) -> str:
    from matplotlib.backend_bases import FigureCanvasBase

    if not is_set(s):
        return FigureCanvasBase.get_default_filetype()

    _, _, ext = s.rpartition(".")
    if ext not in (
        available := list(FigureCanvasBase.get_supported_filetypes().keys())
    ):
        raise ValueError(
            f"Received unknown file format '{s}'. "
            f"Available formated are {available}."
        )
    return ext
