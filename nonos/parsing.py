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


def parse_center_size(
    center, size, xarr: np.ndarray, yarr: np.ndarray, dim: int
) -> (Tuple[float, float], Tuple[float, float]):
    if not is_set(center):
        if dim == 2:
            center = ((xarr.max() + xarr.min()) / 2, (yarr.max() + yarr.min()) / 2)
        elif dim == 1:
            center = [(xarr.max() + xarr.min()) / 2]
        else:
            raise ValueError("dim has to be 1 or 2.")
    if not is_set(size):
        if dim == 2:
            size = (xarr.max() - xarr.min(), yarr.max() - yarr.min())
        elif dim == 1:
            size = [xarr.max() - xarr.min()]
        else:
            raise ValueError("dim has to be 1 or 2.")

    if len(center) != dim:
        raise ValueError(
            f"Need to parse a range from sequence {center} with exactly {dim} values."
        )
    if len(size) != dim:
        raise ValueError(
            f"Need to parse a range from sequence {size} with exactly {dim} values."
        )

    return (center, size)


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
