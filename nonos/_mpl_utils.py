from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from matplotlib.colors import SymLogNorm


def get_log_minorticks(vmin, vmax):
    """calculate positions of linear minorticks on a log colorbar

    Parameters
    ----------
    vmin : float
        the minimum value in the colorbar
    vmax : float
        the maximum value in the colorbar

    """
    # taken from yt 4.0.2 (yt.visualization.plot_container)

    expA = np.floor(np.log10(vmin))
    expB = np.floor(np.log10(vmax))
    cofA = np.ceil(vmin / 10**expA).astype("int64")
    cofB = np.floor(vmax / 10**expB).astype("int64")
    lmticks = []
    while cofA * 10**expA <= cofB * 10**expB:
        if expA < expB:
            lmticks = np.hstack((lmticks, np.linspace(cofA, 9, 10 - cofA) * 10**expA))
            cofA = 1
            expA += 1
        else:
            lmticks = np.hstack(
                (lmticks, np.linspace(cofA, cofB, cofB - cofA + 1) * 10**expA)
            )
            expA += 1
    return np.array(lmticks)


def get_symlog_minorticks(linthresh, vmin, vmax):
    """calculate positions of linear minorticks on a symmetric log colorbar

    Parameters
    ----------
    linthresh : float
        the threshold for the linear region
    vmin : float
        the minimum value in the colorbar
    vmax : float
        the maximum value in the colorbar

    """
    # taken from yt 4.0.2 (yt.visualization.plot_container)

    if vmin > 0:
        return get_log_minorticks(vmin, vmax)
    elif vmax < 0 and vmin < 0:
        return -get_log_minorticks(-vmax, -vmin)
    elif vmin == 0:
        return np.hstack((0, get_log_minorticks(linthresh, vmax)))
    elif vmax == 0:
        return np.hstack((-get_log_minorticks(linthresh, -vmin)[::-1], 0))
    else:
        return np.hstack(
            (
                -get_log_minorticks(linthresh, -vmin)[::-1],
                0,
                get_log_minorticks(linthresh, vmax),
            )
        )


def set_symlog_minor_ticks(norm: "SymLogNorm", cax) -> None:
    # heavily inspired from yt 4.0.2
    flinthresh = 10 ** np.floor(np.log10(norm.linthresh))
    absmax = np.abs((norm.vmin, norm.vmax)).max()
    if (absmax - flinthresh) / absmax < 0.1:
        flinthresh /= 10
    mticks = get_symlog_minorticks(flinthresh, norm.vmin, norm.vmax)
    cax.yaxis.set_ticks(mticks, minor=True)
