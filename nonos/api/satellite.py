import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from lick.lick import lick_box
from scipy.ndimage.filters import uniform_filter1d

from nonos.api.analysis import GasField, Plotable, temporal
from nonos.api.from_simulation import Parameters


def file_analysis(filename, *, inifile="", code="", directory="", norb=None):
    fullpath = os.path.join(directory, filename)
    with open(fullpath) as f1:
        data = f1.readlines()
    y = [[v for v in re.split(r"[\t ]+", r)] for r in data]
    columns = np.array(y, dtype="float64").T
    if norb is not None:
        init = Parameters(inifile=inifile, code=code, directory=directory)
        init.loadIniFile()
        if init.code == "idefix" and "analysis" in init.inifile["Output"].keys():
            analysis = init.inifile["Output"]["analysis"]
            rpini = init.inifile["Planet"]["dpl"]
            Ntmean = round(norb * 2 * np.pi * pow(rpini, 1.5) / analysis)
            for i in range(1, len(columns) - 1):
                columns[i] = uniform_filter1d(columns[i], Ntmean)
        else:
            raise NotImplementedError(
                f"moving average on {norb} orbits is not implemented for the code {init.code}"
            )
    return columns


def planet_analysis(planet_number, *, inifile="", code="", directory="", norb=None):
    init = Parameters(inifile=inifile, code=code, directory=directory)
    init.loadIniFile()
    init.loadPlanetFile(planet_number=planet_number)
    return init


def save_temporal(
    field: str,
    onbeg: int,
    onend: int,
    operation: str,
    *,
    directory: str = "",
    planet_corotation=0,
):
    temporal_evolution = temporal(
        field,
        operation,
        onbeg,
        onend=onend,
        directory=directory,
        planet_corotation=planet_corotation,
    ).save(directory=directory)
    return temporal_evolution


def load_fields(
    field_operation: Dict[str, Any],
    fields: List[str],
    operations: List[Tuple[str, str]],
    onbeg: int,
    onend: int,
    *,
    directory: str = "",
    temporal_bool: bool = True,
    snapshot_bool: bool = True,
    planet_corotation=0,
):
    for field in fields:
        for operation, operation_shortcut in operations:
            if snapshot_bool:
                field_operation[f"{field}{operation_shortcut}"] = temporal(
                    field,
                    operation,
                    onend,
                    onend=None,
                    directory=directory,
                    planet_corotation=planet_corotation,
                )
            if temporal_bool:
                try:
                    field_operation[f"{field}T{operation_shortcut}"] = temporal(
                        f"{field}T_{onbeg}_{onend}",
                        operation,
                        onend,
                        onend=None,
                        directory=directory,
                    )
                except FileNotFoundError:
                    save_temporal(
                        field,
                        onbeg,
                        onend,
                        operation,
                        directory=directory,
                        planet_corotation=planet_corotation,
                    )
                    field_operation[f"{field}T{operation_shortcut}"] = temporal(
                        f"{field}T_{onbeg}_{onend}",
                        operation,
                        onend,
                        onend=None,
                        directory=directory,
                    )
    return field_operation


class NonosLick:
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        lx: GasField,
        ly: GasField,
        field: GasField,
        *,
        xmin: Optional[float] = None,
        xmax: Optional[float] = None,
        ymin: Optional[float] = None,
        ymax: Optional[float] = None,
        size_interpolated: int = 1000,
        niter_lic: int = 6,
        kernel_length: int = 101,
        method: str = "linear",
        light_source: bool = True,
    ):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

        # (x,y) are 2D meshgrids at cell centers
        self.X, self.Y, self.LINE1, self.LINE2, self.F, self.lick = lick_box(
            x,
            y,
            lx.data,
            ly.data,
            field.data,
            xmin=self.xmin,
            xmax=self.xmax,
            ymin=self.ymin,
            ymax=self.ymax,
            size_interpolated=size_interpolated,
            niter_lic=niter_lic,
            kernel_length=kernel_length,
            method=method,
            light_source=light_source,
        )

    def plot(
        self,
        fig,
        ax,
        *,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        alpha: float = 0.45,
        log: bool = False,
        cmap=None,
        title: Optional[str] = None,
        density_streamlines: Optional[float] = None,
        **kwargs,
    ):
        dict_background = {}
        dict_background["field"] = "background"
        dict_background["abscissa"] = "x"
        dict_background["ordinate"] = "y"
        dict_background[dict_background["field"]] = self.F
        dict_background[dict_background["abscissa"]] = self.X
        dict_background[dict_background["ordinate"]] = self.Y

        dict_lick = {}
        dict_lick["field"] = "lick"
        dict_lick["abscissa"] = "x"
        dict_lick["ordinate"] = "y"
        dict_lick[dict_lick["field"]] = self.lick
        dict_lick[dict_lick["abscissa"]] = self.X
        dict_lick[dict_lick["ordinate"]] = self.Y

        im = Plotable(dict_background).plot(
            fig,
            ax,
            vmin=vmin,
            vmax=vmax,
            log=log,
            cmap=cmap,
            filename=None,
            dpi=500,
            title=title,
            shading="nearest",
            rasterized=True,
        )
        Plotable(dict_lick).plot(
            fig,
            ax,
            log=False,
            cmap="binary_r",
            filename=None,
            dpi=500,
            title=None,
            alpha=alpha,
            shading="nearest",
            rasterized=True,
        )
        if density_streamlines is not None:
            ax.streamplot(
                self.X,
                self.Y,
                self.LINE1,
                self.LINE2,
                density=density_streamlines,
                arrowstyle="->",
                linewidth=0.5,
                color="k",
                # color=np.log10(self.F*np.sqrt(self.LINE1**2+self.LINE2**2)),#/np.max(np.log10(self.F*np.sqrt(self.LINE1**2+self.LINE2**2))),
                # cmap=cb.cbmap("binary_r"),
            )
        ax.set_xlim(self.xmin, self.xmax)
        ax.set_ylim(self.ymin, self.ymax)
        return im


def compute(
    field: str,
    data: np.ndarray,
    ref: GasField,
    *,
    inifile: str = "",
    code: str = "",
    directory: str = "",
    rotate_grid: bool = False,
):
    ret_data = data
    ret_coords = ref.coords
    geometry = ret_coords.geometry
    return GasField(
        field,
        ret_data,
        ret_coords,
        geometry,
        ref.on,
        operation=ref.operation,
        inifile=inifile,
        code=code,
        directory=directory,
        rotate_grid=rotate_grid,
    )
