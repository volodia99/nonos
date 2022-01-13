import glob
import os
from pathlib import Path
from typing import Any, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

from nonos.api.from_simulation import Parameters
from nonos.api.tools import find_around, find_nearest
from nonos.logging import logger


class Plotable:
    def __init__(self, dict_plotable: dict):
        self.dict_plotable = dict_plotable
        self.data = self.dict_plotable[self.dict_plotable["field"]]
        self.dimension = len(self.data.shape)

    def plot(
        self,
        fig,
        ax,
        vmin=None,
        vmax=None,
        log=False,
        cmap="inferno",
        nbin=None,
        filename=None,
        fmt="png",
        dpi=500,
        title=None,
        **kwargs,
    ):
        data = self.data
        if log:
            data = np.log10(data)
        if vmin is None:
            vmin = data.min()
        if vmax is None:
            vmax = data.max()
        if self.dimension == 2:
            self.akey = self.dict_plotable["abscissa"]
            self.okey = self.dict_plotable["ordinate"]
            self.avalue = self.dict_plotable[self.akey]
            self.ovalue = self.dict_plotable[self.okey]
            im = ax.pcolormesh(
                self.avalue,
                self.ovalue,
                data,
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
                **kwargs,
            )
            ax.set_xlabel(self.akey)
            ax.set_ylabel(self.okey)
            if title is not None:
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                cbar = fig.colorbar(
                    im, cax=cax, orientation="vertical"
                )  # , format='%.0e')
                cbar.set_label(title)
            else:
                return im
        if self.dimension == 1:
            self.akey = self.dict_plotable["abscissa"]
            self.avalue = self.dict_plotable[self.akey]
            im = ax.plot(
                self.avalue,
                data,
                **kwargs,
            )
            ax.set_ylim(vmin, vmax)
            ax.set_xlabel(self.akey)
            if title is not None:
                ax.set_ylabel(title)
        if filename is not None:
            plt.savefig(f"{filename}.{fmt}", bbox_inches="tight", dpi=dpi)


class Coordinates:
    """Coordinates class from x1, x2, x3"""

    def __init__(self, geometry: str, x1: np.ndarray, x2: np.ndarray, x3: np.ndarray):
        if x1.shape[0] == 1:
            x1 = np.array([x1[0], x1[0]])
        if x2.shape[0] == 1:
            x2 = np.array([x2[0], x2[0]])
        if x3.shape[0] == 1:
            x3 = np.array([x3[0], x3[0]])
        self.geometry = geometry
        if self.geometry == "cartesian":
            self.x = x1
            self.y = x2
            self.z = x3
            self.cube = ("x", "y", "z")
            self.xmed = 0.5 * (self.x[1:] + self.x[:-1])
            self.ymed = 0.5 * (self.y[1:] + self.y[:-1])
            self.zmed = 0.5 * (self.z[1:] + self.z[:-1])
        if self.geometry == "spherical":
            self.r = x1
            self.theta = x2
            self.phi = x3
            if self.phi.max() - np.pi > np.pi / 2:
                self.phi -= np.pi
            self.cube = ("r", "theta", "phi")
            self.rmed = 0.5 * (self.r[1:] + self.r[:-1])
            self.thetamed = 0.5 * (self.theta[1:] + self.theta[:-1])
            self.phimed = 0.5 * (self.phi[1:] + self.phi[:-1])
        if self.geometry == "polar":
            self.R = x1
            self.phi = x2
            if self.phi.max() - np.pi > np.pi / 2:
                self.phi -= np.pi
            self.z = x3
            self.cube = ("R", "phi", "z")
            self.Rmed = 0.5 * (self.R[1:] + self.R[:-1])
            self.phimed = 0.5 * (self.phi[1:] + self.phi[:-1])
            self.zmed = 0.5 * (self.z[1:] + self.z[:-1])

    @property
    def shape(self):
        """
        Returns
        =======
        shape : tuple
        """
        if self.geometry == "cartesian":
            return len(self.x), len(self.y), len(self.z)
        if self.geometry == "spherical":
            return len(self.r), len(self.theta), len(self.phi)
        if self.geometry == "polar":
            return len(self.R), len(self.phi), len(self.z)

    @property
    def get_attributes(self):
        if self.geometry == "cartesian":
            return {"geometry": self.geometry, "x": self.x, "y": self.y, "z": self.z}
        if self.geometry == "spherical":
            return {
                "geometry": self.geometry,
                "r": self.r,
                "theta": self.theta,
                "phi": self.phi,
            }
        if self.geometry == "polar":
            return {
                "geometry": self.geometry,
                "R": self.R,
                "phi": self.phi,
                "z": self.z,
            }

    @property
    def get_coords(self):
        if self.geometry == "cartesian":
            return {
                "x": self.x,
                "y": self.y,
                "z": self.z,
                "xmed": self.xmed,
                "ymed": self.ymed,
                "zmed": self.zmed,
            }
        if self.geometry == "spherical":
            return {
                "r": self.r,
                "theta": self.theta,
                "phi": self.phi,
                "rmed": self.rmed,
                "thetamed": self.thetamed,
                "phimed": self.phimed,
            }
        if self.geometry == "polar":
            return {
                "R": self.R,
                "phi": self.phi,
                "z": self.z,
                "Rmed": self.Rmed,
                "phimed": self.phimed,
                "zmed": self.zmed,
            }

    def _meshgrid_reduction(self, *reducted):
        for i in reducted:
            if i not in self.cube:
                raise KeyError(f"{i} not in {self.cube}")
        dictcoords = {}
        if len(reducted) <= 2:
            for coords in reducted:
                dictcoords[coords] = vars(self)[coords]
            axis = list(set(reducted) ^ set(self.cube))
            dictmesh = {}
            # 2D map
            if len(axis) == 1:
                dictmesh[reducted[0]], dictmesh[reducted[1]] = np.meshgrid(
                    dictcoords[reducted[0]], dictcoords[reducted[1]]
                )
                axismed = "".join([axis[0], "med"])
                dictmesh[axis[0]] = vars(self)[axismed]
                # carefule: takes "xy", "yz", "zx" (all combinations)
                if "".join(reducted) in "".join((*self.cube, self.cube[0])):
                    ordered = True
                else:
                    ordered = False
                dictmesh["ordered"] = ordered
            # 1D curve
            else:
                dictmesh[reducted[0]] = vars(self)["".join([reducted[0], "med"])]
        else:
            raise ValueError(f"more than 2 coordinates were specified: {reducted}.")
        return dictmesh

    # on demande 'x','y' et la geometry est cartesian -> 'x','y'
    # on demande 'x','y' et la geometry est polaire -> 'R','phi'
    # on demande 'x','y' et la geometry est spherique -> 'r','phi'
    def native_from_wanted(self, *wanted):
        if self.geometry == "cartesian":
            conversion = {
                "x": "x",
                "y": "y",
                "z": "z",
            }
        if self.geometry == "polar":
            conversion = {
                "R": "R",
                "phi": "phi",
                "z": "z",
                "x": "R",
                "y": "phi",
                "r": "R",
                "theta": "z",
            }
        if self.geometry == "spherical":
            conversion = {
                "r": "r",
                "theta": "theta",
                "phi": "phi",
                "x": "r",
                "y": "phi",
                "z": "theta",
                "R": "r",
            }
        for i in wanted:
            if i not in tuple(conversion.keys()):
                raise KeyError(f"{i} not in {tuple(conversion.keys())}")
        if set(wanted) & {"x", "y", "z"} == set(wanted):
            target_geometry = "cartesian"
        elif set(wanted) & {"R", "phi", "z"} == set(wanted):
            target_geometry = "polar"
        elif set(wanted) & {"r", "theta", "phi"} == set(wanted):
            target_geometry = "spherical"
        else:
            raise ValueError(f"Unknown wanted plane: {wanted}.")
        native = tuple(conversion[i] for i in wanted)
        return native, target_geometry

    # for 2D arrays
    def target_from_native(self, target_geometry, coords):

        if self.geometry == "polar":
            R, phi, z = (coords["R"], coords["phi"], coords["z"])
            if target_geometry == "cartesian":
                x = R * np.cos(phi)
                y = R * np.sin(phi)
                target_coords = {"x": x, "y": y, "z": z}
            elif target_geometry == "spherical":
                r = np.sqrt(R ** 2 + z ** 2)
                theta = np.arctan2(R, z)
                target_coords = {"r": r, "theta": theta, "phi": phi}
            else:
                raise ValueError(f"Unknown target geometry {target_geometry}.")

        elif self.geometry == "cartesian":
            x, y, z = (coords["x"], coords["y"], coords["z"])
            if target_geometry == "polar":
                R = np.sqrt(x ** 2 + y ** 2)
                phi = np.arctan2(y, x)
                target_coords = {"R": R, "phi": phi, "z": z}
            elif target_geometry == "spherical":
                r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
                theta = np.arctan2(np.sqrt(x ** 2 + y ** 2), z)
                phi = np.arctan2(y, x)
                target_coords = {"r": r, "theta": theta, "phi": phi}
            else:
                raise ValueError(f"Unknown target geometry {target_geometry}.")

        elif self.geometry == "spherical":
            r, theta, phi = (coords["r"], coords["theta"], coords["phi"])
            if target_geometry == "polar":
                R = r * np.sin(theta)
                z = r * np.cos(theta)
                target_coords = {"R": R, "phi": phi, "z": z}
            elif target_geometry == "cartesian":
                x = r * np.sin(theta) * np.cos(phi)
                y = r * np.sin(theta) * np.sin(phi)
                z = r * np.cos(theta)
                if len(theta.shape) == 0:
                    x = r * np.sin(theta) * np.cos(phi)
                    y = r * np.sin(theta) * np.sin(phi)
                    z = np.cos(theta)
                target_coords = {"x": x, "y": y, "z": z}
            else:
                raise ValueError(f"Unknown target geometry {target_geometry}.")

        else:
            raise ValueError(f"Unknown geometry {self.geometry}.")

        target_coords["ordered"] = coords["ordered"]
        return target_coords

    def _meshgrid_conversion(self, *wanted):
        native_from_wanted = self.native_from_wanted(*wanted)
        native = native_from_wanted[0]
        target_geometry = native_from_wanted[1]
        native_meshcoords = self._meshgrid_reduction(*native)
        if len(wanted) == 1:
            return native_meshcoords
        else:
            meshcoords = {}
            if target_geometry == self.geometry:
                meshcoords = native_meshcoords
            else:
                meshcoords = self.target_from_native(target_geometry, native_meshcoords)
            return meshcoords


class GasField:
    """Idefix field class

    Attributes
    ==========
    data : array corresponding to the 3D data cube.
    coords : Coordinates of the data.
        Edge of the cells.
    coordsmed : Coordinates of the data.
        Center of the cells.
    """

    def __init__(
        self,
        field: str,
        data: np.ndarray,
        coords: Coordinates,
        ngeom: str,
        on: int,
        operation: str,
        *,
        inifile: str = "",
        code: str = "",
        directory="",
    ):
        self.field = field
        self.operation = operation
        self.native_geometry = ngeom
        self.data = data
        self.coords = coords
        self.on = on

        self.inifile = inifile
        self.code = code
        self.directory = directory

    @property
    def shape(self) -> Tuple[Any, ...]:
        """
        Returns
        =======
        shape : tuple
        """
        return tuple(i - 1 if i > 1 else i for i in self.coords.shape)

    def map(self, *wanted, planet_corotation: Optional[int] = None):
        data_key = self.field
        # we count the number of 1 in the shape of the data, which gives the real dimension of the data,
        # i.e. the number of reductions already performed (0 -> 3D, 1 -> 2D, 2 -> 1D)
        if self.shape.count(1) not in (1, 2):
            raise ValueError("data has to be 1D or 2D in order to call map.")
        dimension = len(wanted)

        if dimension == 1:
            meshgrid_conversion = self.coords._meshgrid_conversion(*wanted)
            # abscissa = meshgrid_conversion[wanted[0]]
            abscissa_value = list(meshgrid_conversion.values())[0]
            abscissa_key = list(meshgrid_conversion.keys())[0]
            if planet_corotation is not None and "phi" in wanted:
                phip = self.find_phip(planet_number=planet_corotation)
                phicoord = self.coords.phi - phip  # - np.pi
                ipi = find_nearest(phicoord, 0)
                if abs(0 - phicoord[ipi]) > abs(
                    np.ediff1d(find_around(phicoord, 0))[0]
                ):
                    ipi = find_nearest(phicoord, 2 * np.pi)
                if self.native_geometry == "polar":
                    self.data = np.roll(
                        self.data, -ipi - self.coords.phi.shape[0] // 2 + 1, axis=1
                    )
                elif self.native_geometry == "spherical":
                    self.data = np.roll(
                        self.data, -ipi - self.coords.phi.shape[0] // 2 + 1, axis=2
                    )
                else:
                    raise NotImplementedError(
                        f"geometry flag '{self.native_geometry}' not implemented yet if planet_corotation"
                    )

            datamoved_tmp = np.moveaxis(self.data, self.shape.index(1), 0)
            datamoved = np.moveaxis(
                datamoved_tmp[0], datamoved_tmp[0].shape.index(1), 0
            )
            dict_plotable = {
                "abscissa": abscissa_key,
                "field": data_key,
                abscissa_key: abscissa_value,
                data_key: datamoved[0],
            }
        if dimension == 2:
            # meshgrid in polar coordinates P, R (if "R", "phi") or R, P (if "phi", "R")
            # idem for all combinations of R,phi,z
            meshgrid_conversion = self.coords._meshgrid_conversion(*wanted)
            abscissa_value, ordinate_value = (
                meshgrid_conversion[wanted[0]],
                meshgrid_conversion[wanted[1]],
            )
            abscissa_key, ordinate_key = (wanted[0], wanted[1])
            native_from_wanted = self.coords.native_from_wanted(*wanted)[0]
            if planet_corotation is not None and "phi" in native_from_wanted:
                phip = self.find_phip(planet_number=planet_corotation)
                phicoord = self.coords.phi - phip  # - np.pi
                # ipi = find_nearest(phicoord, np.pi)
                # if (abs(np.pi-phicoord[ipi])>abs(np.ediff1d(find_around(phicoord, np.pi))[0])):
                #     ipi = find_nearest(phicoord, -np.pi)
                ipi = find_nearest(phicoord, 0)
                if abs(0 - phicoord[ipi]) > abs(
                    np.ediff1d(find_around(phicoord, 0))[0]
                ):
                    ipi = find_nearest(phicoord, 2 * np.pi)
                if self.native_geometry == "polar":
                    self.data = np.roll(
                        self.data, -ipi - self.coords.phi.shape[0] // 2 + 1, axis=1
                    )
                elif self.native_geometry == "spherical":
                    self.data = np.roll(
                        self.data, -ipi - self.coords.phi.shape[0] // 2 + 1, axis=2
                    )
                else:
                    raise NotImplementedError(
                        f"geometry flag '{self.native_geometry}' not implemented yet if planet_corotation"
                    )
            ordered = meshgrid_conversion["ordered"]
            # move the axis of reduction in the front in order to
            # perform the operation 3D(i,j,1) -> 2D(i,j) in a general way,
            # whatever the position of (i,j,1)
            # for that we must be careful to change the order ("ordered") if the data is reversed
            # in practice, this is tricky only if the "1" is in the middle:
            # 3D(i,1,k) -> 2D(i,k) is not a direct triedre anymore, we need to do 2D(i,k).T = 2D(k,i)
            position_of_3d_dimension = self.shape.index(1)
            datamoved = np.moveaxis(self.data, position_of_3d_dimension, 0)
            if position_of_3d_dimension == 1:
                ordered = not ordered
            if ordered:
                data_value = datamoved[0].T
            else:
                data_value = datamoved[0]

            dict_plotable = {
                "abscissa": abscissa_key,
                "ordinate": ordinate_key,
                "field": data_key,
                abscissa_key: abscissa_value,
                ordinate_key: ordinate_value,
                data_key: data_value,
            }
        return Plotable(dict_plotable)

    def save(self, directory="", header_only=False):
        if not header_only:
            if not os.path.exists(os.path.join(directory, self.field.lower())):
                os.mkdir(os.path.join(directory, self.field.lower()))
            filename = os.path.join(
                directory,
                self.field.lower(),
                f"{self.operation}_{self.field}.{self.on:04d}.npy",
            )
            if Path(filename).is_file():
                logger.info(f"{filename} already exists")
            else:
                with open(filename, "wb") as file:
                    np.save(file, self.data)

        group_of_files = list(
            glob.glob1(
                os.path.join(directory, self.field.lower()), f"{self.operation}*"
            )
        )
        header_file = list(
            glob.glob1(os.path.join(directory, "header"), f"header{self.operation}.npy")
        )
        if (len(group_of_files) > 0 and len(header_file) == 0) or header_only:
            if not os.path.exists(os.path.join(directory, "header")):
                os.mkdir(os.path.join(directory, "header"))
            headername = os.path.join(
                directory, "header", f"header{self.operation}.npy"
            )
            if Path(headername).is_file():
                logger.info(f"{headername} already exists")
            else:
                dictsaved = self.coords.get_attributes
                with open(headername, "wb") as file:
                    np.save(file, dictsaved)

    def find_ir(self, distance=1.0):
        r1 = distance
        if self.native_geometry in ("polar"):
            return find_nearest(r1, self.coords.Rmed)
        if self.native_geometry in ("spherical"):
            return find_nearest(r1, self.coords.rmed)

    def find_imid(self, altitude=None):
        if altitude is None:
            if self.native_geometry in ("cartesian", "polar"):
                altitude = 0.0
            if self.native_geometry in ("spherical"):
                altitude = np.pi / 2
        if self.native_geometry in ("cartesian", "polar"):
            return find_nearest(altitude, self.coords.zmed)
        if self.native_geometry in ("spherical"):
            return find_nearest(np.pi / 2 - altitude, self.coords.thetamed)

    def find_iphi(self, phi=0):
        if self.native_geometry in ("polar", "spherical"):
            return find_nearest(phi, self.coords.phimed) % self.coords.phimed.shape[0]

    def find_phip(self, planet_number: int = 0):
        init = Parameters(
            inifile=self.inifile, code=self.code, directory=self.directory
        )
        init.loadIniFile()
        init.loadPlanetFile(planet_number=planet_number)
        ind_on = find_nearest(init.tpl, init.vtk * self.on)
        return np.arctan2(init.ypl, init.xpl)[ind_on] % (2 * np.pi) - np.pi

    def latitudinal_projection(self, theta=None):
        operation = self.operation + "_latitudinal_projection"
        imid = self.find_imid()
        if self.native_geometry == "polar":
            ret_coords = Coordinates(
                self.native_geometry,
                self.coords.R,
                self.coords.phi,
                find_around(self.coords.z, self.coords.zmed[imid]),
            )
            # ret_coords = Coordinates(self.native_geometry, self.coords.R, find_around(self.coords.phi, self.coords.phimed[0]), self.coords.z)
            R = self.coords.Rmed
            z = self.coords.zmed
            integral = np.zeros((self.shape[0], self.shape[1]), dtype=">f4")
            # integral = np.zeros((self.shape[0],self.shape[2]), dtype='>f4')
            for i in range(self.shape[0]):
                km = find_nearest(z, z.min())
                kp = find_nearest(z, z.max())
                if theta is not None:
                    km = find_nearest(z, -R[i] * theta)
                    kp = find_nearest(z, R[i] * theta)
                integral[i, :] = np.sum(
                    (self.data[i, :, :] * np.ediff1d(self.coords.z)[None, :])[
                        :, km : kp + 1
                    ],
                    axis=1,
                    dtype="float64",
                )
                # integral[i,km] = -1
                # integral[i,kp] = 1
            ret_data = integral.reshape(self.shape[0], self.shape[1], 1)
            # ret_data = integral.reshape(self.shape[0],1,self.shape[2])
        if self.native_geometry == "spherical":
            ret_coords = Coordinates(
                self.native_geometry,
                self.coords.r,
                find_around(self.coords.theta, self.coords.thetamed[imid]),
                self.coords.phi,
            )
            km = find_nearest(self.coords.theta, self.coords.theta.min())
            kp = find_nearest(self.coords.theta, self.coords.theta.max())
            if theta is not None:
                km = find_nearest(self.coords.theta, np.pi / 2 + theta)
                kp = find_nearest(self.coords.theta, np.pi / 2 - theta)
            ret_data = (
                np.sum(
                    (
                        self.data
                        * self.coords.rmed[:, None, None]
                        * np.sin(self.coords.thetamed[None, :, None])
                        * np.ediff1d(self.coords.theta)[None, :, None]
                    )[:, :, km : kp + 1],
                    axis=1,
                    dtype="float64",
                )
            ).reshape(self.shape[0], 1, self.shape[2])
        return GasField(
            self.field,
            np.float32(ret_data),
            ret_coords,
            self.native_geometry,
            self.on,
            operation,
            inifile=self.inifile,
            code=self.code,
            directory=self.directory,
        )

    def vertical_projection(self, z=None):
        operation = self.operation + "_vertical_projection"
        imid = self.find_imid()
        if self.native_geometry == "cartesian":
            ret_coords = Coordinates(
                self.native_geometry,
                self.coords.x,
                self.coords.y,
                find_around(self.coords.z, self.coords.zmed[imid]),
            )
            km = find_nearest(self.coords.z, self.coords.z.min())
            kp = find_nearest(self.coords.z, self.coords.z.max())
            if z is not None:
                km = find_nearest(self.coords.z, -z)
                kp = find_nearest(self.coords.z, z)
            ret_data = (
                np.sum(
                    (self.data * np.ediff1d(self.coords.z))[:, :, km : kp + 1],
                    axis=2,
                    dtype="float64",
                )
            ).reshape(self.shape[0], self.shape[1], 1)
        if self.native_geometry == "polar":
            ret_coords = Coordinates(
                self.native_geometry,
                self.coords.R,
                self.coords.phi,
                find_around(self.coords.z, self.coords.zmed[imid]),
            )
            km = find_nearest(self.coords.z, self.coords.z.min())
            kp = find_nearest(self.coords.z, self.coords.z.max())
            if z is not None:
                km = find_nearest(self.coords.z, -z)
                kp = find_nearest(self.coords.z, z)
            ret_data = (
                np.sum(
                    (self.data * np.ediff1d(self.coords.z))[:, :, km : kp + 1],
                    axis=2,
                    dtype="float64",
                )
            ).reshape(self.shape[0], self.shape[1], 1)
        # if self.native_geometry=="spherical":
        #     ret_coords = Coordinates(self.native_geometry, self.coords.r, find_around(self.coords.theta, self.coords.thetamed[imid]), self.coords.phi)
        #     ret_data = (np.sum(self.data * self.coords.rmed[:,None,None] * np.sin(self.coords.thetamed[None,:,None]) * np.ediff1d(self.coords.theta)[None,:,None], axis=1, dtype="float64")).reshape(self.shape[0],1,self.shape[2])
        if self.native_geometry == "spherical":
            ret_coords = Coordinates(
                self.native_geometry,
                self.coords.r,
                find_around(self.coords.theta, self.coords.thetamed[imid]),
                self.coords.phi,
            )
            # ret_coords = Coordinates(self.native_geometry, self.coords.r, self.coords.theta, find_around(self.coords.phi, self.coords.phimed[0]))
            r = self.coords.rmed
            theta = self.coords.thetamed
            integral = np.zeros((self.shape[0], self.shape[2]), dtype=">f4")
            for i in range(self.shape[0]):
                kp = find_nearest(theta, theta.max())
                km = find_nearest(theta, theta.min())
                if z is not None:
                    kp = find_nearest(theta, np.arctan2(r[i], -z))
                    km = find_nearest(theta, np.arctan2(r[i], z))
                # integral[i,:] = np.sum((((self.data[i,1:,:]*np.sin(theta[1:,None])+self.data[i,:-1,:]*np.sin(theta[:-1,None]))/2)*r[i]*np.ediff1d(theta)[:,None])[km:kp+1,:], axis=0, dtype="float64")
                integral[i, :] = np.sum(
                    (
                        (self.data[i, :, :] * np.sin(theta[:, None]))
                        * r[i]
                        * np.ediff1d(self.coords.theta)[:, None]
                    )[km : kp + 1, :],
                    axis=0,
                    dtype="float64",
                )
                # integral[i,km] = -1
                # integral[i,kp] = 1
            ret_data = integral.reshape(self.shape[0], 1, self.shape[2])
            # ret_data = integral.reshape(self.shape[0],self.shape[1],1)
        return GasField(
            self.field,
            np.float32(ret_data),
            ret_coords,
            self.native_geometry,
            self.on,
            operation,
            inifile=self.inifile,
            code=self.code,
            directory=self.directory,
        )

    def vertical_at_midplane(self):
        # self.field = r"%s$_{\rm mid}$" % self.field
        operation = self.operation + "_vertical_at_midplane"
        imid = self.find_imid()
        if self.native_geometry == "cartesian":
            # find_around looks around the 2 coords values that surround coordmed at imid
            ret_coords = Coordinates(
                self.native_geometry,
                self.coords.x,
                self.coords.y,
                find_around(self.coords.z, self.coords.zmed[imid]),
            )
            # ret_coords = Coordinates(self.native_geometry, self.coords.x, self.coords.y, self.coords.z[imid:imid+2])
            ret_data = self.data[:, :, imid].reshape(self.shape[0], self.shape[1], 1)
            # do geometry conversion!!! -> chainer la conversion (une fois que reduction de dimension -> conversion puis plot egalement chainable)
        if self.native_geometry == "polar":
            ret_coords = Coordinates(
                self.native_geometry,
                self.coords.R,
                self.coords.phi,
                find_around(self.coords.z, self.coords.zmed[imid]),
            )
            # ret_coords = Coordinates(self.native_geometry, self.coords.R, self.coords.phi, self.coords.z[imid:imid+2])
            ret_data = self.data[:, :, imid].reshape(self.shape[0], self.shape[1], 1)
        if self.native_geometry == "spherical":
            ret_coords = Coordinates(
                self.native_geometry,
                self.coords.r,
                find_around(self.coords.theta, self.coords.thetamed[imid]),
                self.coords.phi,
            )
            # ret_coords = Coordinates(self.native_geometry, self.coords.r, self.coords.theta[imid:imid+2], self.coords.phi)
            ret_data = self.data[:, imid, :].reshape(self.shape[0], 1, self.shape[2])
        return GasField(
            self.field,
            ret_data,
            ret_coords,
            self.native_geometry,
            self.on,
            operation,
            inifile=self.inifile,
            code=self.code,
            directory=self.directory,
        )

    def latitudinal_at_theta(self, theta=None, name_operation=None):
        logger.info("latitudinal_at_theta TO BE TESTED")
        if theta is None:
            theta = 0
        if name_operation is None:
            operation = self.operation + f"_latitudinal_at_theta{np.pi/2-theta}"
        else:
            operation = self.operation + "_" + name_operation
        imid = self.find_imid(altitude=theta)
        if self.native_geometry == "polar":
            data_at_theta = np.zeros((self.shape[0], self.shape[1]), dtype=">f4")
            for i in range(self.shape[0]):
                if np.sign(theta) >= 0:
                    if (
                        find_nearest(self.coords.z, theta * self.coords.R[i])
                        < self.shape[2]
                    ):
                        # print(i,find_nearest(rhoon.x,rhoon.x[i]),find_nearest(rhoon.z,4*0.05*rhoon.x[i]))
                        # rvz[find_nearest(rhoon.x,rhoon.x[i]),:,find_nearest(rhoon.z,4*0.05*rhoon.x[i])] = 1
                        data_at_theta[i, :] = self.data[
                            i, :, find_nearest(self.coords.z, theta * self.coords.R[i])
                        ]
                    else:
                        data_at_theta[i, :] = np.nan
                else:
                    if find_nearest(self.coords.z, theta * self.coords.R[i]) > 0:
                        data_at_theta[i, :] = self.data[
                            i, :, find_nearest(self.coords.z, theta * self.coords.R[i])
                        ]
                    else:
                        data_at_theta[i, :] = np.nan
            ret_data = data_at_theta.reshape(self.shape[0], self.shape[1], 1)
            ret_coords = Coordinates(
                self.native_geometry,
                self.coords.R,
                self.coords.phi,
                find_around(self.coords.z, self.coords.zmed[imid]),
            )
        if self.native_geometry == "spherical":
            ret_coords = Coordinates(
                self.native_geometry,
                self.coords.r,
                find_around(self.coords.theta, self.coords.thetamed[imid]),
                self.coords.phi,
            )
            ret_data = self.data[
                :, find_nearest(self.coords.theta, np.pi / 2 - theta), :
            ].reshape(self.shape[0], 1, self.shape[2])
        return GasField(
            self.field,
            ret_data,
            ret_coords,
            self.native_geometry,
            self.on,
            operation,
            inifile=self.inifile,
            code=self.code,
            directory=self.directory,
        )

    def vertical_at_z(self, z=None, name_operation=None):
        logger.info("vertical_at_z TO BE TESTED")
        # self.field = r"%s$_{\rm mid}$" % self.field
        if z is None:
            z = 0
        if name_operation is None:
            operation = self.operation + f"_vertical_at_z{z}"
        else:
            operation = self.operation + "_" + name_operation

        imid = self.find_imid(altitude=z)
        if self.native_geometry == "cartesian":
            ret_coords = Coordinates(
                self.native_geometry,
                self.coords.x,
                self.coords.y,
                find_around(self.coords.z, self.coords.zmed[imid]),
            )
            ret_data = self.data[:, :, find_nearest(self.coords.z, z)].reshape(
                self.shape[0], self.shape[1], 1
            )
        if self.native_geometry == "polar":
            ret_coords = Coordinates(
                self.native_geometry,
                self.coords.R,
                self.coords.phi,
                find_around(self.coords.z, self.coords.zmed[imid]),
            )
            ret_data = self.data[:, :, find_nearest(self.coords.z, z)].reshape(
                self.shape[0], self.shape[1], 1
            )
        if self.native_geometry == "spherical":
            data_at_z = np.zeros((self.shape[0], self.shape[2]), dtype=">f4")
            for i in range(self.shape[0]):
                if np.sign(z) >= 0:
                    if (
                        find_nearest(self.coords.theta, np.arctan2(self.coords.r[i], z))
                        < self.shape[2]
                    ):
                        # print(i,find_nearest(rhoon.x,rhoon.x[i]),find_nearest(rhoon.z,4*0.05*rhoon.x[i]))
                        # rvz[find_nearest(rhoon.x,rhoon.x[i]),:,find_nearest(rhoon.z,4*0.05*rhoon.x[i])] = 1
                        data_at_z[i, :] = self.data[
                            i,
                            find_nearest(
                                self.coords.theta, np.arctan2(self.coords.r[i], z)
                            ),
                            :,
                        ]
                    else:
                        data_at_z[i, :] = np.nan
                else:
                    if (
                        find_nearest(self.coords.theta, np.arctan2(self.coords.r[i], z))
                        > 0
                    ):
                        data_at_z[i, :] = self.data[
                            i,
                            find_nearest(
                                self.coords.theta, np.arctan2(self.coords.r[i], z)
                            ),
                            :,
                        ]
                    else:
                        data_at_z[i, :] = np.nan
            ret_data = data_at_z.reshape(self.shape[0], 1, self.shape[2])
            ret_coords = Coordinates(
                self.native_geometry,
                self.coords.r,
                find_around(self.coords.theta, self.coords.thetamed[imid]),
                self.coords.phi,
            )
        return GasField(
            self.field,
            ret_data,
            ret_coords,
            self.native_geometry,
            self.on,
            operation,
            inifile=self.inifile,
            code=self.code,
            directory=self.directory,
        )

    def azimuthal_at_phi(self, phi=None):
        if phi is None:
            phi = 0.0
        # self.field = r"%s ($\phi_P$)" % self.field
        operation = self.operation + f"_azimuthal_at_phi{phi}"
        # operation = self.operation + f"_azimuthal_at_phivortex"
        iphi = self.find_iphi(phi=phi)
        if self.native_geometry == "cartesian":
            raise NotImplementedError(
                f"geometry flag '{self._native_geometry}' not implemented yet for azimuthal_at_phi"
            )
        if self.native_geometry == "polar":
            ret_coords = Coordinates(
                self.native_geometry,
                self.coords.R,
                find_around(self.coords.phi, self.coords.phimed[iphi]),
                self.coords.z,
            )
            ret_data = self.data[:, iphi, :].reshape(self.shape[0], 1, self.shape[2])
        if self.native_geometry == "spherical":
            ret_coords = Coordinates(
                self.native_geometry,
                self.coords.r,
                self.coords.theta,
                find_around(self.coords.phi, self.coords.phimed[iphi]),
            )
            ret_data = self.data[:, :, iphi].reshape(self.shape[0], self.shape[1], 1)
        return GasField(
            self.field,
            ret_data,
            ret_coords,
            self.native_geometry,
            self.on,
            operation,
            inifile=self.inifile,
            code=self.code,
            directory=self.directory,
        )

    def azimuthal_at_planet(self, planet_number: int = 0):
        operation = self.operation + "_azimuthal_at_planet"
        phip = self.find_phip(planet_number=planet_number)
        aziphip = self.azimuthal_at_phi(phi=phip)
        return GasField(
            self.field,
            aziphip.data,
            aziphip.coords,
            self.native_geometry,
            self.on,
            operation,
            inifile=self.inifile,
            code=self.code,
            directory=self.directory,
        )

    def azimuthal_average(self):
        # self.field = r"$\langle$%s$\rangle$" % self.field
        operation = self.operation + "_azimuthal_average"
        iphi = self.find_iphi(phi=0)
        if self.native_geometry == "cartesian":
            raise NotImplementedError(
                f"geometry flag '{self._native_geometry}' not implemented yet for azimuthal_average"
            )
        if self.native_geometry == "polar":
            ret_coords = Coordinates(
                self.native_geometry,
                self.coords.R,
                find_around(self.coords.phi, self.coords.phimed[iphi]),
                self.coords.z,
            )
            ret_data = np.mean(self.data, axis=1, dtype="float64").reshape(
                self.shape[0], 1, self.shape[2]
            )
        if self.native_geometry == "spherical":
            ret_coords = Coordinates(
                self.native_geometry,
                self.coords.r,
                self.coords.theta,
                find_around(self.coords.phi, self.coords.phimed[iphi]),
            )
            ret_data = np.mean(self.data, axis=2, dtype="float64").reshape(
                self.shape[0], self.shape[1], 1
            )
        return GasField(
            self.field,
            np.float32(ret_data),
            ret_coords,
            self.native_geometry,
            self.on,
            operation,
            inifile=self.inifile,
            code=self.code,
            directory=self.directory,
        )

    def radial_at_r(self, distance=None):
        if distance is None:
            distance = 1.0
        operation = self.operation + f"_radial_at_r{distance}"
        ir1 = self.find_ir(distance=distance)
        if self.native_geometry == "cartesian":
            raise NotImplementedError(
                f"geometry flag '{self._native_geometry}' not implemented yet for radial_at_r"
            )
        if self.native_geometry == "polar":
            # self.field = r"%s (R=1)" % self.field
            ret_coords = Coordinates(
                self.native_geometry,
                find_around(self.coords.R, self.coords.Rmed[ir1]),
                self.coords.phi,
                self.coords.z,
            )
        if self.native_geometry == "spherical":
            # self.field = r"%s (r=1)" % self.field
            ret_coords = Coordinates(
                self.native_geometry,
                find_around(self.coords.r, self.coords.rmed[ir1]),
                self.coords.theta,
                self.coords.phi,
            )
        ret_data = self.data[ir1, :, :].reshape(1, self.shape[1], self.shape[2])
        return GasField(
            self.field,
            ret_data,
            ret_coords,
            self.native_geometry,
            self.on,
            operation,
            inifile=self.inifile,
            code=self.code,
            directory=self.directory,
        )

    def diff(self, on_2):
        ds_2 = GasDataSet(
            on_2,
            geometry=self.native_geometry,
            inifile=self.inifile,
            code=self.code,
            directory=self.directory,
        )
        if self.operation != "":
            raise KeyError(
                "For now, diff should only be applied on the initial Field cube."
            )
        # self.operation += "diff"
        ret_data = (self.data - ds_2[self.field].data) / ds_2[self.field].data
        ret_coords = self.coords
        # self.field = r"$\frac{%s - %s_0}{%s_0}$" % (self.field, self.field, self.field)
        return GasField(
            self.field,
            ret_data,
            ret_coords,
            self.native_geometry,
            self.on,
            self.operation,
            inifile=self.inifile,
            code=self.code,
            directory=self.directory,
        )

    def rotate(self, planet_corotation: Optional[int] = None):
        operation = self.operation
        if self.shape.count(1) != 1:
            raise ValueError("data has to be 2D in order to rotate the data.")
        if planet_corotation is not None:  # and self.coords.phi.shape[0]!=1:
            phip = self.find_phip(planet_number=planet_corotation)
            phicoord = self.coords.phi - phip  # - np.pi
            ipi = find_nearest(phicoord, 0)
            if abs(0 - phicoord[ipi]) > abs(np.ediff1d(find_around(phicoord, 0))[0]):
                ipi = find_nearest(phicoord, 2 * np.pi)
            if self.native_geometry == "polar":
                ret_data = np.roll(
                    self.data, -ipi - self.coords.phi.shape[0] // 2, axis=1
                )
                ret_coords = Coordinates(
                    self.native_geometry, self.coords.R, self.coords.phi, self.coords.z
                )
            elif self.native_geometry == "spherical":
                ret_data = np.roll(
                    self.data, -ipi - self.coords.phi.shape[0] // 2, axis=2
                )
                ret_coords = Coordinates(
                    self.native_geometry,
                    self.coords.r,
                    self.coords.theta,
                    self.coords.phi,
                )
            else:
                raise NotImplementedError(
                    f"geometry flag '{self.native_geometry}' not implemented yet if planet_corotation"
                )
        else:
            ret_data = self.data
            if self.native_geometry == "polar":
                ret_coords = Coordinates(
                    self.native_geometry, self.coords.R, self.coords.phi, self.coords.z
                )
            elif self.native_geometry == "spherical":
                ret_coords = Coordinates(
                    self.native_geometry,
                    self.coords.r,
                    self.coords.theta,
                    self.coords.phi,
                )

        return GasField(
            self.field,
            ret_data,
            ret_coords,
            self.native_geometry,
            self.on,
            operation,
            inifile=self.inifile,
            code=self.code,
            directory=self.directory,
        )


class GasDataSet:
    """Idefix dataset class that contains everything in the .vtk file

    Attributes
    ==========
    on : int corresponding to the output number.
    dict: dictionary[str, array] that contains all the fields
    coords : Coordinates of the data.
        Edge of the cells.
    coordsmed : Coordinates of the data.
        Center of the cells.
    """

    # def __init__(self, on:int, *, code:Optional[str]=None, directory:str=""):
    def __init__(
        self,
        on: int,
        *,
        inifile: str = "",
        code: str = "",
        geometry: str = "unknown",
        directory: str = "",
    ):
        self.on = on
        self.params = Parameters(inifile=inifile, code=code, directory=directory)
        self._read = self.params.loadSimuFile(self.on, geometry=geometry, cell="edges")
        self.native_geometry = self._read.geometry
        self.dict = self._read.data
        self.coords = Coordinates(
            self.native_geometry, self._read.x1, self._read.x2, self._read.x3
        )
        for key in self.dict:
            self.dict[key] = GasField(
                key,
                self.dict[key],
                self.coords,
                self.native_geometry,
                self.on,
                "",
                inifile=inifile,
                code=code,
                directory=directory,
            )

    def __getitem__(self, key):
        if key in self.dict:
            return self.dict[key]
        else:
            raise KeyError

    def keys(self):
        """
        Returns
        =======
        keys of the dict
        """
        return self.dict.keys()

    def values(self):
        """
        Returns
        =======
        values of the dict
        """
        return self.dict.values()

    def items(self):
        """
        Returns
        =======
        items of the dict
        """
        return self.dict.items()


def from_file(*, field: str, filename: str, on: int, directory=""):
    repout = field.lower()
    headername = os.path.join(directory, "header", f"header_{filename}.npy")
    with open(headername, "rb") as file:
        dict_coords = np.load(file, allow_pickle=True).item()

    geometry, coord0, coord1, coord2 = dict_coords.values()
    ret_coords = Coordinates(geometry, coord0, coord1, coord2)

    fileout = os.path.join(directory, repout, f"_{filename}_{field}.{on:04d}.npy")
    with open(fileout, "rb") as file:
        ret_data = np.load(file, allow_pickle=True)

    return GasField(
        field,
        ret_data,
        ret_coords,
        geometry,
        on,
        operation=filename,
        directory=directory,
    )


def from_data(
    *,
    field: str,
    data: np.ndarray,
    coords: Coordinates,
    on: int,
    operation: str,
):
    ret_data = data
    ret_coords = coords
    geometry = coords.geometry
    return GasField(
        field,
        ret_data,
        ret_coords,
        geometry,
        on,
        operation=operation,
    )


def temporal_all(
    field: str,
    operation: str,
    onall,
    directory: str = "",
    planet_corotation: Optional[int] = None,
):
    datasum = 0
    don = len(onall)
    for on in sorted(onall):
        datafield = from_file(
            field=field, filename=operation, on=on, directory=directory
        )
        datafield_rot = datafield.rotate(planet_corotation=planet_corotation)
        datasum += datafield_rot.data
    datafieldsum = from_data(
        field="".join([field, "T_ALL"]),
        data=np.array(datasum / don),
        coords=datafield.coords,
        on=0,
        operation="_" + operation,
    )
    return datafieldsum


def temporal(
    field: str,
    operation: str,
    onbeg: int,
    *,
    onend: Optional[int] = None,
    directory: str = "",
    planet_corotation: Optional[int] = None,
):
    datasum = 0
    if onend is None:
        datafield = from_file(
            field=field, filename=operation, on=onbeg, directory=directory
        )
        datafield_rot = datafield.rotate(planet_corotation=planet_corotation)
        datafieldsum = from_data(
            field=field,
            data=datafield_rot.data,
            coords=datafield_rot.coords,
            on=onbeg,
            operation="_" + operation,
        )
        return datafieldsum
    else:
        don = onend - onbeg
        for on in range(onbeg, onend + 1):
            datafield = from_file(
                field=field, filename=operation, on=on, directory=directory
            )
            datafield_rot = datafield.rotate(planet_corotation=planet_corotation)
            datasum += datafield_rot.data
        datafieldsum = from_data(
            field="".join([field, f"T_{onbeg}_{onend}"]),
            data=np.array(datasum / don),
            coords=datafield.coords,
            on=onend,
            operation="_" + operation,
        )
        return datafieldsum
