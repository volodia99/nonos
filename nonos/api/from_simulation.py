import glob
import os
import re
from pathlib import Path

import inifix
import numpy as np

_INIFILES_LOOKUP_TABLE = {
    "idefix.ini": "idefix",
    "pluto.ini": "pluto",
    "variables.par": "fargo3d",
}


class Parameters:
    def __init__(
        self,
        *,
        inifile: str = "",
        code: str = "",
        directory: str = "",
    ):
        self.directory = directory
        self.paramfile = inifile
        self.code = code
        if (self.code, self.paramfile).count("") == 0:
            found = Path(self.directory).joinpath(self.paramfile).is_file()
            if not found:
                raise FileNotFoundError(f"{self.paramfile} not found.")
        elif (self.code, self.paramfile).count("") == 2:
            found_dict = {
                paramfile: Path(self.directory).joinpath(paramfile).is_file()
                for paramfile in _INIFILES_LOOKUP_TABLE
            }
            nfound = sum(list(found_dict.values()))
            if nfound == 0:
                raise FileNotFoundError(
                    f"{', '.join(_INIFILES_LOOKUP_TABLE.keys())} not found."
                )
            elif nfound > 1:
                raise RuntimeError("found more than one possible ini file.")
            self.paramfile = list(_INIFILES_LOOKUP_TABLE.keys())[
                list(found_dict.values()).index(True)
            ]
            self.code = _INIFILES_LOOKUP_TABLE[self.paramfile]
        elif (self.code, self.paramfile).count("") == 1:
            raise ValueError("both inifile and code have to be given.")

    def loadIniFile(self):
        self.inifile = inifix.load(os.path.join(self.directory, self.paramfile))
        if self.code == "idefix":
            self.vtk = self.inifile["Output"]["vtk"]
        elif self.code == "pluto":
            self.vtk = self.inifile["Static Grid Output"]["vtk"][0]
        elif self.code == "fargo3d":
            self.vtk = self.inifile["NINTERM"] * self.inifile["DT"]
        elif self.code == "fargo-adsg":
            self.vtk = self.inifile["Ninterm"] * self.inifile["DT"]

    def loadPlanetFile(self, *, planet_number: int = 0):
        planet_file = f"planet{planet_number}.dat"

        if self.code in ("idefix", "fargo3d", "fargo-adsg"):
            if Path(self.directory).joinpath(planet_file).is_file():
                columns = np.loadtxt(os.path.join(self.directory, planet_file)).T
                self.qpl = columns[7]
                self.dpl = np.sqrt(np.sum(columns[1:4] ** 2, axis=0))
                self.xpl = columns[1]
                self.ypl = columns[2]
                self.tpl = columns[8]
            else:
                raise FileNotFoundError(f"{planet_file} not found")
        else:
            raise NotImplementedError(
                f"{planet_file} not found for {self.code}. For now, you can't rotate the grid with the planet."
            )

    def countSimuFiles(self):
        if self.code in ("fargo3d", "fargo-adsg"):
            self.data_files = [
                fn
                for fn in glob.glob1(self.directory, "gasdens*.dat")
                if re.match(r"gasdens\d+.dat", fn)
            ]
        elif self.code in ("idefix", "pluto"):
            self.data_files = list(glob.glob1(self.directory, "data.*.vtk"))
        else:
            raise RuntimeError("Unknown file format")

    def loadSimuFile(self, on: int, *, geometry: str = "unknown", cell: str = "edges"):
        codeReadFormat = CodeReadFormat()
        if self.code == "fargo3d":
            return codeReadFormat.fargo3dReadDat(
                on, directory=self.directory, inifile=self.paramfile
            )
        elif self.code == "fargo-adsg":
            return codeReadFormat.fargoAdsgReadDat(
                on, directory=self.directory
            )  # , inifile=self.paramfile)
        elif self.code in ("idefix", "pluto"):
            dataVTK = os.path.join(self.directory, f"data.{on:04d}.vtk")
            return codeReadFormat.idfxReadVTK(dataVTK, geometry=geometry, cell=cell)
        else:
            raise ValueError(f"For now, can't read files from {self.code} simulations.")


class DataStructure:
    """
    Class that helps create the datastructure
    in the idfxReadVTK and fargo3dReadDat functions
    """

    __slots__ = (
        "data",
        "geometry",
        "t",
        "periodicity",
        "n1",
        "n2",
        "n3",
        "x1",
        "x2",
        "x3",
    )
    pass


class CodeReadFormat:
    def idfxReadVTK(
        self, filename, *, geometry="unknown", cell="edges", computedata=True
    ):
        """
        Adapted from Geoffroy Lesur
        Function that reads a vtk file in polar coordinates
        """
        nfound = len(glob.glob(filename))
        if nfound != 1:
            raise FileNotFoundError("In idfxReadVTK: %s not found." % filename)

        fid = open(filename, "rb")

        # define our datastructure
        V = DataStructure()

        # raw data which will be read from the file
        V.data = {}

        # initialize geometry
        if geometry not in ("unknown", "cartesian", "polar", "spherical"):
            raise ValueError(f"Unknown geometry value: {geometry!r}")
        V.geometry = geometry

        # datatype we read
        dt = np.dtype(">f")  # Big endian single precision floats
        dint = np.dtype(">i4")  # Big endian integer

        s = fid.readline()  # VTK DataFile Version x.x
        s = fid.readline()  # Comments

        s = fid.readline()  # BINARY
        s = fid.readline()  # DATASET RECTILINEAR_GRID or STRUCTURED_GRID
        slist = s.split()

        s = fid.readline()  # DIMENSIONS NX NY NZ
        slist = s.split()
        entry = str(slist[0], "utf-8")
        if entry == "FIELD":
            nfield = int(slist[2])
            for _field in range(nfield):
                s = fid.readline()
                slist = s.split()
                entry = str(slist[0], "utf-8")
                if entry == "TIME":
                    V.t = np.fromfile(fid, dt, 1)
                elif entry == "GEOMETRY":
                    g = np.fromfile(fid, dint, 1)
                    if g == 0:
                        thisgeometry = "cartesian"
                    elif g == 1:
                        thisgeometry = "polar"
                    elif g == 2:
                        thisgeometry = "spherical"
                    elif g == 3:
                        thisgeometry = "cylindrical"
                    else:
                        raise ValueError(
                            f"Unknown value for GEOMETRY flag ('{g}') was found in the VTK file."
                        )

                    if V.geometry != "unknown":
                        # We already have a proposed geometry, check that what is read from the file matches
                        if thisgeometry != V.geometry:
                            raise ValueError(
                                f"geometry argument ('{V.geometry}') is inconsistent with GEOMETRY flag from the VTK file ('{thisgeometry}')"
                            )
                    V.geometry = thisgeometry
                elif entry == "PERIODICITY":
                    periodicity = np.fromfile(fid, dint, 3).astype(bool)
                    V.periodicity = tuple(periodicity)
                else:
                    raise ValueError(f"Received unknown field: '{entry}'.")

                s = fid.readline()  # extra linefeed

            # finished reading the field entry
            # read next line
            s = fid.readline()  # DIMENSIONS...

        if V.geometry == "unknown":
            raise RuntimeError(
                "Geometry couldn't be determined from data. "
                "Try to set the geometry keyword argument explicitely."
            )

        slist = s.split()  # DIMENSIONS....
        V.n1 = int(slist[1])
        V.n2 = int(slist[2])
        V.n3 = int(slist[3])

        if V.geometry == "cartesian":
            # CARTESIAN geometry
            s = fid.readline()  # X_COORDINATES NX float
            inipos = (
                fid.tell()
            )  # we store the file pointer position before computing points
            # logger.debug("loading the X-grid cells: %d" % V.n1)
            x = np.memmap(
                fid, mode="r", dtype=dt, offset=inipos, shape=V.n1
            )  # some smart memory efficient way to store the array
            newpos = (
                np.float32().nbytes * 1 * V.n1 + inipos
            )  # we calculate the offset that we would expect normally with a np.fromfile
            fid.seek(
                newpos, os.SEEK_SET
            )  # we set the file pointer position to this offset
            s = fid.readline()  # Extra line feed added by idefix

            s = fid.readline()  # Y_COORDINATES NY float
            inipos = (
                fid.tell()
            )  # we store the file pointer position before computing points
            # logger.debug("loading the Y-grid cells: %d" % V.n2)
            y = np.memmap(
                fid, mode="r", dtype=dt, offset=inipos, shape=V.n2
            )  # some smart memory efficient way to store the array
            newpos = (
                np.float32().nbytes * 1 * V.n2 + inipos
            )  # we calculate the offset that we would expect normally with a np.fromfile
            fid.seek(
                newpos, os.SEEK_SET
            )  # we set the file pointer position to this offset
            s = fid.readline()  # Extra line feed added by idefix

            s = fid.readline()  # Z_COORDINATES NZ float
            inipos = (
                fid.tell()
            )  # we store the file pointer position before computing points
            # logger.debug("loading the Z-grid cells: %d" % V.n3)
            z = np.memmap(
                fid, mode="r", dtype=dt, offset=inipos, shape=V.n3
            )  # some smart memory efficient way to store the array
            newpos = (
                np.float32().nbytes * 1 * V.n3 + inipos
            )  # we calculate the offset that we would expect normally with a np.fromfile
            fid.seek(
                newpos, os.SEEK_SET
            )  # we set the file pointer position to this offset
            s = fid.readline()  # Extra line feed added by idefix

            s = fid.readline()  # POINT_DATA NXNYNZ

            slist = s.split()
            point_type = str(slist[0], "utf-8")
            npoints = int(slist[1])
            s = fid.readline()  # EXTRA LINE FEED

            if point_type == "CELL_DATA" or cell == "centers":
                # The file contains face coordinates, so we extrapolate to get the cell center coordinates.
                if V.n1 > 1:
                    V.n1 = V.n1 - 1
                    V.x1 = 0.5 * (x[1:] + x[:-1])
                else:
                    V.x1 = x
                if V.n2 > 1:
                    V.n2 = V.n2 - 1
                    V.x2 = 0.5 * (y[1:] + y[:-1])
                else:
                    V.x2 = y
                if V.n3 > 1:
                    V.n3 = V.n3 - 1
                    V.x3 = 0.5 * (z[1:] + z[:-1])
                else:
                    V.x3 = z
            elif point_type == "POINT_DATA" or cell == "edges":
                V.x1 = x
                V.x2 = y
                V.x3 = z
                if V.n1 > 1:
                    V.n1 = V.n1 - 1
                    V.x1 = x
                else:
                    V.x1 = x
                if V.n2 > 1:
                    V.n2 = V.n2 - 1
                    V.x2 = y
                else:
                    V.x2 = y
                if V.n3 > 1:
                    V.n3 = V.n3 - 1
                    V.x3 = z
                else:
                    V.x3 = z

            if V.n1 * V.n2 * V.n3 != npoints:
                raise ValueError(
                    "In idfxReadVTK: Grid size (%d) incompatible with number of points (%d) in the data set"
                    % (V.n1 * V.n2 * V.n3, npoints)
                )

        else:
            # POLAR or SPHERICAL coordinates
            if V.n3 == 1:
                is2d = 1
            else:
                is2d = 0

            s = fid.readline()  # POINTS NXNYNZ float
            slist = s.split()
            npoints = int(slist[1])

            inipos = (
                fid.tell()
            )  # we store the file pointer position before computing points
            # print(inipos)
            # points = np.fromfile(fid, dt, 3 * npoints)
            # logger.debug("loading the grid cells: (%d,%d,%d)." % (V.n1, V.n2, V.n3))
            points = np.memmap(
                fid, mode="r", dtype=dt, offset=inipos, shape=3 * npoints
            )  # some smart memory efficient way to store the array
            # print(fid.tell())
            newpos = (
                np.float32().nbytes * 3 * npoints + inipos
            )  # we calculate the offset that we would expect normally with a np.fromfile
            fid.seek(
                newpos, os.SEEK_SET
            )  # we set the file pointer position to this offset
            # print(fid.tell())
            s = fid.readline()  # EXTRA LINE FEED

            # V.points=points
            if V.n1 * V.n2 * V.n3 != npoints:
                raise ValueError(
                    "In idfxReadVTK: Grid size (%d) incompatible with number of points (%d) in the data set"
                    % (V.n1 * V.n2 * V.n3, npoints)
                )

            # Reconstruct the polar coordinate system
            x1d = points[::3]
            y1d = points[1::3]
            z1d = points[2::3]

            xcart = np.transpose(x1d.reshape(V.n3, V.n2, V.n1))
            ycart = np.transpose(y1d.reshape(V.n3, V.n2, V.n1))
            zcart = np.transpose(z1d.reshape(V.n3, V.n2, V.n1))

            # Reconstruct the polar coordinate system
            if V.geometry == "polar":

                r = np.sqrt(xcart[:, 0, 0] ** 2 + ycart[:, 0, 0] ** 2)
                theta = np.unwrap(np.arctan2(ycart[0, :, 0], xcart[0, :, 0]))
                z = zcart[0, 0, :]

                s = fid.readline()  # CELL_DATA (NX-1)(NY-1)(NZ-1)
                slist = s.split()
                data_type = str(slist[0], "utf-8")
                if data_type != "CELL_DATA":
                    fid.close()
                    raise ValueError(
                        "In idfxReadVTK: this routine expect 'CELL DATA' as produced by PLUTO, not '%s'."
                        % data_type
                    )
                s = fid.readline()  # Line feed

                if cell == "edges":
                    if V.n1 > 1:
                        V.n1 = V.n1 - 1
                        V.x1 = r
                    else:
                        V.x1 = r
                    if V.n2 > 1:
                        V.n2 = V.n2 - 1
                        V.x2 = theta
                    else:
                        V.x2 = theta
                    if V.n3 > 1:
                        V.n3 = V.n3 - 1
                        V.x3 = z
                    else:
                        V.x3 = z

                # Perform averaging on coordinate system to get cell centers
                # The file contains face coordinates, so we extrapolate to get the cell center coordinates.
                elif cell == "centers":
                    if V.n1 > 1:
                        V.n1 = V.n1 - 1
                        V.x1 = 0.5 * (r[1:] + r[:-1])
                    else:
                        V.x1 = r
                    if V.n2 > 1:
                        V.n2 = V.n2 - 1
                        V.x2 = (0.5 * (theta[1:] + theta[:-1]) + np.pi) % (
                            2.0 * np.pi
                        ) - np.pi
                    else:
                        V.x2 = theta
                    if V.n3 > 1:
                        V.n3 = V.n3 - 1
                        V.x3 = 0.5 * (z[1:] + z[:-1])
                    else:
                        V.x3 = z

            # Reconstruct the spherical coordinate system
            if V.geometry == "spherical":
                if is2d:
                    r = np.sqrt(xcart[:, 0, 0] ** 2 + ycart[:, 0, 0] ** 2)
                    phi = np.unwrap(
                        np.arctan2(zcart[0, V.n2 // 2, :], xcart[0, V.n2 // 2, :])
                    )
                    theta = np.arccos(
                        ycart[0, :, 0]
                        / np.sqrt(xcart[0, :, 0] ** 2 + ycart[0, :, 0] ** 2)
                    )
                else:
                    r = np.sqrt(
                        xcart[:, 0, 0] ** 2 + ycart[:, 0, 0] ** 2 + zcart[:, 0, 0] ** 2
                    )
                    phi = np.unwrap(
                        np.arctan2(
                            ycart[V.n1 // 2, V.n2 // 2, :],
                            xcart[V.n1 // 2, V.n2 // 2, :],
                        )
                    )
                    theta = np.arccos(
                        zcart[0, :, 0]
                        / np.sqrt(
                            xcart[0, :, 0] ** 2
                            + ycart[0, :, 0] ** 2
                            + zcart[0, :, 0] ** 2
                        )
                    )

                s = fid.readline()  # CELL_DATA (NX-1)(NY-1)(NZ-1)
                slist = s.split()
                data_type = str(slist[0], "utf-8")
                if data_type != "CELL_DATA":
                    fid.close()
                    raise ValueError(
                        "In idfxReadVTK: this routine expect 'CELL DATA' as produced by PLUTO, not '%s'."
                        % data_type
                    )
                s = fid.readline()  # Line feed

                if cell == "edges":
                    if V.n1 > 1:
                        V.n1 = V.n1 - 1
                        V.x1 = r
                    else:
                        V.x1 = r
                    if V.n2 > 1:
                        V.n2 = V.n2 - 1
                        V.x2 = theta
                    else:
                        V.x2 = theta
                    if V.n3 > 1:
                        V.n3 = V.n3 - 1
                        V.x3 = phi
                    else:
                        V.x3 = phi

                # Perform averaging on coordinate system to get cell centers
                # The file contains face coordinates, so we extrapolate to get the cell center coordinates.
                elif cell == "centers":
                    if V.n1 > 1:
                        V.n1 = V.n1 - 1
                        V.x1 = 0.5 * (r[1:] + r[:-1])
                    else:
                        V.x1 = r
                    if V.n2 > 1:
                        V.n2 = V.n2 - 1
                        V.x2 = 0.5 * (theta[1:] + theta[:-1])
                    else:
                        V.x2 = theta
                    if V.n3 > 1:
                        V.n3 = V.n3 - 1
                        V.x3 = 0.5 * (phi[1:] + phi[:-1])
                    else:
                        V.x3 = phi

        if computedata:
            # logger.debug("loading the data arrays:")
            while 1:
                s = (
                    fid.readline()
                )  # SCALARS/VECTORS name data_type (ex: SCALARS imagedata unsigned_char)
                # print repr(s)
                if len(s) < 2:  # leave if end of file
                    break
                slist = s.split()
                datatype = str(slist[0], "utf-8")
                varname = str(slist[1], "utf-8").upper()
                if datatype == "SCALARS":
                    fid.readline()  # LOOKUP TABLE

                    inipos = (
                        fid.tell()
                    )  # we store the file pointer position before computing points
                    # array = np.fromfile(fid, dt, V.n1 * V.n2 * V.n3).reshape(V.n3, V.n2, V.n1)
                    array = np.memmap(
                        fid, mode="r", dtype=dt, offset=inipos, shape=V.n1 * V.n2 * V.n3
                    ).reshape(
                        V.n3, V.n2, V.n1
                    )  # some smart memory efficient way to store the array
                    newpos = (
                        np.float32().nbytes * V.n1 * V.n2 * V.n3 + inipos
                    )  # we calculate the offset that we would expect normally with a np.fromfile
                    fid.seek(
                        newpos, os.SEEK_SET
                    )  # we set the file pointer position to this offset

                    V.data[varname] = np.transpose(array)
                elif datatype == "VECTORS":
                    inipos = (
                        fid.tell()
                    )  # we store the file pointer position before computing points
                    Q = np.memmap(
                        fid, mode="r", dtype=dt, offset=inipos, shape=V.n1 * V.n2 * V.n3
                    )  # some smart memory efficient way to store the array
                    # Q = np.fromfile(fid, dt, 3 * V.n1 * V.n2 * V.n3)
                    newpos = (
                        np.float32().nbytes * V.n1 * V.n2 * V.n3 + inipos
                    )  # we calculate the offset that we would expect normally with a np.fromfile
                    fid.seek(
                        newpos, os.SEEK_SET
                    )  # we set the file pointer position to this offset

                    V.data[varname + "_X"] = np.transpose(
                        Q[::3].reshape(V.n3, V.n2, V.n1)
                    )
                    V.data[varname + "_Y"] = np.transpose(
                        Q[1::3].reshape(V.n3, V.n2, V.n1)
                    )
                    V.data[varname + "_Z"] = np.transpose(
                        Q[2::3].reshape(V.n3, V.n2, V.n1)
                    )

                else:
                    raise ValueError(
                        "In idfxReadVTK: Unknown datatype '%s', should be 'SCALARS' or 'VECTORS'"
                        % datatype
                    )
                    break

                # logger.debug("field: %s" % varname)

                fid.readline()  # extra line feed
        fid.close()

        return V

    def fargoAdsgReadDat(self, on, *, directory=""):
        V = DataStructure()
        filebeg = "gas"
        densfile = os.path.join(directory, f"{filebeg}dens{on}.dat")
        vyfile = os.path.join(directory, f"{filebeg}vrad{on}.dat")
        vxfile = os.path.join(directory, f"{filebeg}vtheta{on}.dat")

        V.geometry = "polar"
        V.data = {}

        phi = np.loadtxt(os.path.join(directory, "used_azi.dat"))[:, 0]
        domain_x = np.zeros(len(phi) + 1)
        domain_x[:-1] = phi
        domain_x[-1] = 2 * np.pi
        domain_x -= np.pi
        # We avoid ghost cells
        domain_y = np.loadtxt(os.path.join(directory, "used_rad.dat"))
        domain_z = np.zeros(2)

        V.x1 = domain_y  # X-Edge
        V.x2 = domain_x  # Y-Edge
        V.x3 = domain_z  # Z-Edge #latitute

        V.n1 = len(V.x1) - 1  # if len(V.x1)>2 else 2
        V.n2 = len(V.x2) - 1  # if len(V.x2)>2 else 2
        V.n3 = len(V.x3) - 1  # if len(V.x3)>2 else 2
        if Path(densfile).is_file():
            V.data["RHO"] = (
                np.fromfile(densfile, dtype="float64")
                .reshape(V.n3, V.n1, V.n2)
                .transpose(1, 2, 0)
            )  # rad, pĥi, z
        if Path(vyfile).is_file():
            V.data["VX1"] = (
                np.fromfile(vyfile, dtype="float64")
                .reshape(V.n3, V.n1, V.n2)
                .transpose(1, 2, 0)
            )  # rad, pĥi, z
        if Path(vxfile).is_file():
            V.data["VX2"] = (
                np.fromfile(vxfile, dtype="float64")
                .reshape(V.n3, V.n1, V.n2)
                .transpose(1, 2, 0)
            )  # rad, pĥi, z

        return V

    def fargo3dReadDat(self, on, *, directory="", inifile=""):
        V = DataStructure()
        filebeg = "gas"
        densfile = os.path.join(directory, f"{filebeg}dens{on}.dat")
        vyfile = os.path.join(directory, f"{filebeg}vy{on}.dat")
        vxfile = os.path.join(directory, f"{filebeg}vx{on}.dat")
        vzfile = os.path.join(directory, f"{filebeg}vz{on}.dat")

        if inifile == "":
            params = Parameters(directory=directory, inifile=inifile, code="")
        else:
            params = Parameters(directory=directory, inifile=inifile, code="fargo3d")

        params.loadIniFile()

        V.geometry = params.inifile["COORDINATES"]
        V.data = {}

        domain_x = np.loadtxt(os.path.join(directory, "domain_x.dat"))
        # We avoid ghost cells
        domain_y = np.loadtxt(os.path.join(directory, "domain_y.dat"))[3:-3]
        domain_z = np.loadtxt(os.path.join(directory, "domain_z.dat"))
        if domain_z.shape[0] > 6:
            domain_z = domain_z[3:-3]

        if V.geometry == "cylindrical":
            V.geometry = "polar"
            V.x1 = domain_y  # X-Edge
            V.x2 = domain_x  # Y-Edge
            V.x3 = domain_z  # Z-Edge #latitute

            V.n1 = len(V.x1) - 1  # if len(V.x1)>2 else 2
            V.n2 = len(V.x2) - 1  # if len(V.x2)>2 else 2
            V.n3 = len(V.x3) - 1  # if len(V.x3)>2 else 2
            if Path(densfile).is_file():
                V.data["RHO"] = (
                    np.fromfile(densfile, dtype="float64")
                    .reshape(V.n3, V.n1, V.n2)
                    .transpose(1, 2, 0)
                )  # rad, pĥi, z
            if Path(vyfile).is_file():
                V.data["VX1"] = (
                    np.fromfile(vyfile, dtype="float64")
                    .reshape(V.n3, V.n1, V.n2)
                    .transpose(1, 2, 0)
                )  # rad, pĥi, z
            if Path(vxfile).is_file():
                V.data["VX2"] = (
                    np.fromfile(vxfile, dtype="float64")
                    .reshape(V.n3, V.n1, V.n2)
                    .transpose(1, 2, 0)
                )  # rad, pĥi, z
            if Path(vzfile).is_file():
                V.data["VX3"] = (
                    np.fromfile(vzfile, dtype="float64")
                    .reshape(V.n3, V.n1, V.n2)
                    .transpose(1, 2, 0)
                )  # rad, pĥi, z
            for key in list(V.data.keys()):
                V.data[key] = np.roll(V.data[key], V.n2 // 2, axis=1)
        elif V.geometry == "spherical":
            V.x1 = domain_y  # X-Edge
            V.x2 = domain_z  # Z-Edge #latitute
            V.x3 = domain_x  # Y-Edge

            V.n1 = len(V.x1) - 1  # if len(V.x1)>2 else 2
            V.n2 = len(V.x2) - 1  # if len(V.x2)>2 else 2
            V.n3 = len(V.x3) - 1  # if len(V.x3)>2 else 2
            if Path(densfile).is_file():
                V.data["RHO"] = (
                    np.fromfile(densfile, dtype="float64")
                    .reshape(V.n2, V.n1, V.n3)
                    .transpose(1, 0, 2)
                )  # rad, pĥi, z
            if Path(vyfile).is_file():
                V.data["VX1"] = (
                    np.fromfile(vyfile, dtype="float64")
                    .reshape(V.n2, V.n1, V.n3)
                    .transpose(1, 0, 2)
                )  # rad, pĥi, z
            if Path(vzfile).is_file():
                V.data["VX2"] = (
                    np.fromfile(vzfile, dtype="float64")
                    .reshape(V.n2, V.n1, V.n3)
                    .transpose(1, 0, 2)
                )  # rad, pĥi, z
            if Path(vxfile).is_file():
                V.data["VX3"] = (
                    np.fromfile(vxfile, dtype="float64")
                    .reshape(V.n2, V.n1, V.n3)
                    .transpose(1, 0, 2)
                )  # rad, pĥi, z
            for key in list(V.data.keys()):
                V.data[key] = np.roll(V.data[key], V.n2 // 2, axis=2)
        else:
            raise ValueError(f"{V.geometry} not implemented yet for fargo3d.")

        return V
