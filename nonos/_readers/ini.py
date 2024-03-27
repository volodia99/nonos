__all__ = [
    "IdefixVTKReader",
    "PlutoVTKReader",
    "Fargo3DReader",
    "FargoADSGReader",
]
from pathlib import Path
from typing import final

import inifix

from nonos._readers._base import ReaderMixin
from nonos._types import FrameType, IniData, PathT


@final
class IdefixVTKReader(ReaderMixin):
    @staticmethod
    def read(file: PathT, /) -> IniData:

        class IdefixIniOutput:
            def __init__(self, *, vtk, **_kwargs) -> None:
                self.vtk = float(vtk)

        class IdefixIniHydro:
            def __init__(self, **kwargs) -> None:
                if "rotation" in kwargs:
                    self.frame = FrameType.FIXED
                    self.rotation = float(kwargs["rotation"])
                else:
                    self.frame = FrameType.UNSET
                    self.rotation = 0.0

        class IdefixIni:
            def __init__(self, *, Hydro, Output, **_kwargs) -> None:
                self.hydro = IdefixIniHydro(**Hydro)
                self.output = IdefixIniOutput(**Output)

        meta = inifix.load(file)
        ini = IdefixIni(**meta)

        return IniData(
            file=Path(file).resolve(),
            frame=ini.hydro.frame,
            rotational_rate=ini.hydro.rotation,
            output_time_interval=ini.output.vtk,
            meta=meta,
        )


@final
class PlutoVTKReader(ReaderMixin):
    @staticmethod
    def read(file: PathT, /) -> IniData:
        class PlutoIniOutput:
            def __init__(self, *, vtk, **_kwargs) -> None:
                self.vtk = float(list(vtk)[0])

        class PlutoIni:
            def __init__(self, **kwargs) -> None:
                self.output = PlutoIniOutput(**kwargs["Static Grid Output"])

        meta = inifix.load(file)
        ini = PlutoIni(**meta)

        return IniData(
            file=Path(file).resolve(),
            frame=FrameType.UNSET,
            rotational_rate=0.0,
            output_time_interval=ini.output.vtk,
            meta=meta,
        )


@final
class Fargo3DReader(ReaderMixin):
    @staticmethod
    def read(file: PathT, /) -> IniData:
        class Fargo3DIni:
            def __init__(self, *, NINTERM, DT, FRAME, **kwargs) -> None:
                self.NINTERM = int(NINTERM)
                self.DT = float(DT)
                self.FRAME: FrameType
                self.OMEGAFRAME: float

                str_frame = str(FRAME)
                if str_frame == "F":
                    self.FRAME = FrameType.FIXED
                    self.OMEGAFRAME = float(kwargs["OMEGAFRAME"])
                elif str_frame == "C":
                    self.FRAME = FrameType.COROT
                    self.OMEGAFRAME = float("nan")
                else:
                    self.FRAME = FrameType.UNSET
                    self.OMEGAFRAME = 0.0

        meta = inifix.load(file)
        ini = Fargo3DIni(**meta)

        return IniData(
            file=Path(file).resolve(),
            frame=ini.FRAME,
            rotational_rate=ini.OMEGAFRAME,
            output_time_interval=ini.NINTERM * ini.DT,
            meta=meta,
        )


@final
class FargoADSGReader(ReaderMixin):
    @staticmethod
    def read(file: PathT, /) -> IniData:
        class Fargo3DIni:
            def __init__(self, *, Ninterm, DT, Frame, **kwargs) -> None:
                self.NINTERM = int(Ninterm)
                self.DT = float(DT)
                self.FRAME: FrameType
                self.OMEGAFRAME: float

                str_frame = str(Frame)
                if str_frame == "F":
                    self.FRAME = FrameType.FIXED
                    self.OMEGAFRAME = float(kwargs["OmegaFrame"])
                elif str_frame == "C":
                    self.FRAME = FrameType.COROT
                    self.OMEGAFRAME = float("nan")
                else:
                    self.FRAME = FrameType.UNSET
                    self.OMEGAFRAME = 0.0

        meta = inifix.load(file)
        ini = Fargo3DIni(**meta)

        return IniData(
            file=Path(file).resolve(),
            frame=ini.FRAME,
            rotational_rate=ini.OMEGAFRAME,
            output_time_interval=ini.NINTERM * ini.DT,
            meta=meta,
        )
