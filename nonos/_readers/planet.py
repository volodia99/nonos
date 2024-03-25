__all__ = [
    "NullReader",
    "IdefixReader",
    "Fargo3DReader",
    "FargoADSGReader",
]
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, final

import numpy as np

from nonos._readers._base import ReaderMixin
from nonos._types import PathT, PlanetData


@final
class NullReader(ReaderMixin):
    @staticmethod
    def get_planet_files(directory: Path, /) -> List[Path]:
        raise NotImplementedError(
            f"{directory} couldn't be read. The default reader class (NullReader) "
            "was previously selected, possibly by mistake ?"
        )

    @staticmethod
    def read(file: PathT, /) -> PlanetData:
        raise NotImplementedError(
            f"{file} couldn't be read. The default reader class (NullReader) "
            "was previously selected, possibly by mistake ?"
        )


@final
class IdefixReader(ReaderMixin):
    @staticmethod
    def get_planet_files(directory: Path, /) -> List[Path]:
        return sorted(directory.glob("planet*.dat"))

    @staticmethod
    def read(file: PathT, /) -> PlanetData:
        dt, x, y, z, vx, vy, vz, q, t = np.loadtxt(file).T
        return PlanetData(x, y, z, vx, vy, vz, q, t, dt)


class FargoReader(ReaderMixin, ABC):
    @staticmethod
    def get_planet_files(directory: Path, /) -> List[Path]:
        return [
            fn
            for fn in sorted(directory.glob("planet*.dat"))
            if re.search(r"planet\d+.dat$", str(fn)) is not None
        ]

    @staticmethod
    @abstractmethod
    def read(file: PathT, /) -> PlanetData: ...


@final
class Fargo3DReader(FargoReader):
    @staticmethod
    def read(file: PathT, /) -> PlanetData:
        dt, x, y, z, vx, vy, vz, q, t, *_ = np.loadtxt(file).T
        return PlanetData(x, y, z, vx, vy, vz, q, t, dt)


@final
class FargoADSGReader(FargoReader):
    @staticmethod
    def read(file: PathT, /) -> PlanetData:
        dt, x, y, vx, vy, q, _, _, t, *_ = np.loadtxt(file).T
        z = np.zeros_like(x)
        vz = np.zeros_like(vx)
        return PlanetData(x, y, z, vx, vy, vz, q, t, dt)
