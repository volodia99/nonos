__all__ = [
    "Loader",
    "Recipe",
    "loader_from",
    "recipe_from",
]
import sys
from dataclasses import dataclass
from enum import auto
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Type, TypedDict, final

import nonos._readers as readers
from nonos._types import BinReader, IniReader, PlanetReader

if sys.version_info >= (3, 11):
    from enum import StrEnum
    from typing import assert_never
else:
    from typing_extensions import assert_never

    from nonos._backports import StrEnum


if TYPE_CHECKING:
    from nonos._types import BinData, IniData, PathT, PlanetData


class Recipe(StrEnum):
    IDEFIX_VTK = auto()
    PLUTO_VTK = auto()
    FARGO3D = auto()
    FARGO_ADSG = auto()


@final
@dataclass(frozen=True)
class Loader:
    # TODO: use slots=True in @dataclass when Python 3.9 is dropped
    __slots__ = [
        "parameter_file",
        "binary_reader",
        "planet_reader",
        "ini_reader",
    ]
    parameter_file: Path
    binary_reader: Type["BinReader"]
    planet_reader: Type["PlanetReader"]
    ini_reader: Type["IniReader"]

    def __post_init__(self) -> None:
        pf = Path(self.parameter_file).resolve()
        if not pf.is_file():
            raise FileNotFoundError(pf)
        object.__setattr__(self, "parameter_file", pf)

    def load_bin_data(self, file: "PathT", /, **meta) -> "BinData":
        ini = self.load_ini_file()
        if sys.version_info >= (3, 9):
            meta = ini.meta | meta
        else:
            meta = {**ini.meta, **meta}
        return self.binary_reader.read(file, **meta)

    def load_planet_data(self, file: "PathT") -> "PlanetData":
        return self.planet_reader.read(file)

    def load_ini_file(self) -> "IniData":
        return self.ini_reader.read(self.parameter_file)


def loader_from(
    *,
    code: Optional[str] = None,
    parameter_file: Optional["PathT"] = None,
    directory: Optional["PathT"] = None,
) -> Loader:
    return _compose_loader(
        recipe_from(
            code=code,
            parameter_file=parameter_file,
            directory=directory,
        ),
        _parameter_file_from(
            parameter_file=parameter_file,
            directory=directory,
        ),
    )


def _compose_loader(recipe: Recipe, /, parameter_file: Path) -> Loader:
    return Loader(parameter_file, **_ingredients_from(recipe))


def _parameter_file_from(
    *,
    parameter_file: Optional["PathT"] = None,
    directory: Optional["PathT"] = None,
) -> Path:
    if parameter_file is not None:
        return Path(parameter_file).resolve()
    elif directory is not None:
        directory = Path(directory).resolve()
        return _parameter_file_from_dir(directory)
    else:
        raise TypeError(
            "Missing required keyword arguments: 'parameter_file', 'directory' "
            "(need at least one)"
        )


def _parameter_file_from_dir(directory: "PathT", /) -> Path:
    directory = Path(directory).resolve()
    candidates = list(directory.glob("*.ini"))
    candidates.extend(list(directory.glob("*.par")))
    if len(candidates) == 1:
        return candidates[0]
    elif not candidates:
        raise FileNotFoundError(f"Could not find a parameter file in {directory}")
    else:
        raise RuntimeError(
            f"Found multiple parameter files in {directory}\n - "
            + "\n - ".join(str(c) for c in candidates)
        )


class Ingredients(TypedDict):
    binary_reader: Type[BinReader]
    planet_reader: Type[PlanetReader]
    ini_reader: Type[IniReader]


def _ingredients_from(recipe: Recipe, /) -> Ingredients:
    if recipe is Recipe.IDEFIX_VTK:
        return {
            "binary_reader": readers.binary.VTKReader,
            "planet_reader": readers.planet.IdefixReader,
            "ini_reader": readers.ini.IdefixVTKReader,
        }
    elif recipe is Recipe.PLUTO_VTK:
        return {
            "binary_reader": readers.binary.VTKReader,
            "planet_reader": readers.planet.NullReader,
            "ini_reader": readers.ini.PlutoVTKReader,
        }
    elif recipe is Recipe.FARGO3D:
        return {
            "binary_reader": readers.binary.Fargo3DReader,
            "planet_reader": readers.planet.Fargo3DReader,
            "ini_reader": readers.ini.Fargo3DReader,
        }
    elif recipe is Recipe.FARGO_ADSG:
        return {
            "binary_reader": readers.binary.FargoADSGReader,
            "planet_reader": readers.planet.FargoADSGReader,
            "ini_reader": readers.ini.FargoADSGReader,
        }
    else:
        assert_never(recipe)


def recipe_from(
    *,
    code: Optional[str] = None,
    parameter_file: Optional["PathT"] = None,
    directory: Optional["PathT"] = None,
) -> Recipe:
    if code is not None:
        return _code_to_recipe(code)

    parameter_file = _parameter_file_from(
        parameter_file=parameter_file,
        directory=directory,
    )

    recipe_candidates: List[Recipe] = []
    for recipe in Recipe.__members__.values():
        loader = _compose_loader(recipe, parameter_file)
        try:
            loader.load_ini_file()
        except Exception:
            continue
        else:
            recipe_candidates.append(recipe)
    if len(recipe_candidates) == 1:
        return recipe_candidates[0]
    elif len(recipe_candidates) == 0:
        msg = (
            f"Could not determine data format from {parameter_file=!r} "
            "(failed to read with any loader)"
        )
    else:  # pragma: no cover
        msg = (
            f"Could not determine unambiguous data format from {parameter_file=!r} "
            f"(found {len(recipe_candidates)} candidates {recipe_candidates})"
        )

    raise ValueError(msg)


def _code_to_recipe(code: str, /) -> Recipe:
    if code in ("pluto", "idefix"):
        # backward compatibility layer
        # this could be deprecated at some point
        code += "_vtk"
    return Recipe(code)
