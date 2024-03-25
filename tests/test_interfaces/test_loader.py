import sys
from pathlib import Path

import pytest

from nonos.loaders import Loader, Recipe, loader_from, recipe_from

if sys.version_info >= (3, 9):
    removeprefix = str.removeprefix
    removesuffix = str.removesuffix
else:
    from nonos._backports import removeprefix, removesuffix


class TestLoader:

    @pytest.fixture(params=Loader.__slots__, ids=lambda s: removesuffix(s, "_"))
    def loader_slot(self, request):
        return request.param

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            Loader(
                parameter_file=tmp_path / "this_file_does_not_exist",
                binary_reader=None,
                planet_reader=None,
                ini_reader=None,
            )

    def test_read_only_interface(self, tmp_path, loader_slot):
        # mock a minimal loader, even if not type-check compliant
        parameter_file = tmp_path / "fake.ini"
        parameter_file.touch()
        loader = Loader(
            parameter_file=parameter_file,
            binary_reader=None,
            planet_reader=None,
            ini_reader=None,
        )

        public_name = removeprefix(loader_slot, "_")
        assert hasattr(loader, public_name)
        with pytest.raises(AttributeError):
            setattr(loader, public_name, None)


class TestGetRecipe:
    def test_no_args(self):
        with pytest.raises(TypeError):
            recipe_from()

    def test_only_position_arg(self):
        with pytest.raises(TypeError):
            recipe_from("idefix_vtk")

    @pytest.mark.parametrize(
        "in_, expected",
        [
            pytest.param("idefix_vtk", Recipe.IDEFIX_VTK, id="idefix_vtk"),
            pytest.param("pluto_vtk", Recipe.PLUTO_VTK, id="pluto_vtk"),
            pytest.param("fargo3d", Recipe.FARGO3D, id="fargo3D"),
            pytest.param("fargo_adsg", Recipe.FARGO_ADSG, id="fargo_adsg"),
            pytest.param("idefix", Recipe.IDEFIX_VTK, id="(backward-compat) idefix"),
            pytest.param("pluto", Recipe.PLUTO_VTK, id="(backward-compat) pluto"),
        ],
    )
    def test_recipe_from_code(self, in_, expected):
        assert recipe_from(code=in_) is expected

    @pytest.mark.parametrize(
        "in_",
        [
            # using '-' instead of '_'
            "idefix-vtk",
            "pluto-vtk",
            "fargo-adsg",
            # empty string
            "",
            # and something random
            "bleurg",
        ],
    )
    def test_invalid_code(self, in_):
        with pytest.raises(ValueError):
            recipe_from(code=in_)

    @pytest.mark.parametrize(
        "in_, expected",
        [
            pytest.param(
                Path("idefix_planet3d", "idefix.ini"),
                Recipe.IDEFIX_VTK,
                id="idefix_vtk",
            ),
            pytest.param(
                Path("pluto_spherical", "pluto.ini"),
                Recipe.PLUTO_VTK,
                id="pluto_vtk",
            ),
            pytest.param(
                Path("fargo3d_planet2d", "variables.par"),
                Recipe.FARGO3D,
                id="fargo3d",
            ),
            pytest.param(
                Path("fargo_adsg_planet", "planetpendragon_200k.par"),
                Recipe.FARGO_ADSG,
                id="fargo_adsg",
            ),
        ],
    )
    def test_valid_parameter_file(self, test_data_dir, in_, expected):
        assert recipe_from(parameter_file=test_data_dir / in_) is expected

    def test_ambiguous_parameter_file(self, tmp_path):
        fake_ini = tmp_path / "fake.ini"
        fake_ini.write_text("")
        with pytest.raises(
            ValueError,
            match=r"^Could not determine data format from",
        ):
            recipe_from(
                parameter_file=fake_ini,
            )

    def test_ambiguous_parameter_file_with_directory(self, tmp_path):
        fake_ini = tmp_path / "fake.ini"
        fake_ini.write_text("")
        with pytest.raises(
            ValueError,
            match=r"^Could not determine data format from parameter_file",
        ):
            recipe_from(parameter_file=fake_ini, directory=tmp_path)

    @pytest.mark.parametrize(
        "in_, expected",
        [
            pytest.param(
                "idefix_planet3d",
                Recipe.IDEFIX_VTK,
                id="idefix_vtk",
            ),
            pytest.param(
                "pluto_spherical",
                Recipe.PLUTO_VTK,
                id="pluto_vtk",
            ),
            pytest.param(
                "fargo3d_planet2d",
                Recipe.FARGO3D,
                id="fargo3d",
            ),
        ],
    )
    def test_directory(self, test_data_dir, in_, expected):
        assert recipe_from(directory=test_data_dir / in_) is expected

    def test_ambiguous_directory(self, tmp_path):
        tmp_path.joinpath("idefix.ini").touch()
        tmp_path.joinpath("pluto.ini").touch()
        with pytest.raises(
            RuntimeError,
            match=r"^Found multiple parameter files",
        ):
            recipe_from(directory=tmp_path)


class TestGetLoader:
    @pytest.mark.parametrize(
        "in_",
        [
            pytest.param(
                Path("idefix_planet3d", "idefix.ini"),
                id="idefix_vtk",
            ),
            pytest.param(
                Path("pluto_spherical", "pluto.ini"),
                id="pluto_vtk",
            ),
            pytest.param(
                Path("fargo3d_planet2d", "variables.par"),
                id="fargo3d",
            ),
            pytest.param(
                Path("fargo_adsg_planet", "planetpendragon_200k.par"),
                id="fargo_adsg",
            ),
        ],
    )
    def test_loader_from_parameter_file(self, test_data_dir, in_):
        assert type(loader_from(parameter_file=test_data_dir / in_)) is Loader
