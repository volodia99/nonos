import os
from shutil import copytree

import pytest

from nonos.api import GasDataSet


def test_from_npy_error(test_data_dir):
    os.chdir(test_data_dir / "idefix_spherical_planet3d")

    with pytest.raises(
        FileNotFoundError, match="Original output was not reduced, or file"
    ):
        GasDataSet.from_npy(500, operation="azimuthal_averag")


def test_roundtrip_simple(test_data_dir, tmp_path):
    copytree(test_data_dir / "idefix_spherical_planet3d", tmp_path / "mydir")

    os.chdir(tmp_path / "mydir")
    ds = GasDataSet(500)
    assert len(list(ds.keys())) == 7

    gf = ds["RHO"].azimuthal_average()

    gf.save()
    dsnpy = GasDataSet.from_npy(500, operation="azimuthal_average")
    assert len(list(dsnpy.keys())) == 1


def test_roundtrip_other_dir(test_data_dir, tmp_path):
    os.chdir(test_data_dir / "idefix_spherical_planet3d")
    gf = GasDataSet(500)["RHO"].azimuthal_average()
    gf.save(tmp_path)
    dsnpy = GasDataSet.from_npy(500, operation="azimuthal_average", directory=tmp_path)
    assert len(list(dsnpy.keys())) == 1


def test_api_vtk_by_name(test_data_dir):
    os.chdir(test_data_dir / "idefix_spherical_planet3d")

    on = 500

    ds = GasDataSet(f"data.{on:04d}.vtk")
    assert ds.on == on

    with pytest.raises(
        FileNotFoundError, match="In idfxReadVTK: datawrong.0500.vtk not found."
    ):
        GasDataSet(f"datawrong.{on:04d}.vtk")


def test_api_vtk_by_name_fargo(test_data_dir):
    os.chdir(test_data_dir / "fargo3d_planet2d")

    with pytest.raises(TypeError, match="on can only be an int for fargo3d"):
        GasDataSet(f"gasdens{40:04d}.dat")


def test_api_fluid_fargo3d(test_data_dir):
    os.chdir(test_data_dir / "fargo3d_multifluid")

    on = 5

    ds = GasDataSet(on, fluid="dust2")
    assert len(list(ds.keys())) == 1

    with pytest.raises(
        FileNotFoundError,
        match=r"No file matches the pattern 'dust4\*5\.dat'",
    ):
        GasDataSet(on, fluid="dust4")


def test_api_fluid_idefix(test_data_dir):
    os.chdir(test_data_dir / "idefix_spherical_planet3d")

    on = 500

    with pytest.raises(ValueError, match="fluid is defined only for fargo3d outputs"):
        GasDataSet(on, fluid="dust1")
