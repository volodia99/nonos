import os
from glob import glob
from pathlib import Path

import pytest

from nonos.main import main


@pytest.fixture()
def test_data_dir():
    return Path(__file__).parent / "data"


@pytest.fixture(params=["idefix_rwi", "idefix_planet3d", "fargo3d_planet2d"])
def simulation_dir(test_data_dir, request):
    return test_data_dir / request.param


ARGS_TO_CHECK = {
    "vanilla_conf": ["-geometry", "polar"],
    "diff": ["-geometry", "polar", "-diff"],
    "log": ["-geometry", "polar", "-log"],
    "movie_xy": ["-geometry", "polar", "-all", "-plane", "x", "y"],
    "movie_with_diff": ["-geometry", "polar", "-all", "-diff"],
    "movie_with_multiproc": ["-geometry", "polar", "-all", "-ncpu", "2"],
}


@pytest.mark.parametrize("argv", ARGS_TO_CHECK.values(), ids=ARGS_TO_CHECK.keys())
def test_plot_simple(argv, simulation_dir, capsys, tmp_path):
    os.chdir(tmp_path)
    ret = main(argv + ["-dir", str(simulation_dir), "-geometry", "polar"])

    out, err = capsys.readouterr()
    assert err == ""
    assert out == ""
    assert ret == 0
    assert len(glob("*.png")) > 0


@pytest.mark.parametrize("format", ["pdf", "png", "jpg"])
def test_common_image_formats(format, simulation_dir, capsys, tmp_path):
    os.chdir(tmp_path)
    ret = main(["-dir", str(simulation_dir), "-fmt", format, "-geometry", "polar"])

    out, err = capsys.readouterr()
    assert err == ""
    assert out == ""
    assert ret == 0
    assert len(glob(f"*.{format}")) == 1


@pytest.mark.parametrize(
    "datadir",
    [
        str(Path(__file__).parent.joinpath("data", "idefix_planet3d")),
        str(Path(__file__).parent.joinpath("data", "fargo3d_planet2d")),
    ],
)
def test_plot_simple_corotation(datadir, capsys, tmp_path):
    os.chdir(tmp_path)
    # just check that the call returns no err
    ret = main(["-cor", "0", "-dir", datadir, "-geometry", "polar"])

    out, err = capsys.readouterr()
    assert out == ""
    assert err == ""
    assert ret == 0


@pytest.mark.parametrize(
    "datadir",
    [str(Path(__file__).parent.joinpath("data", "idefix_rwi"))],
)
def test_unknown_geometry(datadir, tmp_path):
    os.chdir(tmp_path)
    with pytest.raises(
        RuntimeError, match=r"Geometry couldn't be determined from data"
    ):
        main(["-dir", datadir])


@pytest.mark.parametrize(
    "datadir",
    [str(Path(__file__).parent.joinpath("data", "idefix_newvtk_planet2d"))],
)
def test_newvtk_geometry(datadir, capsys, tmp_path):
    os.chdir(tmp_path)
    ret = main(["-cor", "0", "-dir", datadir])
    out, err = capsys.readouterr()
    assert out == ""
    assert err == ""
    assert ret == 0


@pytest.mark.parametrize(
    "datadir",
    [str(Path(__file__).parent.joinpath("data", "idefix_rwi"))],
)
def test_error_no_planet(datadir, tmp_path):
    os.chdir(tmp_path)
    # just check that the call returns the correct err
    with pytest.raises(FileNotFoundError, match=r"planet0\.dat not found"):
        main(["-cor", "0", "-dir", datadir, "-geometry", "polar"])


def test_verbose_info(simulation_dir, capsys):
    ret = main(["-v", "-dir", str(simulation_dir), "-geometry", "polar"])

    out, err = capsys.readouterr()
    assert err == ""
    assert "Operation took" in out
    assert "INFO" in out
    assert ret == 0


# # @pytest.mark.xfail("broken test for now.")
# def test_verbose_debug(simulation_dir, capsys):
#     ret = main(["-vv", "-dir", str(simulation_dir), "-geometry", "polar"])

#     out, err = capsys.readouterr()
#     assert err == ""
#     assert "DEBUG" in out
#     assert ret == 0


def test_plot_planet_corotation(test_data_dir):
    from nonos.api import GasDataSet, find_nearest

    os.chdir(test_data_dir / "idefix_planet3d")

    ds = GasDataSet(43, geometry="polar")
    azimfield = ds["RHO"].radial_at_r().vertical_at_midplane().map("phi").data
    assert find_nearest(azimfield, azimfield.max()) != ds["RHO"].shape[2] // 2

    azimfieldPlanet = (
        ds["RHO"]
        .radial_at_r()
        .vertical_at_midplane()
        .map("phi", planet_corotation=0)
        .data
    )
    assert (
        find_nearest(azimfieldPlanet, azimfieldPlanet.max()) == ds["RHO"].shape[2] // 2
    )
