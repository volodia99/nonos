import os
from pathlib import Path
import pytest
from nonos.main import main

@pytest.fixture()
def test_data_dir():
    return Path(__file__).parent / "data"

def test_plot_simple(test_data_dir, capsys):
    # just check that the call returns no err
    os.chdir(test_data_dir / "idefix_rwi") 
    ret = main(["-mod", "d"], show=False)
    assert ret == 0

    out, err = capsys.readouterr()
    assert out == ""
    assert err == ""

def test_plot_simple_corotation(test_data_dir, capsys):
    # just check that the call returns no err
    os.chdir(test_data_dir / "idefix_rwi") 
    ret = main(["-mod", "d", "-cor"], show=False)
    assert ret == 0

    out, err = capsys.readouterr()
    assert out == ""
    # ignore differences in text wrapping because they are an implementation detail
    # due to the fact we use rich to display warnings
    assert err.strip().replace("\n", " ") == "Warning | We don't rotate the grid if there is no planet for now. omegagrid = 0."

def test_plot_planet_corotation(test_data_dir):
    from nonos import InitParamNonos, FieldNonos
    from nonos.main import find_nearest

    os.chdir(test_data_dir / "idefix_planet3d") 

    init = InitParamNonos()
    fieldon = FieldNonos(init, field='RHO', on=43)
    azimfield=fieldon.data[find_nearest(fieldon.x,1.0),:,fieldon.imidplane]
    assert find_nearest(azimfield, azimfield.max()) != fieldon.ny//2

    initPlanet = InitParamNonos(isPlanet=True, corotate=True)
    fieldonPlanet = FieldNonos(initPlanet, field='RHO', on=43)
    azimfieldPlanet = fieldonPlanet.data[find_nearest(fieldonPlanet.x,1.0),:,fieldonPlanet.imidplane]
    assert find_nearest(azimfieldPlanet, azimfieldPlanet.max()) == fieldonPlanet.ny//2
