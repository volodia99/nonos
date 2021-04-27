import os
import re

import pytest
import sys
from nonos import InitParamNonos
from nonos.main import main
from nonos.config import DEFAULTS
import toml

@pytest.mark.skipif(sys.platform.startswith("win"), reason="does not run on windows")
def test_config_dir_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        InitParamNonos(sim_paramfile=tmp_path / "notafile")


@pytest.fixture()
def minimal_paramfile(tmp_path):
    ifile = tmp_path / "nonos.toml"
    # check that this setup still makes sense
    assert DEFAULTS["dimensionality"] == 2
    with open(ifile, "w") as fh:
        fh.write("dimensionality = 1")
    return ifile

def test_load_config_file(minimal_paramfile, capsys):
    os.chdir(minimal_paramfile.parent)
    ret = main(["-input", "nonos.toml", "-config"])
    
    assert ret == 0
    out, err = capsys.readouterr()
    assert err == ""
    conf = toml.loads(out)
    assert conf["dimensionality"] == 1


def test_isolated_mode(minimal_paramfile, capsys):
    os.chdir(minimal_paramfile.parent)
    ret = main(["-config", "-isolated"])

    assert ret == 0
    out, err = capsys.readouterr()
    assert err == ""
    conf = toml.loads(out)
    assert conf["dimensionality"] == DEFAULTS["dimensionality"]