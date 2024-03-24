import os

import inifix
import pytest

from nonos.config import DEFAULTS
from nonos.main import main


@pytest.fixture()
def minimal_paramfile(tmp_path):
    ifile = tmp_path / "nonos.ini"
    # check that this setup still makes sense
    assert DEFAULTS["field"] == "RHO"
    with open(ifile, "w") as fh:
        fh.write("field  VX1")
    return ifile


def test_load_config_file(minimal_paramfile, capsys):
    os.chdir(minimal_paramfile.parent)
    ret = main(["-input", "nonos.ini", "-config"])

    assert ret == 0
    out, err = capsys.readouterr()
    assert "Using parameters from" in err
    conf = inifix.loads(out)
    assert conf["field"] == "VX1"


def test_isolated_mode(minimal_paramfile, capsys):
    os.chdir(minimal_paramfile.parent)
    ret = main(["-config", "-isolated"])

    assert ret == 0
    out, err = capsys.readouterr()
    assert err == ""
    conf = inifix.loads(out)
    assert conf["field"] == DEFAULTS["field"]
