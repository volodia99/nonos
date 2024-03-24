import os
import sys

import pytest

from nonos.api import Parameters


def test_init_params_wo_a_file(tmp_path):
    os.chdir(tmp_path)
    with pytest.raises(
        FileNotFoundError, match=r"idefix.ini, pluto.ini, variables.par not found"
    ):
        Parameters()


@pytest.mark.skipif(sys.platform.startswith("win"), reason="does not run on windows")
def test_config_dir_not_found(tmp_path):
    with pytest.raises(FileNotFoundError, match=r"not found"):
        Parameters(inifile="notafile", code="idefix", directory=tmp_path)


def test_config_inifile_but_nocode(tmp_path):
    with pytest.raises(ValueError, match=r"both inifile and code have to be given"):
        Parameters(inifile="notafile", directory=tmp_path)
