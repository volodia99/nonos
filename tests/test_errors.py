import pytest

from nonos.api import Parameters


def test_init_params_wo_a_file():
    with pytest.raises(
        FileNotFoundError, match=r"idefix.ini, pluto.ini, variables.par not found"
    ):
        init = Parameters()
        init.loadIniFile()
