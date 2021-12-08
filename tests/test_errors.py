import pytest

from nonos.api.from_simulation import Parameters


def test_init_params_wo_a_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        init = Parameters()
        init.loadIniFile()
