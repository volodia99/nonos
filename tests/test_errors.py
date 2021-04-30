import pytest

from nonos import InitParamNonos


def test_init_params_wo_a_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        init = InitParamNonos()
        init.load()
