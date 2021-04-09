from nonos import InitParamNonos, FieldNonos
import os
import pytest

def test_init_params_wo_a_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        init = InitParamNonos()
        init.load(info=True)
