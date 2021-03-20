import os
import re

import pytest

from nonos.main import AnalysisNonos

def test_load_analysis(tmp_path):
    os.chdir(tmp_path)
    AnalysisNonos()

def test_config_dir_not_found(tmp_path):
    with pytest.raises(
        FileNotFoundError,
        match=re.escape(f"[Errno 2] No such file or directory: '{tmp_path / 'config.toml'}'")
    ):
        # the error is raised by toml.load
        AnalysisNonos(directory_of_script=tmp_path)