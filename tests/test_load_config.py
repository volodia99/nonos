import os
import re

import pytest
import sys

from nonos.main import AnalysisNonos

def test_load_analysis(tmp_path):
    os.chdir(tmp_path)
    AnalysisNonos()

@pytest.mark.skipif(sys.platform.startswith("win"), reason="does not run on windows")
def test_config_dir_not_found(tmp_path):
    with pytest.raises(
        FileNotFoundError,
        match=re.escape(f"[Errno 2] No such file or directory: '{tmp_path / 'config.toml'}'")
    ):
        # the error is raised by toml.load
        AnalysisNonos(directory_of_script=tmp_path)