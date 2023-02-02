import pathlib
import sys
from io import StringIO

import pytest
from cogapp import Cog


@pytest.mark.skipif(
    sys.version_info >= (3, 10),
    reason="argparse --help messages were slightly modified in Python 3.10",
)
def test_if_cog_needs_to_be_run():
    _stdout = sys.stdout
    sys.stdout = StringIO()
    readme = pathlib.Path(__file__).parents[1] / "README.md"

    Cog().main(["cog", str(readme)])

    output = sys.stdout.getvalue()
    sys.stdout = _stdout
    assert (
        output == readme.read_text()
    ), "Run 'cog -r README.md' from the top level of the repo, with Python < 3.10"
