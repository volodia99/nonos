import re
from pathlib import Path

import pytest

from nonos.config import DEFAULTS

SOURCE_DIR = Path(__file__).parents[1]
with open(SOURCE_DIR.joinpath("nonos", "main.py")) as fh:
    source = fh.read()

used_keys = set(re.findall(r"""args\[['"](\w+)['"]\]""", source))
def_keys = set(DEFAULTS)


# check that every key used is defined
def test_no_undef_key():
    if missing := used_keys.difference(def_keys):
        print(f"The following {len(missing)} keys are used but not defined:\n")
        print("\n".join(list(missing)))
    assert not missing


@pytest.mark.xfail(
    reason="at the moment, at least one arg (grid) isn't used externally but inside a class"
)
def test_no_unused_key():
    # TODO: remove the xfail marker and refactor as needed
    if missing := def_keys.difference(used_keys):
        print(f"The following {len(missing)} keys are defined but not used:\n")
        print("\n".join(list(missing)))
    assert not missing
