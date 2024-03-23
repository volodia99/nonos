import os
from pathlib import Path

import numpy as np
import pytest

from nonos.api import GasDataSet, file_analysis

TEST_DATA_DIR = Path(__file__).parent / "data"


class TestFileAnalysis:
    @pytest.mark.parametrize(
        "directory",
        [TEST_DATA_DIR / "idefix_planet3d", TEST_DATA_DIR / "fargo_adsg_planet"],
    )
    def test_simple(self, directory):
        result = file_analysis(
            "planet0.dat",
            directory=directory,
        )
        assert isinstance(result, np.ndarray)

    def test_norb(self):
        result = file_analysis(
            "planet0.dat",
            directory=TEST_DATA_DIR / "idefix_planet3d",
            norb=10,
        )
        assert isinstance(result, np.ndarray)


class TestGasDataSetFromNpy:
    def setup_class(cls):
        cls.kwargs = {
            "on": 7283,
            "directory": TEST_DATA_DIR / "pluto_spherical",
            "operation": "azimuthal_average",
        }
        cls.expected_keys = ["RHO"]

    def test_local_load(self):
        kwargs = self.kwargs.copy()
        os.chdir(kwargs.pop("directory"))
        ds = GasDataSet.from_npy(**kwargs)
        assert sorted(ds.keys()) == sorted(self.expected_keys)

    def test_load_from_anywhere(self):
        ds = GasDataSet.from_npy(**self.kwargs)
        assert sorted(ds.keys()) == sorted(self.expected_keys)
