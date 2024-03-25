import os

import numpy as np
import pytest

from nonos.api import GasDataSet, file_analysis


class TestFileAnalysis:
    @pytest.mark.parametrize(
        "directory",
        ["idefix_planet3d", "fargo_adsg_planet"],
    )
    def test_simple(self, test_data_dir, directory):
        result = file_analysis(
            "planet0.dat",
            directory=test_data_dir / directory,
        )
        assert isinstance(result, np.ndarray)

    def test_norb(self, test_data_dir):
        result = file_analysis(
            "planet0.dat",
            directory=test_data_dir / "idefix_planet3d",
            norb=10,
        )
        assert isinstance(result, np.ndarray)

    def test_norb_not_idefix(self, test_data_dir):
        with pytest.raises(NotImplementedError):
            file_analysis(
                "planet0.dat",
                directory=test_data_dir / "fargo_adsg_planet",
                norb=10,
            )

    def test_implicit_directory(self, test_data_dir):
        os.chdir(test_data_dir / "idefix_planet3d")
        result = file_analysis("planet0.dat")
        assert isinstance(result, np.ndarray)


class TestGasDataSetFromNpy:
    expected_keys = ["RHO"]
    args = (7283,)
    kwargs = {"operation": "azimuthal_average"}
    directory = "pluto_spherical"

    def test_from_npy_implicit_directory(self, test_data_dir):
        os.chdir(test_data_dir / self.directory)
        ds = GasDataSet(*self.args, **self.kwargs)
        assert sorted(ds.keys()) == self.expected_keys

    def test_from_npy_explicit_directory(self, test_data_dir):
        ds = GasDataSet(
            *self.args,
            **self.kwargs,
            directory=test_data_dir / self.directory,
        )
        assert sorted(ds.keys()) == self.expected_keys

    def test_deprecation(self, test_data_dir):
        with pytest.deprecated_call():
            ds = GasDataSet.from_npy(
                *self.args,
                **self.kwargs,
                directory=test_data_dir / self.directory,
            )
        assert sorted(ds.keys()) == self.expected_keys
