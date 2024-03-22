import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import SymLogNorm

from nonos.api import GasDataSet
from nonos.styling import set_mpl_style


@pytest.fixture
def tmp_mpl_state():
    # reset matplotlib's state when the test is over
    style = mpl.rcParams.copy()
    yield
    mpl.rcParams.update(style)


@pytest.mark.usefixtures("tmp_mpl_state")
@pytest.mark.parametrize("scaling", [0.5, 1.0, 2.0])
@pytest.mark.mpl_image_compare(style="default")
def test_set_mpl_style(scaling):
    set_mpl_style(scaling)

    fig, ax = plt.subplots()

    x = np.linspace(0, 2 * np.pi)
    for phase in np.linspace(0, np.pi / 2, 5):
        y = np.sin(x + phase)
        ax.plot(x, y)

    ax.set(
        title="nonos style",
        xlabel="$x$ axis",
        ylabel="$y$ axis",
    )
    ax.annotate(f"{scaling=}", (0.05, 0.1), xycoords="axes fraction", fontsize=15)

    return fig


@pytest.mark.mpl_image_compare()
def test_2D_polar_plane(test_data_dir, temp_figure_and_axis):
    ds = GasDataSet(23, directory=test_data_dir / "idefix_newvtk_planet2d")
    fig, ax = temp_figure_and_axis
    ds["VX1"].map("R", "phi").plot(fig, ax, title="vr")
    return fig


@pytest.mark.mpl_image_compare()
def test_symlog(test_data_dir, temp_figure_and_axis):
    ds = GasDataSet(23, directory=test_data_dir / "idefix_newvtk_planet2d")
    fig, ax = temp_figure_and_axis
    ds["VX1"].map("R", "phi").plot(
        fig,
        ax,
        title="vr",
        norm=SymLogNorm(vmin=-0.05, vmax=0.05, linthresh=1e-4, base=10),
    )
    return fig


@pytest.mark.mpl_image_compare()
def test_3D_aa_xz(test_data_dir, temp_figure_and_axis):
    ds = GasDataSet(500, directory=test_data_dir / "idefix_spherical_planet3d")
    fig, ax = temp_figure_and_axis

    ds["RHO"].azimuthal_average().map("x", "z").plot(fig, ax, log=True, title="rho")
    return fig


@pytest.mark.mpl_image_compare()
def test_3D_vm_phiR(test_data_dir, temp_figure_and_axis):
    ds = GasDataSet(500, directory=test_data_dir / "idefix_spherical_planet3d")
    fig, ax = temp_figure_and_axis

    ds["RHO"].vertical_at_midplane().map("phi", "R").plot(
        fig, ax, log=True, title="rho"
    )
    return fig


@pytest.mark.mpl_image_compare()
def test_3D_vm_xy(test_data_dir, temp_figure_and_axis):
    ds = GasDataSet(500, directory=test_data_dir / "idefix_spherical_planet3d")
    fig, ax = temp_figure_and_axis

    ds["RHO"].vertical_at_midplane().map("x", "y").plot(fig, ax, log=True, title="rho")
    return fig
