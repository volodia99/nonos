import numpy as np
import numpy.testing as npt
import pytest

from nonos._types import FrameType, PlanetData


class TestPlanetData:
    def setup_class(cls):
        cls.data = PlanetData(
            x=np.array([1.0]),
            y=np.array([0.0]),
            z=np.array([0.0]),
            vx=np.array([0.0]),
            vy=np.array([1.0]),
            vz=np.array([0.0]),
            q=np.array([1e-12]),  # almost a test particle
            t=np.array([0.0]),
            dt=np.array([0.1]),
        )

    def test_distance_to_origin(self):
        pd = self.data
        # given z=0
        npt.assert_allclose(pd.d, np.hypot(pd.x, pd.y))

    def test_get_orbital_elements_unset(self):
        oe = self.data.get_orbital_elements(FrameType.FIXED_FRAME)
        npt.assert_allclose(oe.i, 0.0)
        npt.assert_allclose(oe.e, 1e-12, rtol=1e-4)
        npt.assert_allclose(oe.a, 1.0)

    def test_get_orbital_elements_fixed(self):
        with pytest.raises(NotImplementedError):
            self.data.get_orbital_elements(FrameType.CONSTANT_ROTATION)

    def test_get_orbital_elements_corot(self):
        self.data.get_orbital_elements(FrameType.PLANET_COROTATION)

    def test_get_rotational_rate(self):
        res = self.data.get_rotational_rate()
        npt.assert_allclose(res, 1.0)
