import numpy as np

from drift_sense.geometry import CaptureGeometry


def test_capture_geometry_round_trip():
    geometry = CaptureGeometry(
        width=1000,
        height=1000,
        world_center_x=21.3,
        world_center_y=-8.4,
        pixel_size=0.97,
        rotation_deg=0.8,
        anisotropy=0.003,
        drift_linear=1.1,
        drift_quadratic=-0.3,
    )
    for x, y in [(499.5, 499.5), (123.25, 777.75), (880.1, 82.6)]:
        wx, wy = geometry.sensor_to_world(np.asarray(x), np.asarray(y))
        recovered_x, recovered_y = geometry.world_to_sensor(float(wx), float(wy))
        assert abs(recovered_x - x) < 1e-8
        assert abs(recovered_y - y) < 1e-8


def test_optical_center_is_invariant_to_drift():
    geometry = CaptureGeometry(101, 101, 7.0, -3.0, 0.1, drift_linear=5.0, drift_quadratic=2.0)
    wx, wy = geometry.sensor_to_world(np.asarray(50.0), np.asarray(50.0))
    assert wx == 7.0
    assert wy == -3.0
