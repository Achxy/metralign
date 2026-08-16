import numpy as np
import pytest

from drift_sense.architectures import render_architecture


@pytest.mark.parametrize("architecture", ["dram", "finfet"])
def test_geometry_is_deterministic_and_bounded(architecture):
    y, x = np.mgrid[-20:20:80j, -20:20:80j]
    first = render_architecture(architecture, x, y, seed=42)
    second = render_architecture(architecture, x, y, seed=42)
    assert np.array_equal(first, second)
    assert 0 <= first.min() <= first.max() <= 1
    assert first.std() > 0.05


def test_process_variation_is_wafer_persistent():
    # Overlapping world points evaluate identically even when requested as
    # differently shaped capture grids.
    x = np.linspace(-15, 15, 31)
    y = np.linspace(-12, 12, 25)
    xx, yy = np.meshgrid(x, y)
    direct = render_architecture("dram", xx, yy, seed=8)
    flat = render_architecture("dram", xx.ravel(), yy.ravel(), seed=8).reshape(xx.shape)
    assert np.array_equal(direct, flat)


@pytest.mark.parametrize("architecture", ["dram", "finfet"])
def test_process_variations_are_independently_removable(architecture):
    y, x = np.mgrid[-35:35:140j, -35:35:140j]
    default = render_architecture(architecture, x, y, seed=42)
    components = ("pitch_variation", "center_variation", "width_variation", "edge_roughness")
    for component in components:
        without = render_architecture(
            architecture, x, y, seed=42, disabled_variations=(component,)
        )
        assert not np.array_equal(without, default)
    all_disabled = render_architecture(
        architecture, x, y, seed=42, disabled_variations=components
    )
    ideal = render_architecture(architecture, x, y, seed=42, geometry_variant="ideal")
    assert np.allclose(all_disabled, ideal)
