import matplotlib
import numpy as np

matplotlib.use("Agg")

from framebfn import per_residue_ramachandra


def test_per_residue_ramachandra_with_free_energy():
    rng = np.random.default_rng(3)
    angles = rng.uniform(-180, 180, size=(100, 4, 3))
    fig, axes = per_residue_ramachandra(angles, free_energy=True, bins=12)
    assert axes.size == 6
    assert len(fig.axes) == 7

