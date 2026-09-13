"""Conformational plotting utilities."""

import math

import matplotlib.pyplot as plt
import numpy as np


def per_residue_ramachandra(
    angles: np.ndarray,
    *,
    residue_names=None,
    free_energy: bool = False,
    bins: int = 50,
    temperature: float = 300.0,
    max_free_energy: float | None = 5.0,
    ncols: int = 3,
    cmap: str | None = None,
):
    """Plot one Ramachandran histogram or free-energy surface per residue.

    ``angles`` must have shape ``(frames, residues, >=2)`` in degrees. The
    function returns ``(figure, axes)`` so callers can further customize it.
    Free energy is computed as ``-RT log(P)`` in kcal/mol and shifted to a
    minimum of zero independently for every residue.
    """
    angles = np.asarray(angles)
    if angles.ndim != 3 or angles.shape[2] < 2:
        raise ValueError("angles must have shape (n_frames, n_residues, >=2)")
    if bins < 2 or ncols < 1 or temperature <= 0:
        raise ValueError("bins and ncols must be positive and temperature > 0")

    n_residues = angles.shape[1]
    if residue_names is not None and len(residue_names) != n_residues:
        raise ValueError("residue_names must contain one label per residue")

    nrows = math.ceil(n_residues / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4 * ncols, 4 * nrows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    flat_axes = axes.ravel()
    image = None
    cmap = cmap or ("turbo" if free_energy else "viridis")

    for index in range(n_residues):
        ax = flat_axes[index]
        phi, psi = angles[:, index, 0], angles[:, index, 1]
        valid = np.isfinite(phi) & np.isfinite(psi)
        if valid.any():
            histogram, x_edges, y_edges = np.histogram2d(
                phi[valid], psi[valid], bins=bins,
                range=[[-180, 180], [-180, 180]],
            )
            if free_energy:
                probability = histogram / histogram.sum()
                with np.errstate(divide="ignore"):
                    surface = -0.0019872041 * temperature * np.log(probability)
                finite = np.isfinite(surface)
                surface[finite] -= surface[finite].min()
                image = ax.imshow(
                    surface.T, origin="lower", extent=(-180, 180, -180, 180),
                    aspect="equal", cmap=cmap, vmin=0, vmax=max_free_energy,
                )
            else:
                image = ax.imshow(
                    histogram.T, origin="lower", extent=(-180, 180, -180, 180),
                    aspect="equal", cmap=cmap,
                )
        label = residue_names[index] if residue_names is not None else str(index + 1)
        ax.set_title(label)
        ax.set_xlim(-180, 180)
        ax.set_ylim(-180, 180)
        ax.set_xticks([-180, -90, 0, 90, 180])
        ax.set_yticks([-180, -90, 0, 90, 180])

    for ax in flat_axes[n_residues:]:
        ax.set_visible(False)
    fig.supxlabel(r"$\phi$ [°]")
    fig.supylabel(r"$\psi$ [°]")
    if free_energy and image is not None:
        fig.colorbar(image, ax=list(flat_axes[:n_residues]), label="ΔF [kcal/mol]")
    fig.subplots_adjust(wspace=0.25, hspace=0.35)
    return fig, axes


def per_residue_ramachandran(*args, **kwargs):
    """Correctly spelled alias for :func:`per_residue_ramachandra`."""
    return per_residue_ramachandra(*args, **kwargs)

