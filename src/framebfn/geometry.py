"""Backbone geometry extraction."""

from pathlib import Path
from typing import Union

import mdtraj as md
import numpy as np

PathLike = Union[str, Path]


def _load_trajectory(trajectory, topology: PathLike | None = None) -> md.Trajectory:
    if isinstance(trajectory, md.Trajectory):
        if topology is not None:
            raise ValueError("topology must be omitted for an mdtraj.Trajectory")
        return trajectory

    path = Path(trajectory)
    if path.is_dir():
        xtc = path / "samples.xtc"
        pdb = path / "topology.pdb"
        if not xtc.exists() or not pdb.exists():
            raise FileNotFoundError(
                "A trajectory directory must contain samples.xtc and topology.pdb"
            )
        return md.load(xtc, top=pdb)

    return md.load(path, top=topology) if topology is not None else md.load(path)


def extract_backbone_angles(
    trajectory,
    topology: PathLike | None = None,
    *,
    degrees: bool = True,
) -> np.ndarray:
    """Extract per-residue phi, psi and omega angles.

    Parameters
    ----------
    trajectory:
        An ``mdtraj.Trajectory``, a trajectory/PDB path, or a BioEmu output
        directory containing ``samples.xtc`` and ``topology.pdb``.
    topology:
        Topology path for formats such as XTC that do not contain topology.
    degrees:
        Return degrees when true, radians otherwise.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n_frames, n_residues, 3)``. The last dimension is
        ``(phi, psi, omega_to_next_residue)``. Undefined terminal or missing
        angles are represented by NaN.
    """
    traj = _load_trajectory(trajectory, topology)
    angles = np.full((traj.n_frames, traj.n_residues, 3), np.nan, dtype=np.float32)

    computations = (
        (0, md.compute_phi, 1),
        (1, md.compute_psi, 1),
        (2, md.compute_omega, 1),
    )
    for output_index, compute, residue_atom_position in computations:
        atom_indices, values = compute(traj)
        if degrees:
            values = np.rad2deg(values)
        for angle_index, atoms in enumerate(atom_indices):
            residue_index = traj.topology.atom(
                int(atoms[residue_atom_position])
            ).residue.index
            angles[:, residue_index, output_index] = values[:, angle_index]

    return angles

