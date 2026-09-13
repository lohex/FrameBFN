"""DynoDB dataset access."""

from pathlib import Path
from typing import Any

import mdtraj as md
from datasets import load_dataset
from huggingface_hub import hf_hub_download

DYNODB_REPO_ID = "renzilin/DynoDB"


def _normalise(value: Any) -> str:
    return "" if value is None else str(value).strip().upper()


def load_dyndb_trajectory(
    peptide: str,
    *,
    repo_id: str = DYNODB_REPO_ID,
    split: str = "train",
    cache_dir: str | Path | None = None,
    streaming: bool = True,
    return_metadata: bool = False,
):
    """Find a DynoDB peptide by sequence or PDB ID and load its trajectory.

    DynoDB stores trajectories as multi-model PDB files named
    ``trajectory_frames/{PDB_ID}_frame.pdb``. If several rows match a
    sequence, the first dataset row is used.

    Parameters
    ----------
    peptide:
        Peptide amino-acid sequence or DynoDB PDB ID.
    return_metadata:
        When true, return ``(trajectory, row)``.
    """
    query = _normalise(peptide)
    if not query:
        raise ValueError("peptide must be a non-empty sequence or PDB ID")

    dataset = load_dataset(
        repo_id,
        split=split,
        streaming=streaming,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
    )
    match = None
    for row in dataset:
        sequence = _normalise(row.get("Sequence_y(fixed)") or row.get("Sequence"))
        pdb_id = _normalise(row.get("PDB_ID"))
        if query in {sequence, pdb_id}:
            match = row
            break

    if match is None:
        raise LookupError(f"No DynoDB entry found for {peptide!r}")

    pdb_id = str(match["PDB_ID"]).strip()
    path = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=f"trajectory_frames/{pdb_id}_frame.pdb",
        cache_dir=str(cache_dir) if cache_dir is not None else None,
    )
    trajectory = md.load_pdb(path)
    return (trajectory, match) if return_metadata else trajectory

