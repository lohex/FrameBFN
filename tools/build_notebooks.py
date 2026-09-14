"""Generate the small, reviewable Colab notebooks shipped with FrameBFN."""

from pathlib import Path
import json


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"
REPO = "lohex/FrameBFN"


def markdown(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": (text.strip() + "\n").splitlines(keepends=True),
    }


def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": (text.strip() + "\n").splitlines(keepends=True),
    }


def badge(path):
    url = f"https://colab.research.google.com/github/{REPO}/blob/main/{path}"
    return (
        f'<a href="{url}" target="_parent"><img src="https://colab.research.google.com/'
        'assets/colab-badge.svg" alt="Open In Colab"/></a>'
    )


SETUP = r'''
%pip install -q datasets huggingface_hub mdtraj matplotlib

import os
import subprocess
import sys
from pathlib import Path

repo_dir = Path("/content/FrameBFN")
if not repo_dir.exists():
    url = "https://github.com/lohex/FrameBFN.git"
    try:
        from google.colab import userdata
        token = userdata.get("GITHUB_TOKEN")
    except Exception:
        token = None
    if token:
        url = f"https://x-access-token:{token}@github.com/lohex/FrameBFN.git"
    subprocess.run(
        ["git", "clone", "--depth", "1", url, str(repo_dir)],
        check=True,
        stdout=subprocess.DEVNULL,
    )
sys.path.insert(0, str(repo_dir / "src"))
'''


def write(name, title, cells):
    path = f"notebooks/{name}"
    notebook = {
        "cells": [markdown(badge(path) + f"\n\n# {title}")] + cells,
        "metadata": {
            "colab": {"name": name, "provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NOTEBOOKS.mkdir(exist_ok=True)
    (NOTEBOOKS / name).write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


write(
    "01_extract_dyndb_peptides.ipynb",
    "Build a peptide catalogue from DynoDB",
    [
        markdown(
            "Stream DynoDB metadata, select a configurable peptide length range, "
            "save the resulting catalogue to Google Drive, and optionally retrieve "
            "one trajectory by PDB ID or exact peptide sequence."
        ),
        code(SETUP),
        code(r'''
from google.colab import drive
drive.mount("/content/drive")

from datasets import load_dataset
import pandas as pd
from framebfn import load_dyndb_trajectory

MIN_LENGTH = 7
MAX_LENGTH = 15

# Change this path if the project lives elsewhere in My Drive.
DRIVE_PROJECT_DIR = Path("/content/drive/MyDrive/FrameBFN")
DATA_DIR = DRIVE_PROJECT_DIR / "data"
CATALOG_PATH = DATA_DIR / "dyndb_peptides.csv"
DATA_DIR.mkdir(parents=True, exist_ok=True)
'''),
        code(r'''
dataset = load_dataset("renzilin/DynoDB", split="train", streaming=True)
rows = []
for entry in dataset:
    length = int(entry["seq_length"])
    if MIN_LENGTH <= length <= MAX_LENGTH:
        rows.append({
            "pdb_id": entry["PDB_ID"],
            "sequence": entry["Sequence_y(fixed)"],
            "seq_length": length,
        })

peptides = (
    pd.DataFrame(rows)
    .drop_duplicates()
    .sort_values(["seq_length", "sequence", "pdb_id"])
    .reset_index(drop=True)
)
peptides.to_csv(CATALOG_PATH, index=False)
print(f"Saved {len(peptides):,} DynoDB entries to {CATALOG_PATH}")
peptides.head()
'''),
        code(r'''
peptides.groupby("seq_length").size().rename("systems").to_frame()
'''),
        markdown(
            "## Retrieve one trajectory\n\nSet `PEPTIDE_QUERY` to either a PDB ID "
            "or an exact amino acid sequence. If a sequence occurs in multiple "
            "DynoDB rows, the first matching entry is selected."
        ),
        code(r'''
PEPTIDE_QUERY = "1L2Y"  # PDB ID or exact peptide sequence

trajectory, metadata = load_dyndb_trajectory(
    PEPTIDE_QUERY,
    return_metadata=True,
)

pdb_id = str(metadata["PDB_ID"]).strip()
trajectory_dir = DATA_DIR / "dyndb" / pdb_id
trajectory_dir.mkdir(parents=True, exist_ok=True)
trajectory_path = trajectory_dir / f"{pdb_id}_trajectory.pdb"
trajectory.save_pdb(trajectory_path)

print("PDB ID:", pdb_id)
print("Sequence:", metadata["Sequence_y(fixed)"])
print("Frames:", trajectory.n_frames)
print("Residues:", trajectory.n_residues)
print("Saved trajectory to:", trajectory_path)
'''),
    ],
)

write(
    "02_dyndb_trajectory_angles.ipynb",
    "Load a DynoDB peptide trajectory and extract backbone angles",
    [
        markdown(
            "Resolve a sequence or PDB ID against DynoDB, download its multi-frame "
            "PDB trajectory, extract phi, psi and omega, and save reusable arrays."
        ),
        code(SETUP),
        code(r'''
import numpy as np
from framebfn import (
    extract_backbone_angles,
    load_dyndb_trajectory,
    per_residue_ramachandran,
)

PEPTIDE = "1L2Y"  # PDB ID or exact amino-acid sequence
OUTPUT_DIR = repo_dir / "data" / "dyndb" / PEPTIDE.upper()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
'''),
        code(r'''
trajectory, metadata = load_dyndb_trajectory(PEPTIDE, return_metadata=True)
print(metadata["PDB_ID"], metadata["Sequence_y(fixed)"])
print(trajectory)

trajectory.save_pdb(OUTPUT_DIR / "trajectory.pdb")
angles = extract_backbone_angles(trajectory)
np.save(OUTPUT_DIR / "backbone_angles.npy", angles)
print("Angle tensor:", angles.shape, "(frames, residues, phi/psi/omega)")
'''),
        code(r'''
residue_names = [
    f"{res.index + 1}: {res.name}" for res in trajectory.topology.residues
]
fig, axes = per_residue_ramachandran(
    angles,
    residue_names=residue_names,
    free_energy=False,
)
'''),
        code(r'''
fig, axes = per_residue_ramachandran(
    angles,
    residue_names=residue_names,
    free_energy=True,
    temperature=300.0,
)
'''),
    ],
)

write(
    "03_bioemu_peptide_sampling.ipynb",
    "Sample a peptide sequence with BioEmu",
    [
        markdown(
            "Generate a BioEmu conformational ensemble for one sequence, then use "
            "the FrameBFN package to extract and visualize backbone angles. A Colab "
            "GPU runtime is recommended."
        ),
        code(r'''
%pip install -q uv
!uv pip install --system --prerelease if-necessary-or-explicit bioemu
%pip install -q datasets huggingface_hub mdtraj matplotlib

import os
import subprocess
import sys
from pathlib import Path

repo_dir = Path("/content/FrameBFN")
if not repo_dir.exists():
    url = "https://github.com/lohex/FrameBFN.git"
    try:
        from google.colab import userdata
        token = userdata.get("GITHUB_TOKEN")
    except Exception:
        token = None
    if token:
        url = f"https://x-access-token:{token}@github.com/lohex/FrameBFN.git"
    subprocess.run(
        ["git", "clone", "--depth", "1", url, str(repo_dir)],
        check=True,
        stdout=subprocess.DEVNULL,
    )
sys.path.insert(0, str(repo_dir / "src"))
'''),
        code(r'''
import torch
from bioemu.sample import main as sample
from framebfn import extract_backbone_angles, per_residue_ramachandran

SEQUENCE = "INWKGIAAMAKKLL"
NUM_SAMPLES = 1_000
OUTPUT_DIR = Path("/content/bioemu_samples")
OUTPUT_DIR.mkdir(exist_ok=True)

print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
print("Sequence length:", len(SEQUENCE))
'''),
        code(r'''
sample(
    sequence=SEQUENCE,
    num_samples=NUM_SAMPLES,
    output_dir=str(OUTPUT_DIR),
)
'''),
        code(r'''
angles = extract_backbone_angles(OUTPUT_DIR)
np_path = OUTPUT_DIR / "backbone_angles.npy"
import numpy as np
np.save(np_path, angles)
print("Saved", angles.shape, "to", np_path)
'''),
        code(r'''
residue_names = [f"{i + 1}: {aa}" for i, aa in enumerate(SEQUENCE)]
fig, axes = per_residue_ramachandran(
    angles,
    residue_names=residue_names,
    free_energy=True,
)
'''),
    ],
)
