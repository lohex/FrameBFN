# FrameBFN

Utilities and reproducible Colab workflows for peptide conformational data used by
FrameBFN.

## Colab notebooks

| Workflow | Colab |
| --- | --- |
| Extract a length-filtered DynoDB peptide catalogue to `data/dyndb_peptides.csv` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lohex/FrameBFN/blob/main/notebooks/01_extract_dyndb_peptides.ipynb) |
| Download one DynoDB trajectory and extract backbone angles | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lohex/FrameBFN/blob/main/notebooks/02_dyndb_trajectory_angles.ipynb) |
| Generate a peptide ensemble with BioEmu | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lohex/FrameBFN/blob/main/notebooks/03_bioemu_peptide_sampling.ipynb) |

Because the repository is private, Colab users can add a `GITHUB_TOKEN` with read
access under **Secrets**. The notebooks also work without that secret if the repository
is later made public.

## Package API

```python
from framebfn import (
    extract_backbone_angles,
    load_dyndb_trajectory,
    per_residue_ramachandran,
)

trajectory = load_dyndb_trajectory("1L2Y")
angles = extract_backbone_angles(trajectory)
fig, axes = per_residue_ramachandran(angles, free_energy=True)
```

`extract_backbone_angles` accepts an `mdtraj.Trajectory`, a trajectory path with an
optional topology path, or a BioEmu output directory. Its output is shaped as
`(frames, residues, 3)` with phi, psi and omega in degrees. Undefined terminal angles
are `NaN`.

## Local installation

```bash
python -m pip install -e ".[dev]"
pytest
```

The notebooks are generated from `tools/build_notebooks.py`; rerun that script after
changing their shared setup or structure.
