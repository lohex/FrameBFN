import mdtraj as md
import numpy as np

from framebfn import extract_backbone_angles


def test_extract_backbone_angles_shape_and_terminals():
    topology = md.Topology()
    chain = topology.add_chain()
    for _ in range(3):
        residue = topology.add_residue("ALA", chain)
        topology.add_atom("N", md.element.nitrogen, residue)
        topology.add_atom("CA", md.element.carbon, residue)
        topology.add_atom("C", md.element.carbon, residue)
        topology.add_atom("O", md.element.oxygen, residue)
    atoms = list(topology.atoms)
    for i in range(3):
        n, ca, c, o = atoms[4 * i : 4 * i + 4]
        topology.add_bond(n, ca)
        topology.add_bond(ca, c)
        topology.add_bond(c, o)
        if i:
            topology.add_bond(atoms[4 * (i - 1) + 2], n)
    xyz = np.arange(2 * 12 * 3, dtype=float).reshape(2, 12, 3) / 100
    trajectory = md.Trajectory(xyz, topology)

    result = extract_backbone_angles(trajectory)

    assert result.shape == (2, 3, 3)
    assert np.isnan(result[:, 0, 0]).all()
    assert np.isnan(result[:, -1, 1]).all()
    assert np.isnan(result[:, -1, 2]).all()

