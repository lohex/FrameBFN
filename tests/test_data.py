from framebfn import data


def test_load_dyndb_trajectory_matches_sequence(monkeypatch, tmp_path):
    rows = [
        {"PDB_ID": "AAAA", "Sequence_y(fixed)": "AAAA", "seq_length": 4},
        {"PDB_ID": "1ABC", "Sequence_y(fixed)": "PEPTIDE", "seq_length": 7},
    ]
    sentinel = object()
    monkeypatch.setattr(data, "load_dataset", lambda *args, **kwargs: rows)

    def fake_download(**kwargs):
        assert kwargs["filename"] == "trajectory_frames/1ABC_frame.pdb"
        return tmp_path / "1ABC_frame.pdb"

    monkeypatch.setattr(data, "hf_hub_download", fake_download)
    monkeypatch.setattr(data.md, "load_pdb", lambda path: sentinel)

    trajectory, metadata = data.load_dyndb_trajectory(
        "peptide", return_metadata=True
    )
    assert trajectory is sentinel
    assert metadata["PDB_ID"] == "1ABC"

