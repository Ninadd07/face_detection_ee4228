from pathlib import Path

from mobilefacenet_face.dataset import create_split_manifest, infer_group_id


def test_infer_group_id_uses_frame_pattern():
    path = Path("Abhiram/aug_vid0_frame00011_fog_0033.jpg")
    assert infer_group_id(path) == "vid0_frame00011"


def test_create_split_manifest_preserves_groups(tmp_path):
    root = tmp_path / "data"
    for identity in ("alice", "bob"):
        person = root / identity
        person.mkdir(parents=True)
        for suffix in ("fog", "rain"):
            (person / f"aug_vid0_frame00001_{suffix}_0001.jpg").write_bytes(b"test")
        (person / "aug_vid0_frame00002_clear_0001.jpg").write_bytes(b"test")
        (person / "aug_vid0_frame00003_clear_0001.jpg").write_bytes(b"test")
    records = create_split_manifest(root, 0.5, 0.25, 0.25, None, seed=42)
    group_splits = {}
    for record in records:
        key = (record.identity, record.group_id)
        group_splits.setdefault(key, set()).add(record.split)
    assert all(len(splits) == 1 for splits in group_splits.values())
