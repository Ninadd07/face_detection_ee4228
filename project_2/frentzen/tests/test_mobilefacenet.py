import numpy as np
import pytest
import torch

from mobilefacenet_face.recognizers.mobilefacenet import MobileFaceNetEmbedder


def test_mobilefacenet_wrapper_returns_normalized_embeddings():
    embedder = MobileFaceNetEmbedder(device=torch.device("cpu"))
    face = np.zeros((112, 96, 3), dtype=np.uint8)
    embeddings = embedder.embed([face, face])
    assert embeddings.shape == (2, 128)
    norms = np.linalg.norm(embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4)


def test_mobilefacenet_checkpoint_empty_file_raises_clear_error(tmp_path):
    checkpoint_path = tmp_path / "empty_checkpoint.pth"
    checkpoint_path.write_bytes(b"")

    with pytest.raises(ValueError, match="checkpoint is empty"):
        MobileFaceNetEmbedder(device=torch.device("cpu"), checkpoint_path=str(checkpoint_path))
