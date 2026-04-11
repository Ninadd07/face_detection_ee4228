import numpy as np
import pytest

from mobilefacenet_face.gallery import EmbeddingGallery


def test_gallery_match_threshold_rejects_low_scores():
    gallery = EmbeddingGallery.from_templates(
        {
            "alice": [np.array([1.0, 0.0], dtype=np.float32)],
            "bob": [np.array([0.0, 1.0], dtype=np.float32)],
        }
    )
    result = gallery.match(np.array([0.2, 0.2], dtype=np.float32), threshold=0.9)
    assert result.accepted is False
    assert result.identity is None


def test_gallery_match_returns_top_identity():
    gallery = EmbeddingGallery.from_templates(
        {"alice": [np.array([1.0, 0.0], dtype=np.float32)]}
    )
    result = gallery.match(np.array([1.0, 0.0], dtype=np.float32), threshold=0.1)
    assert result.accepted is True
    assert result.identity == "alice"


def test_gallery_match_dimension_mismatch_raises_clear_error():
    gallery = EmbeddingGallery(
        identities=["alice"],
        embeddings=np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
    )
    with pytest.raises(ValueError, match="Gallery embedding dimension mismatch"):
        gallery.match(np.array([1.0, 0.0], dtype=np.float32), threshold=0.1)
