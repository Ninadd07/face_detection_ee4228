import numpy as np

from mobilefacenet_face.alignment import FaceAligner


def test_aligner_returns_expected_output_shape():
    image = np.zeros((160, 160, 3), dtype=np.uint8)
    landmarks = np.array(
        [[40, 50], [80, 50], [60, 70], [45, 90], [75, 90]],
        dtype=np.float32,
    )
    aligner = FaceAligner(image_size=(112, 96))
    aligned = aligner.align(image, landmarks)
    assert aligned.shape == (112, 96, 3)
