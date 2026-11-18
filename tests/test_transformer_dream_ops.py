import unittest

import torch

from transformer_dream import bootstrap  # noqa: F401
from transformer_dream.image_ops import augment, decode_patches, patchify, tv_loss, validate_size


class TestTransformerDreamOps(unittest.TestCase):
    def test_patchify_decode_roundtrip(self):
        x = torch.rand(1, 3, 256, 256)
        patches = patchify(x)
        self.assertEqual(tuple(patches.shape), (1, 256, 1536))
        y = decode_patches(patches[0], 16, 16)
        self.assertEqual(tuple(y.shape), (1, 3, 256, 256))
        self.assertLess(float((x - y).abs().max()), 1e-5)

    def test_tv_zero_on_flat_image(self):
        x = torch.full((1, 3, 128, 128), 0.5)
        self.assertAlmostEqual(float(tv_loss(x)), 0.0, places=6)

    def test_augment_keeps_shape_and_range(self):
        x = torch.rand(1, 3, 256, 256)
        y = augment(x, 0)
        self.assertEqual(tuple(y.shape), (1, 3, 256, 256))
        self.assertGreaterEqual(float(y.min()), 0.0)
        self.assertLessEqual(float(y.max()), 1.0)

    def test_size_validation(self):
        self.assertEqual(validate_size(256), 256)
        self.assertEqual(validate_size(448), 448)
        with self.assertRaises(ValueError):
            validate_size(250)
        with self.assertRaises(ValueError):
            validate_size(512)


if __name__ == "__main__":
    unittest.main()
