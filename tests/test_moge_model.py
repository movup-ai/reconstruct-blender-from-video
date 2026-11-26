import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'pipeline')]

import moge_model
import lens_auto


class MoGeChoice(unittest.TestCase):
    def test_version_names_and_infer_arguments(self):
        self.assertEqual(moge_model.version({}), "v2")
        self.assertEqual(moge_model.version({"moge_version": "V3"}), "v3")
        self.assertEqual(moge_model.version({"moge_version": "v9"}), "v2", "an unknown version falls back")
        self.assertEqual(moge_model.name({"moge_version": "v3"}), "MoGe-3")
        self.assertEqual(moge_model.infer_kwargs("v2", {"moge_refine_steps": 5}), {})
        self.assertEqual(moge_model.infer_kwargs("v3", {}), {"refine_steps": 3})
        self.assertEqual(moge_model.infer_kwargs("v3", {"moge_refine_steps": 1}), {"refine_steps": 1})

    def test_missing_weights_are_reported(self):
        with patch.dict(moge_model.WEIGHTS, {"v3": os.path.join(tempfile.gettempdir(), "no_such_moge3.pt")}):
            why = moge_model.missing({"moge_version": "v3"})
        self.assertIn("MoGe-3 weights missing", why)

    def test_lens_cache_is_kept_per_moge_version(self):
        frames = [np.zeros((40, 60, 3), np.uint8)] * 8
        with tempfile.TemporaryDirectory() as work, patch.object(moge_model, "missing", return_value=None), \
                patch.object(lens_auto, "measure", return_value=[60.0] * 3) as measure:
            lens_auto.clip_lens(frames, 0, 40, [(0, 8)], work, {"moge_version": "v2"})
            lens_auto.clip_lens(frames, 0, 40, [(0, 8)], work, {"moge_version": "v2"})
            self.assertEqual(measure.call_count, 1, "same version: cached")
            lens, info = lens_auto.clip_lens(frames, 0, 40, [(0, 8)], work, {"moge_version": "v3"})
            self.assertEqual(measure.call_count, 2, "another version measures again")
            self.assertEqual(info["model"], "MoGe-3")
            self.assertAlmostEqual(lens, lens_auto.lens_from_fov(60.0), places=1)


if __name__ == "__main__":
    unittest.main()
