"""Supported model flow, saved-setting migration, and SAM 3 scene-mask contracts."""
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'pipeline')]
import automark
import common
from agent import config
import run_pipeline as RP
import sam3_scene as SCENE
import sam3_tracking as SAM


class ModelFlowTests(unittest.TestCase):
    def test_old_config_migrates_without_changing_folders_or_fitting_weights(self):
        saved = {'tracker_backend': 'geometry', 'human_backend': 'motionbert', 'pose_backend': 'rtmo',
                 'rtmo_model': 'retired.onnx', 'detect_stride': 'auto', 'pose3d': True,
                 'pose3d_weight': .7, 'source_root': 'my footage', 'sam3_checkpoint': 'approved.pt'}
        migrated = common.normalize_backends(saved)
        for key, value in common.FIXED_BACKENDS.items():
            self.assertEqual(migrated[key], value)
        self.assertTrue(set(common.RETIRED_SETTINGS).isdisjoint(migrated))
        self.assertEqual(migrated['pose3d_weight'], .7)
        self.assertEqual(migrated['source_root'], 'my footage')
        self.assertEqual(migrated['sam3_checkpoint'], 'approved.pt')
        self.assertEqual(saved['human_backend'], 'motionbert', 'migration must not mutate its input')

    def test_app_and_cli_load_and_save_the_same_supported_backends(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, 'config.json')
            path.write_text(json.dumps({'human_backend': 'motionbert', 'rtmo_model': 'old.onnx',
                                        'sam3_checkpoint': 'local.pt', 'deepseek_key': 'private-local-value'}))
            with patch.object(common, 'CFG_PATH', str(path)), patch.object(config, 'CFG_PATH', str(path)):
                self.assertEqual(common.cfg()['human_backend'], 'sam3d_body')
                self.assertEqual(config.load()['human_backend'], 'sam3d_body')
                config.save({'sam3d_stride': 1})
            result = json.loads(path.read_text())
            self.assertEqual(result['sam3_checkpoint'], 'local.pt')
            self.assertEqual(result['deepseek_key'], 'private-local-value')
            self.assertNotIn('rtmo_model', result)
            self.assertEqual(result['tracker_backend'], 'sam3')

    def test_retired_selectors_cannot_be_saved_from_the_settings_api(self):
        for key in ('tracker_backend', 'human_backend', 'pose_backend', 'rtmo_model'):
            with self.assertRaisesRegex(ValueError, 'cannot be changed'):
                config.save({key: 'retired'})

    def test_body_assets_follow_the_configured_checkpoint_folder(self):
        paths = dict(common.human_model_assets({'sam3d_checkpoint': os.path.join('custom', 'model.ckpt')}))
        self.assertEqual(paths['SAM 3D Body config'], os.path.join('custom', 'model_config.yaml'))
        self.assertEqual(paths['SAM 3 checkpoint'], common.DEFAULT_CFG['sam3_checkpoint'])

    def test_marker_loads_only_the_joint_model_and_does_not_require_a_detector(self):
        # This fake library deliberately has no YOLOX or RTMO attributes.
        library = types.ModuleType('rtmlib')
        library.Body = types.SimpleNamespace(MODE={'performance': {
            'pose': 'pose-download', 'pose_input_size': (4, 4)}})
        library.RTMPose = type('RTMPose', (), {})
        files = types.ModuleType('rtmlib.tools.file')
        files.download_checkpoint = Mock(return_value='pose.onnx')
        session = Mock()
        session.get_providers.return_value = ['CUDAExecutionProvider']
        ort = Mock()
        ort.get_available_providers.return_value = ['CUDAExecutionProvider']
        modules = {'rtmlib': library, 'rtmlib.tools': types.ModuleType('rtmlib.tools'),
                   'rtmlib.tools.file': files}
        with patch.dict(sys.modules, modules), patch.object(common, 'gpu_dlls', return_value=ort), \
             patch.object(common, 'gpu_session', return_value=(session, 'cuda')) as gpu:
            marker = automark.Marker(engine='cuda')
        self.assertEqual(marker.pose.onnx_model, 'pose.onnx')
        files.download_checkpoint.assert_called_once_with('pose-download')
        self.assertEqual(gpu.call_count, 1)
        self.assertFalse(hasattr(marker, 'det'))
        session.disable_fallback.assert_called_once()

    def test_fresh_marks_preserve_two_mask_ids_even_with_identical_joint_observations(self):
        from test_sam3_tracking import FakePose, bundle
        masks = bundle([(f, p) for f in range(4) for p in (2, 1)])
        frames = [np.full((20, 40, 3), 200, np.uint8)] * 4
        args = types.SimpleNamespace(mode='performance', device='cuda', engine='cuda', min_track=4, step=2)
        with tempfile.TemporaryDirectory() as tmp:
            def worker(*arguments):
                np.savez_compressed(Path(arguments[4], 'sam3_masks.npz'), **masks)
                return masks
            with patch.object(automark, 'read_frames', return_value=(frames, 30.)), \
                 patch.object(SAM, 'run_masks', side_effect=worker):
                automark._generate_marks('clip.mp4', tmp, args, FakePose(), {})
            pose = json.loads(Path(tmp, 'pose.json').read_text())
            self.assertEqual({p['sam3_id'] for p in pose['people']}, {1, 2})
            self.assertEqual(len(pose['people']), 2)
            self.assertTrue(all(len(p['frames']) == 4 for p in pose['people']))
            self.assertTrue(Path(tmp, 'check.html').is_file())

    def test_background_cache_invalidates_a_retired_segmentation_result(self):
        old = {'frames': 4, 'lens': 35., 'cam': [[0] * 6] * 4}
        current = dict(old, segmentation={'backend': 'sam3', 'version': 1})
        self.assertFalse(RP.same_set_inputs(old, current))
        self.assertTrue(RP.same_set_inputs(current, dict(current)))
        self.assertFalse(RP.same_set_inputs(None, current))


class SceneMaskTests(unittest.TestCase):
    def test_quality_filter_rejects_empty_full_unstable_and_nonfinite_masks(self):
        logits = np.full((5, 4, 4), -3., np.float32)
        logits[:, :2] = 3.
        logits[1] = -3.; logits[2] = 3.; logits[3] = .2; logits[4, 0, 0] = np.nan
        masks, scores = SCENE.grid_candidates(logits, np.ones(5) * .9, .7, .86, .002)
        self.assertEqual(masks.shape, (1, 4, 4))
        self.assertEqual(int(masks.sum()), 8)
        self.assertGreater(scores[0], 1.)

    def test_mask_suppression_keeps_disjoint_objects_and_highest_scored_duplicate(self):
        masks = np.zeros((3, 8, 8), bool)
        masks[0, :4, :4] = True; masks[1] = masks[0]; masks[2, 4:, 4:] = True
        kept = SCENE.suppress_masks(masks, [.8, .9, .7], (16, 16))
        self.assertEqual(kept.shape, (2, 16, 16))
        self.assertFalse((kept[0] & kept[1]).any())
        self.assertEqual(len(SCENE.suppress_masks(masks, [.8, .9, .7], (16, 16), limit=1)), 1)

    def test_empty_masks_keep_output_dimensions(self):
        self.assertEqual(SCENE.suppress_masks([], [], (10, 20)).shape, (0, 10, 20))

    def test_prompted_boxes_clip_to_image_and_reject_degenerate_geometry(self):
        np.testing.assert_allclose(SCENE.valid_boxes([[-4, 0, 40, 20]], 30, 15), [[0, 0, 30, 15]])
        for box in ([3, 4, 2, 8], [0, 0, np.nan, 5], [40, 0, 50, 10]):
            with self.assertRaises(ValueError):
                SCENE.valid_boxes([box], 30, 15)

    def test_scene_cache_reuses_validated_output_and_invalidates_changed_grid(self):
        image = np.zeros((10, 20, 3), np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint, python = Path(tmp, 'sam3.pt'), Path(tmp, 'python.exe')
            checkpoint.touch(); python.touch()
            c = {'sam3_checkpoint': str(checkpoint), 'sam3_python': str(python)}
            def worker(command, **kwargs):
                np.savez_compressed(command[-1], masks=np.zeros((0, 10, 20), bool),
                                    box_masks=np.ones((1, 10, 20), bool), peak_vram_mb=123.)
            release = Mock()
            with patch.object(SCENE.subprocess, 'run', side_effect=worker) as run:
                for _ in range(2):
                    output = SCENE.segment(image, c, boxes=[[1, 1, 5, 8]], cache_dir=tmp, before_inference=release)
                self.assertTrue(output['box_masks'].all())
                self.assertEqual(run.call_count, 1)
                release.assert_called_once()
                SCENE.segment(image, c, boxes=[[1, 1, 5, 8]], cache_dir=tmp, points=16)
                self.assertEqual(run.call_count, 2)

    def test_worker_failure_does_not_publish_a_cached_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint, python = Path(tmp, 'sam3.pt'), Path(tmp, 'python.exe')
            checkpoint.touch(); python.touch()
            c = {'sam3_checkpoint': str(checkpoint), 'sam3_python': str(python)}
            with patch.object(SCENE.subprocess, 'run', side_effect=RuntimeError('GPU failure')):
                with self.assertRaisesRegex(RuntimeError, 'GPU failure'):
                    SCENE.segment(np.zeros((10, 20, 3), np.uint8), c, cache_dir=tmp)
            self.assertFalse(list(Path(tmp).glob('*.npz')))


if __name__ == '__main__':
    unittest.main()
