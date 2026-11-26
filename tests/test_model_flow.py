"""Supported model flow and saved-setting migration."""
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
        self.assertEqual(common.normalize_backends({'character_style': 'l3', 'camera_backend': 'da3'})['character_style'], 'sample')
        self.assertNotIn('camera_backend', common.normalize_backends({'camera_backend': 'da3'}))
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
            self.assertFalse(Path(tmp, 'check.html').exists(), 'Stage 1 writes no joint editor')
            self.assertFalse(Path(tmp, 'review').exists())
            self.assertNotIn('review_frames', pose)

if __name__ == '__main__':
    unittest.main()
