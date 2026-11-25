"""Content-based stage dependencies. Prop and set edits never invalidate human fitting."""
import hashlib
import json
import os
from pathlib import Path

from common import DEFAULT_CFG, jload, jsave

VERSION = 4  # supplied L3 geometry, stable actor scale and source-color backgrounds
FIT_KEYS = ('lens_mm', 'person_height', 'character_style', 'pose_mode', 'pose_engine', 'camera_backend',
            'da3_model', 'da3_meters_per_unit', 'da3_max_views', 'foot_contact', 'metric_scene',
            'sam3_checkpoint', 'sam3_mask_side', 'sam3_probability', 'sam3_max_objects',
            'sam3d_checkpoint', 'sam3d_mhr', 'sam3d_stride', 'sam3d_surface_side', 'fit_step',
            'smooth_frames', 'pose3d_weight', 'pose3d_sigma', 'metric_max_views', 'metric_resolution',
            'max_people', 'camera_estimator', 'lens_auto', 'motion_prior', 'previs_standard',
            'temporal_refinement', 'source_exposure')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=lambda x: x.tolist(),
                                     separators=(',', ':')).encode()).hexdigest()


def read(path, default=None):
    try:
        return jload(str(path))
    except (OSError, ValueError):
        return default


def stamp(path):
    try:
        s = os.stat(path)
        return [os.path.abspath(path), s.st_size, s.st_mtime_ns]
    except (OSError, TypeError):
        return None


def inputs(video, c, work, marks, out, review=None, overrides=None):
    work, marks, out = Path(work), Path(marks), Path(out)
    rv = review if review is not None else read(marks / 'review.json', {})
    rv = rv or {}
    fixes = sorted(marks.glob('checked*.json'), key=lambda p: p.stat().st_mtime)
    overrides = overrides or {}
    checked = overrides.get('checked', read(fixes[-1], {}) if fixes else {}) or {}
    checked = {k: v for k, v in checked.items() if k != 'saved'}
    hints_path = marks / 'hints.json' if (marks / 'hints.json').exists() else out / 'hints.json'
    hints = overrides.get('hints', read(hints_path, {})) or {}
    c = dict(c)
    for key in ('lens_mm','person_height','smooth_frames','fit_step'):
        if key in hints: c[key] = hints[key]
    if rv.get('camera',{}).get('lens'): c['lens_mm'] = float(rv['camera']['lens'])
    fit_hints = {k: v for k, v in hints.items() if k not in
                 ('scene_justifications', 'held_objects', 'light', 'exposure', 'palette', 'vegetation_prototypes')}
    from rig import ASSET_FILES
    style = hints.get('character_style') or c.get('character_style', DEFAULT_CFG['character_style'])
    asset = Path(__file__).resolve().parents[1] / 'assets' / ASSET_FILES[style] if style in ASSET_FILES else None
    fit = digest({'version': VERSION, 'video': stamp(video), 'pose': overrides.get('pose', read(marks / 'pose.json')),
                  'masks': stamp(marks / 'sam3_masks.npz'), 'checked': checked,
                  'people': rv.get('people', {}), 'camera': rv.get('camera', {}), 'hints': fit_hints,
                  'settings': {k: c.get(k, DEFAULT_CFG.get(k)) for k in FIT_KEYS},
                  'rig_definition': stamp(asset),
                  'weights': [stamp(c.get(k, DEFAULT_CFG.get(k))) for k in
                              ('sam3_checkpoint', 'sam3d_checkpoint', 'sam3d_mhr')]})
    env = rv.get('environment', {})
    environment = digest({'version': 6, 'fit': fit, 'objects': read(marks / 'env_ai.json'),
                          'hand_set': read(out / 'env_claude.json'),
                          'settings': {k: c.get(k, DEFAULT_CFG.get(k)) for k in
                                       ('set_builder', 'set_detail', 'set_adaptive_frames', 'set_max_frames',
                                        'sam_points', 'sam3_image_max_masks', 'sam3_vegetation_prompts', 'vegetation_prototypes',
                                        'protect_people', 'scene_verify', 'scene_repair', 'set_colours')},
                          'review': {k: v for k, v in env.items() if k != 'props'}})
    return {'fit': fit, 'environment': environment,
            'limb_depth': digest(rv.get('limb_depth', [])),
            'props': digest({'timeline': rv.get('prop_timeline', []), 'legacy': env.get('props', []),
                             'mode': rv.get('prop_mode', 'manual' if rv.get('prop_timeline') else 'auto'),
                             'auto': c.get('held_objects', True), 'hint': hints.get('held_objects')})}


def record(video, c, work, marks, out):
    state = inputs(video, c, work, marks, out)
    jsave(os.path.join(work, 'build_state.json'), state)
    return state


def adopt_existing(video, c, work, marks, out):
    """Capture a legacy scene BEFORE a review edit, retaining safe human caches during migration."""
    path = Path(work) / 'build_state.json'
    if path.exists():
        return
    scene = read(Path(work) / 'scene_data.json', {}) or {}
    pose = read(Path(marks) / 'pose.json', {}) or {}
    if scene.get('human_backend') != 'sam3d_body' or pose.get('tracker_backend') != 'sam3':
        return
    from body_constraints import VERSION as BODY_VERSION
    if any(ch.get('body_constraints',{}).get('version') != BODY_VERSION for ch in scene.get('chars',{}).values()):
        return  # these scenes used incompatible floor/hip assumptions; refit them once
    hints = read(Path(marks)/'hints.json',{}) or {}
    style = hints.get('character_style') or c.get('character_style',DEFAULT_CFG['character_style'])
    if any(ch.get('style') != style for ch in scene.get('chars',{}).values()):
        return  # a render-style swap cannot reuse a fit made with another skeleton
    state = inputs(video, c, work, marks, out)
    state['environment'] = None  # rebuild old geometry with the new protection/prototype rules once
    jsave(str(path), state)
