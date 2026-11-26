"""Validate a new L3 fit using cached SAM observations/cameras, in a separate folder.

python pipeline/previs_validate.py --work EXISTING_WORK --marks MARKS --output NEW_FOLDER
Reads existing results without changing them. This is a fitting/asset regression,
not a fresh tracking, camera estimation, or background reconstruction benchmark.
"""
import argparse
import copy
from pathlib import Path

import numpy as np

from common import cfg, jload, jsave, read_frames
from build_env import cam_mats
from camera_solve import Camera
from ground_solve import World
from fit_people import fit_all
import quality
from run_pipeline import coco_people, slice_pose, widen


def validate(work, marks, output):
    work, marks, output = Path(work), Path(marks), Path(output)
    if output.exists():
        raise FileExistsError('Choose a new validation output folder')
    output.mkdir(parents=True)
    original = jload(str(work / 'scene_data.json'))
    info = jload(str(work / 'clip_info.json'))
    pose = jload(str(marks / 'pose.json'))
    review = jload(str(marks/'review.json')) if (marks/'review.json').exists() else {}
    hints = jload(str(marks/'hints.json')) if (marks/'hints.json').exists() else {}
    settings = dict(cfg(), character_style='l3', _review=review, _hints=hints,
                    _work=str(output), _fps=info['fps'])
    settings['_hints'] = dict(hints, character_style='l3')
    data = copy.deepcopy(original)
    data['name'] = 'l3_validation'
    data['package_name'] = 'l3-1'
    data['work_dir'] = str(output.resolve())
    data['submit_dir'] = str((output/'SUBMIT/l3_validation').resolve())
    data['chars'] = {}; data['props'] = []
    data['previs_standard'] = 'l3'
    data['background_policy'] = 'source'
    shots = original.get('shots') or [{'frames': [0, original['frames']-1]}]
    for i, shot in enumerate(shots):
        first, last = shot['frames']; end = last+1
        tag = '_s%d' % (i+1) if len(shots) > 1 else ''
        transforms = [cam_mats(c) for c in original['cam'][first:end]]
        rotations = [np.diag([1.,-1.,-1.]) @ r.T for r,c in transforms]
        positions = np.asarray([c for r,c in transforms])
        camera = Camera(*original['res'], original['lens'])
        zoom = (original.get('zoom') or [1.]*original['frames'])[first:end]
        world = World(camera, [r @ rotations[0].T for r in rotations],
                      rotations[0] @ [0.,0.,1.], float(positions[0,2]), zoom=zoom)
        world.M = rotations[0]; world.C = positions[0]; world.T = positions-positions[0]
        body_path = work / ('sam3d_body%s.npz' % tag)
        with np.load(body_path, allow_pickle=False) as body:
            settings['_sam3d_body'] = {k:body[k] for k in body.files}
        people = coco_people(slice_pose(pose, first, end), None)
        characters, _ = fit_all(world, info['y0'], people, end-first, settings)
        for name, character in characters.items():
            character['source_range'] = [first,last]
            data['chars'][(tag.strip('_')+'_' if tag else '')+name] = widen(character, first, end, original['frames'])
    data['required_track_ids'] = sorted(settings.get('_required_tracks',set()),key=str)
    import source_exposure
    source_exposure.attach(data,info['video'],str(output),settings,pose,info)
    data['render_scale'] = min(1., 960./data['res'][0])
    import colour_card
    colour_card.apply(data, log=None)
    report = {'scope': 'Cached tracking/body/camera observations; newly fitted L3 characters. Props omitted; cached set retained.',
              'baseline': quality.alignment_audit(original, pose, info['y0']),
              'l3': quality.alignment_audit(data, pose, info['y0']),
              'motion_quality': quality.measure(data, pose, info['y0'])}
    jsave(str(output/'scene_data.json'), data)
    jsave(str(output/'validation.json'), report)
    quality.require_alignment(report['l3'])
    print('L3_VALIDATION_SAVED', output)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work', required=True); p.add_argument('--marks',required=True); p.add_argument('--output',required=True)
    a = p.parse_args()
    validate(a.work, a.marks, a.output)
