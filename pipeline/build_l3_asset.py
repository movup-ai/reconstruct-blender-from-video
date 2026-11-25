"""Build the numeric fitting descriptor from the supplied L3 (run with Blender).

blender --background --factory-startup --disable-autoexec --python pipeline/build_l3_asset.py
Only this explicit build step reads the documentation package. Runtime uses assets/l3.
"""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile


def main():
    import bpy
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / 'pipeline'))
    import extract_human_master as extractor
    source = root / 'docs/ai_skill3.1/staff_package_L3_fixed_white_marks_2026-11-18/live-action-block-previs/assets'
    template = source / 'L3_standard_fixed_white_marks.blend'
    bpy.ops.wm.open_mainfile(filepath=str(template), load_ui=False, use_scripts=False)
    mesh = bpy.data.objects['L3']
    native = bpy.data.objects['L3_SOURCE_RIG']
    bind = bpy.data.objects['L3_RIGID_BIND']
    bpy.context.view_layer.update()
    groups = {g.index: g.name for g in mesh.vertex_groups}
    owners = {}
    for vertex in mesh.data.vertices:
        weights = [(groups[g.group], g.weight) for g in vertex.groups if g.weight > 1e-6]
        if len(weights) != 1 or abs(weights[0][1] - 1.) > 1e-6:
            raise ValueError('L3 requires one rigid weight per vertex')
        owners[vertex.index] = weights[0][0]
    dump = {'blender': bpy.app.version_string, 'pieces': {}, 'bones': {}}
    for bone in bind.data.bones:
        indices = [i for i, owner in owners.items() if owner == bone.name]
        remap = {old: new for new, old in enumerate(indices)}
        polygons = [p for p in mesh.data.polygons if all(v in remap for v in p.vertices)]
        dump['pieces'][bone.name] = {'bone': bone.name,
            'v': [list(mesh.matrix_world @ mesh.data.vertices[i].co) for i in indices],
            'f': [[remap[i] for i in p.vertices] for p in polygons],
            'mat': [p.material_index for p in polygons], 'smooth': any(p.use_smooth for p in polygons)}
    if sum(len(p['f']) for p in dump['pieces'].values()) != len(mesh.data.polygons):
        raise ValueError('A face crosses rigid body parts')
    for bone in native.pose.bones:
        if bone.name.startswith(('DEF-spine', 'DEF-upper_arm', 'DEF-forearm', 'DEF-hand', 'DEF-thigh', 'DEF-shin', 'DEF-foot')):
            matrix = native.matrix_world @ bone.matrix
            # Native object scale must affect joint positions, never the rotation basis.
            rotation = matrix.to_quaternion().to_matrix().to_4x4()
            rotation.translation = matrix.translation
            dump['bones'][bone.name] = {'head': list(matrix.translation), 'matrix': [list(row) for row in rotation]}
    target = root / 'assets/l3'
    target.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temporary:
        dump_path = Path(temporary) / 'dump.json'
        dump_path.write_text(json.dumps(dump), encoding='utf-8')
        previous = sys.argv
        try:
            sys.argv = ['extract', str(dump_path), str(target / 'character.json')]
            extractor.main()
        finally:
            sys.argv = previous
    path = target / 'character.json'
    data = json.loads(path.read_text(encoding='utf-8'))
    data.update(source='Supplied L3-2026-11-18, fixed rigid white marks', version='L3-2026-11-18',
                template_sha256=hashlib.sha256(template.read_bytes()).hexdigest())
    path.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
    shutil.copyfile(template, target / 'L3_standard.blend')
    shutil.copyfile(source / 'model-profile.json', target / 'model-profile.json')
    shutil.copyfile(source / 'semantic-palette.json', target / 'semantic-palette.json')
    print('L3_ASSET_BUILT', target)


if __name__ == '__main__':
    main()
