"""Neutral white integrated face stripes; callable from Blender scene scripts."""
import bpy

def apply_white_marks():
    """Change only the material slots used by integrated head/torso marks.

    Neutral emission avoids cyan spill from corridor lighting. Scene exposure can
    still dim the marks in dark source frames. No extra mesh or light is added.
    """
    material = bpy.data.materials.get('FRONT_MARK_PURE_WHITE')
    if material is None:
        material = bpy.data.materials.new('FRONT_MARK_PURE_WHITE')
    material.diffuse_color = (1, 1, 1, 1)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new('ShaderNodeOutputMaterial')
    emission = nodes.new('ShaderNodeEmission')
    emission.inputs['Color'].default_value = (1, 1, 1, 1)
    emission.inputs['Strength'].default_value = 1
    material.node_tree.links.new(emission.outputs['Emission'], output.inputs['Surface'])
    material['display_srgb'] = '255,255,255'
    material['dark_scene_policy'] = 'Follows global exposure; neutral equal RGB, no cyan light tint.'
    changes = []
    for obj in bpy.data.objects:
        if obj.type != 'MESH':
            continue
        is_l3 = obj.get('model_standard') == 'L3-2026-11-18'
        if not is_l3 and obj.get('bone') not in ('head', 'torso'):
            continue
        for slot in range(len(obj.data.materials)):
            old = obj.data.materials[slot]
            if old and ((is_l3 and slot == 1) or old.name.startswith('FRONT_MARK') or old.name == 'MARK_PURE_WHITE'):
                polygons = [p for p in obj.data.polygons if p.material_index == slot]
                if not polygons:
                    continue
                obj.data.materials[slot] = material
                changes.append(dict(object=obj.name, bone=obj.get('bone','L3 integrated head/chest'), slot=slot, faces=len(polygons)))
    if not changes:
        raise RuntimeError('No integrated head/chest mark material found; scene was not modified.')
    return changes
