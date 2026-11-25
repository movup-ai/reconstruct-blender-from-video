"""Append and drive the complete supplied L3 hierarchy without scale deformation."""
import json
import hashlib
import math
from pathlib import Path

ASSETS = Path(__file__).resolve().parents[1] / 'assets/l3'


def descriptor():
    return json.loads((ASSETS / 'character.json').read_text(encoding='utf-8'))


def append_actor(bpy, kind, joints, destination, material, height):
    from mathutils import Matrix
    asset = descriptor()
    if hashlib.sha256((ASSETS / 'L3_standard.blend').read_bytes()).hexdigest() != asset['template_sha256']:
        raise RuntimeError('L3 template and fitting descriptor differ; rebuild with build_l3_asset.py')
    with bpy.data.libraries.load(str(ASSETS / 'L3_standard.blend'), link=False) as (_, target):
        target.collections = ['L3_STANDARD']
    collection = target.collections[0]
    if collection is None:
        raise RuntimeError('L3_STANDARD collection is missing')
    destination.children.link(collection)
    objects = list(collection.all_objects)
    # Blender library loading remaps all internal constraints and modifiers to this
    # actor's newly appended IDs. Resolve names before assigning actor prefixes.
    find = lambda name: next(o for o in objects if o.name == name or o.name.startswith(name+'.'))
    root, native = find('L3_ACTOR_ROOT'), find('L3_SOURCE_RIG')
    # "L3" must be resolved by type: other IDs may have the same prefix.
    mesh = next(o for o in objects if o.type == 'MESH' and o.get('model_standard') == asset['version'])
    for obj in objects:
        obj.name = kind + '_' + obj.name
        if obj.animation_data:
            obj.animation_data_clear()
    collection.name = kind + '_L3_STANDARD'
    root.parent = joints['root']
    root.location = (0., 0., 0.)
    root.rotation_euler = (0., 0., math.pi)
    scale = height / 1.75 * asset['scale_to_1_75']
    root.scale = (scale,) * 3
    mesh.data.materials[0] = material
    # Keep the template's direct neutral emission material and all 17 mark faces.
    bpy.context.view_layer.objects.active = native
    native.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    for bone in asset['bones'].values():
        native.data.edit_bones[bone['bone']].use_connect = False
    bpy.ops.object.mode_set(mode='OBJECT')
    native.select_set(False)
    for joint, bone in asset['bones'].items():
        pose = native.pose.bones[bone['bone']]
        for constraint in list(pose.constraints):
            pose.constraints.remove(constraint)
        target = bpy.data.objects.new(kind + '_' + joint + '_L3_driver', None)
        destination.objects.link(target)
        target.parent = joints[joint]
        target.rotation_mode = 'QUATERNION'
        target.rotation_quaternion = Matrix(bone['R']).to_quaternion()
        target.empty_display_size = .03
        for constraint_type in ('COPY_LOCATION', 'COPY_ROTATION'):
            constraint = pose.constraints.new(constraint_type)
            constraint.target = target
            constraint.target_space = constraint.owner_space = 'WORLD'
    return mesh


def validation_frames(data, mode='normal'):
    frames=int(data['frames'])
    if mode=='audit':return list(range(1,frames+1))
    selected={1,frames,(frames+1)//2}
    for shot in data.get('shots',[]):
        a,b=shot['frames'];selected.update((a+1,b+1))
    return sorted(f for f in selected if 1<=f<=frames)


def cached_validation(bpy,data,work):
    import validation as V
    policy=V.mode(data)
    key=V.digest({'scene':V.geometry(data,people_only=True),'assets':V.rig_assets(data),'mode':policy,
        'blender':bpy.app.version_string,'implementation':V.implementation(('l3_blender.py','blender_build.py'))})
    return V.cached_report(Path(work)/'l3_geometry.json',key,lambda:validate_actors(bpy,data,policy))


def validate_actors(bpy, data, mode='audit'):
    """Fail before rendering if rigid dimensions/marks/actor bindings drift."""
    import numpy as np
    profile = json.loads((ASSETS/'model-profile.json').read_text(encoding='utf-8'))
    edges = np.asarray(profile['rigid_reference_edges'],float)
    a,b = edges[:,:2].astype(int).T
    scene = bpy.context.scene; saved = scene.frame_current; reports=[]
    try:
        for name,character in data.get('chars',{}).items():
            if character.get('style') != 'l3':continue
            mesh = next(o for o in scene.objects if o.type == 'MESH' and o.name.startswith(name+'_') and o.get('model_standard') == profile['version'])
            if len(mesh.data.vertices)!=profile['vertices'] or len(mesh.data.polygons)!=profile['polygons']:
                raise ValueError(name+': L3 topology changed')
            if [p.index for p in mesh.data.polygons if p.material_index==1]!=profile['white_face_indices']:
                raise ValueError(name+': white mark faces changed')
            material=mesh.data.materials[1]
            emission=next((n for n in material.node_tree.nodes if n.type=='EMISSION'),None) if material and material.use_nodes else None
            output=next((n for n in material.node_tree.nodes if n.type=='OUTPUT_MATERIAL' and n.is_active_output),None) if emission else None
            if emission is None or output is None or not any(link.from_node==emission and link.to_node==output for link in material.node_tree.links):
                raise ValueError(name+': direct white emission missing')
            if any(abs(v-1)>1e-6 for v in emission.inputs['Color'].default_value) or abs(emission.inputs['Strength'].default_value-1)>1e-6:
                raise ValueError(name+': neutral white material changed')
            bind=next(m.object for m in mesh.modifiers if m.type=='ARMATURE')
            if len(bind.data.bones)!=10:
                raise ValueError(name+': rigid binding changed')
            for bone in bind.pose.bones:
                constraints=list(bone.constraints)
                if {c.type for c in constraints if not c.mute} != {'COPY_LOCATION','COPY_ROTATION'} or any(c.influence!=1 or not c.target or not c.target.name.startswith(name+'_') or c.subtarget!=bone.name for c in constraints):
                    raise ValueError(name+': rigid constraint or actor target changed')
                source=constraints[0].target.pose.bones[bone.name]
                if any(not c.target or not c.target.name.startswith(name+'_') for c in source.constraints):
                    raise ValueError(name+': animation constraint points to another actor')
            fixed={}; max_error=0.
            checked_frames=validation_frames(data,mode)
            for frame in checked_frames:
                scene.frame_set(frame);bpy.context.view_layer.update()
                nodes=[mesh,bind];parent=mesh.parent
                while parent:
                    nodes.append(parent);parent=parent.parent
                nodes.append(bind.pose.bones[0].constraints[0].target)
                for node in nodes:
                    scale=tuple(node.scale)
                    if min(scale)<=0 or max(scale)-min(scale)>1e-5 or (node.name in fixed and max(abs(x-y) for x,y in zip(fixed[node.name],scale))>1e-5):
                        raise ValueError(name+': nonuniform or animated scale at frame '+str(frame))
                    fixed[node.name]=scale
                evaluated=mesh.evaluated_get(bpy.context.evaluated_depsgraph_get())
                vertices=np.empty(profile['vertices']*3)
                evaluated.data.vertices.foreach_get('co',vertices)
                vertices=vertices.reshape(-1,3) @ np.array(evaluated.matrix_world.to_3x3()).T
                scale=abs(mesh.parent.matrix_world.to_scale().x)
                error=float(np.max(np.abs(np.linalg.norm(vertices[a]-vertices[b],axis=1)/scale-edges[:,2])))
                max_error=max(max_error,error)
                if error>2e-4:
                    raise ValueError(name+': rigid body/mark dimensions changed at frame '+str(frame))
            reports.append({'actor':name,'frames':[1,int(data['frames'])],'checked_frames':checked_frames, 'white_faces':17,
                            'max_edge_error_model_units':max_error,'fixed_uniform_scale':True})
    finally:
        scene.frame_set(saved)
    return {'passed':True,'actors':reports,'mode':mode,'exhaustive':mode=='audit',
        'scope':('Full-frame' if mode=='audit' else 'Sampled-frame')+' rigid geometry, white marks and actor-specific binding; visual source fidelity not established.'}
