"""Blender-only L3 geometry and mark checks, without modifying the scene."""
import bpy,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PROFILE=json.loads((ROOT/'assets/model-profile.json').read_text(encoding='utf-8'))

def validate_scene(first=None,last=None):
    scene=bpy.context.scene
    actors=[o for o in scene.objects if o.type=='MESH' and o.get('model_standard')==PROFILE['version']]
    if not actors:raise RuntimeError('No current L3 standard actor found.')
    first=scene.frame_start if first is None else first;last=scene.frame_end if last is None else last
    original_frame=scene.frame_current;expected=PROFILE['rigid_reference_edges'];expected_white=PROFILE['white_face_indices'];reports=[]
    for ob in actors:
        if ob.data.shape_keys:raise RuntimeError(ob.name+': shape keys can change fixed proportions.')
        if len(ob.data.vertices)!=PROFILE['vertices'] or len(ob.data.polygons)!=PROFILE['polygons']:raise RuntimeError(ob.name+': topology differs from standard L3.')
        if [p.index for p in ob.data.polygons if p.material_index==1]!=expected_white:raise RuntimeError(ob.name+': integrated white faces changed.')
        armatures=[m.object for m in ob.modifiers if m.type=='ARMATURE']
        if len(armatures)!=1 or armatures[0] is None or len(armatures[0].data.bones)!=10:raise RuntimeError(ob.name+': rigid bind is missing or replaced.')
        bind=armatures[0]
        for pb in bind.pose.bones:
            types={c.type for c in pb.constraints if not c.mute}
            if types!={'COPY_LOCATION','COPY_ROTATION'}:raise RuntimeError(ob.name+': rigid location/rotation constraints changed.')
            if any(c.target is None for c in pb.constraints):raise RuntimeError('Missing source rig target.')
        actor_root=ob.parent
        if actor_root is None:raise RuntimeError(ob.name+': actor root missing.')
        max_error=0.;root_scale=None;constant_scales={}
        try:
            for frame in range(first,last+1):
                scene.frame_set(frame);bpy.context.view_layer.update();chain=[ob,bind];ancestor=actor_root
                while ancestor:chain.append(ancestor);ancestor=ancestor.parent
                chain.append(next(c.target for c in bind.pose.bones[0].constraints))
                for node in chain:
                    scales=tuple(node.scale)
                    if max(scales)-min(scales)>1e-5 or min(scales)<=0:raise RuntimeError(f'{ob.name}: non-uniform/negative scale at {node.name}, frame {frame}')
                    if node.name in constant_scales and max(abs(a-b) for a,b in zip(scales,constant_scales[node.name]))>1e-5:raise RuntimeError(f'{ob.name}: scale changes at {node.name}, frame {frame}')
                    constant_scales[node.name]=scales
                scale=abs(actor_root.matrix_world.to_scale().x)
                if root_scale is None:root_scale=scale
                if abs(scale-root_scale)>1e-5:raise RuntimeError('Animated actor placement scale.')
                ev=ob.evaluated_get(bpy.context.evaluated_depsgraph_get())
                for a,b,length in expected:
                    measured=(ev.matrix_world@ev.data.vertices[a].co-ev.matrix_world@ev.data.vertices[b].co).length/scale
                    error=abs(measured-length);max_error=max(max_error,error)
                    if error>2e-4:raise RuntimeError(f'{ob.name}: segment/mark changed size, frame {frame}, error {error}')
                material=ob.data.materials[1]
                if material is None or not material.use_nodes:raise RuntimeError('White material missing.')
                emission=next((n for n in material.node_tree.nodes if n.type=='EMISSION'),None)
                output=next((n for n in material.node_tree.nodes if n.type=='OUTPUT_MATERIAL' and n.is_active_output),None)
                if emission is None or output is None or not any(link.from_node==emission and link.to_node==output for link in material.node_tree.links):raise RuntimeError('White emission is not directly connected to output.')
                if any(abs(x-1)>1e-6 for x in emission.inputs['Color'].default_value) or abs(emission.inputs['Strength'].default_value-1)>1e-6:raise RuntimeError('White RGB/strength changed.')
        finally:scene.frame_set(original_frame)
        reports.append(dict(actor=ob.name,frames=[first,last],max_edge_length_error=max_error,uniform_fixed_root_scale=root_scale,white_faces=17))
    return dict(standard=PROFILE['version'],passed=True,actors=reports,scope='Fixed edge lengths and mark topology/material across frames. Does not validate camera, source motion, contact or final render pixels.')
