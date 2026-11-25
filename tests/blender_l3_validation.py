"""Run in Blender against a generated L3 .blend to verify fit/render agreement."""
import json
from pathlib import Path
import sys

import bpy
from mathutils import Vector
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'pipeline'))
from l3_blender import descriptor


def validate(blend, scene_path, report):
    bpy.ops.wm.open_mainfile(filepath=str(blend),load_ui=False,use_scripts=False)
    data=json.loads(Path(scene_path).read_text(encoding='utf-8'))
    asset=descriptor(); reports=[]
    for name,ch in data['chars'].items():
        if ch['style']!='l3':continue
        mesh=next(o for o in bpy.context.scene.objects if o.type=='MESH' and o.name.startswith(name+'_') and o.get('model_standard')==asset['version'])
        bind=next(m.object for m in mesh.modifiers if m.type=='ARMATURE')
        actors={o for o in bpy.context.scene.objects if o.name.startswith(name+'_')}
        for bone in bind.pose.bones:
            for constraint in bone.constraints:
                assert constraint.target in actors,'Cross-actor rigid binding'
        for bone in asset['bones'].values():
            native=bind.pose.bones[bone['bone']].constraints[0].target
            assert all(c.target in actors for c in native.pose.bones[bone['bone']].constraints),'Cross-actor animation driver'
        owners={v.index:mesh.vertex_groups[next(g.group for g in v.groups if g.weight>.99)].name for v in mesh.data.vertices}
        maximum=0.
        for frame in range(1,data['frames']+1):
            bpy.context.scene.frame_set(frame);bpy.context.view_layer.update()
            evaluated=mesh.evaluated_get(bpy.context.evaluated_depsgraph_get())
            for joint,part in asset['parts'].items():
                vertices=[i for i,owner in owners.items() if owner==asset['bones'][joint]['bone']]
                matrix=bpy.data.objects[name+'_'+joint].matrix_world
                for index,local in zip(vertices,part['v']):
                    expected=matrix @ Vector([x*ch['height']/1.75 for x in local])
                    actual=evaluated.matrix_world @ evaluated.data.vertices[index].co
                    maximum=max(maximum,(expected-actual).length)
        assert maximum<1e-4,('Fitted joints and actual L3 render disagree',name,maximum)
        reports.append({'actor':name,'frames':data['frames'],'max_numeric_to_render_vertex_error_m':maximum,'constraint_targets_isolated':True})
    Path(report).write_text(json.dumps({'passed':True,'actors':reports},indent=2)+'\n',encoding='utf-8')
    print('L3_RENDER_GEOMETRY_PASSED',json.dumps(reports))


if __name__=='__main__':
    validate(*sys.argv[sys.argv.index('--')+1:])
