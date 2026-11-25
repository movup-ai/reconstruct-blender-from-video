"""Check rebuilt L3 rigid blocks and camera against numerical scene data in Blender."""
import json
from pathlib import Path
import sys
import math
import bpy
import numpy as np
from mathutils import Euler

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'pipeline'))
from rig import Rig, JOINTS
from l3_blender import descriptor


def validate(blend,scene_path,report):
    bpy.ops.wm.open_mainfile(filepath=blend,load_ui=False,use_scripts=False)
    data=json.loads(Path(scene_path).read_text(encoding='utf-8'));asset=descriptor()
    maximum=0.;camera_error=0.
    for frame in range(data['frames']):
        bpy.context.scene.frame_set(frame+1);bpy.context.view_layer.update()
        for name,ch in data['chars'].items():
            if ch['style']!='l3':continue
            rig=Rig(ch['height'],'l3');root=ch['root'][frame]
            posed=rig.fk(root[:3],root[3],{j:ch['rot'][j][frame] for j in JOINTS},ch['pelvis_z'][frame])
            for joint,part in asset['parts'].items():
                obj=bpy.data.objects[name+'_'+joint+'_geo_l3']
                actual=np.array([list(obj.matrix_world@v.co) for v in obj.data.vertices])
                rotation,location=posed[joint]
                expected=np.array(part['v'])*ch['height']/1.75 @ rotation.T+location
                maximum=max(maximum,float(np.linalg.norm(actual-expected,axis=1).max()))
        camera=bpy.context.scene.camera
        transform=data['cam'][frame]
        rotation=np.array(Euler([math.radians(v) for v in transform[3:]],'XYZ').to_matrix())
        location=np.array(transform[:3])
        camera_error=max(camera_error,float(np.abs(np.array(camera.matrix_world.to_3x3())-rotation).max()),
                         float(np.abs(np.array(camera.location)-location).max()))
        zoom=(data.get('zoom') or [1.]*data['frames'])[frame]
        assert abs(camera.data.lens-data['lens']*zoom)<.001
    assert maximum<1e-4,(maximum,'Rebuilt geometry differs from fit')
    assert camera_error<1e-5,(camera_error,'Rebuilt camera differs from fit')
    white=[m for m in bpy.data.materials if m.name.startswith('FRONT_MARK_PURE_WHITE') and m.users]
    assert white and all(any(n.type=='EMISSION' for n in m.node_tree.nodes) for m in white)
    result={'passed':True,'frames':data['frames'],'max_vertex_error_m':maximum,
            'max_camera_matrix_error':camera_error,'white_emission_preserved':True,'camera_zoom_preserved':True}
    Path(report).write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print('RECONSTRUCTION_CHECK_PASSED',json.dumps(result))


if __name__=='__main__':
    validate(*sys.argv[sys.argv.index('--')+1:])
