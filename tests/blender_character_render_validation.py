"""Compare evaluated Blender geometry with production occlusion protection.

blender -b -P tests/blender_character_render_validation.py -- PROJECT.blend SCENE.json GEOMETRY.json
python tests/blender_character_render_validation.py GEOMETRY.json SCENE.json REPORT.json
The Blender phase needs no OpenCV installation. Both phases write only the chosen output.
"""
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'pipeline'))
from render_settings import engine


def export_geometry(blend,scene_path,output):
    import bpy
    data=json.loads(Path(scene_path).read_text(encoding='utf-8'))
    bpy.ops.wm.open_mainfile(filepath=blend,load_ui=False,use_scripts=False)
    scene=bpy.context.scene;requested=engine(data,data)
    assert scene.render.engine==('BLENDER_WORKBENCH' if requested=='workbench' else 'BLENDER_EEVEE')
    shape=(90,160);rows=[]
    for frame in sorted({0,(data['frames']-1)//2,data['frames']-1}):
        scene.frame_set(frame+1);bpy.context.view_layer.update()
        camera=scene.camera;matrix=np.array(camera.matrix_world)
        C=matrix[:3,3];R=np.diag([1.,-1.,-1.])@matrix[:3,:3].T
        focal=camera.data.lens/camera.data.sensor_width*shape[1]
        K=np.array([[focal,0,shape[1]/2],[0,focal,shape[0]/2],[0,0,1.]])
        parts=[];graph=bpy.context.evaluated_depsgraph_get()
        for name,ch in data['chars'].items():
            if not ch.get('visible',[True]*data['frames'])[frame]:continue
            objects=[o for o in scene.objects if o.type=='MESH' and o.name.startswith(name+'_') and not o.hide_render]
            if ch.get('style')=='l3':objects=[o for o in objects if o.get('model_standard')=='L3-2026-11-18']
            assert objects,'Character is missing from Blender: '+name
            for obj in objects:
                evaluated=obj.evaluated_get(graph);mesh=evaluated.to_mesh()
                try:
                    coordinates=np.empty(len(mesh.vertices)*3);mesh.vertices.foreach_get('co',coordinates)
                    transform=np.array(evaluated.matrix_world)
                    world=coordinates.reshape(-1,3)@transform[:3,:3].T+transform[:3,3]
                    parts.append({'v':world.tolist(),'f':[list(p.vertices) for p in mesh.polygons]})
                finally:evaluated.to_mesh_clear()
        rows.append({'frame':frame,'K':K.tolist(),'R':R.tolist(),'C':C.tolist(),'parts':parts})
    Path(output).write_text(json.dumps({'engine':requested,'shape':shape,'frames':rows}),encoding='utf-8')


def validate(geometry,scene_path,output):
    from geometry_protection import rendered_people_depth
    from scene_verify import rasterize
    data=json.loads(Path(scene_path).read_text());source=json.loads(Path(geometry).read_text())
    shape=tuple(source['shape']);rows=[]
    for row in source['frames']:
        K,R,C=[np.asarray(row[k]) for k in ('K','R','C')];frame=row['frame'];actual=np.full(shape,np.inf)
        for part in row['parts']:np.minimum(actual,rasterize(part['v'],part['f'],K,R,C,shape),out=actual)
        predicted=rendered_people_depth(data,frame,K,R,C,shape)
        a,p=np.isfinite(actual),np.isfinite(predicted);overlap=a&p
        iou=float(overlap.sum()/max(1,(a|p).sum()))
        error=float(np.max(np.abs(actual[overlap]-predicted[overlap]))) if overlap.any() else float('inf')
        assert iou>.99,(frame,'silhouette disagreement',iou)
        assert error<.01,(frame,'depth disagreement',error)
        rows.append({'frame':frame,'silhouette_iou':iou,'max_depth_error_m':error})
    report={'passed':True,'engine':source['engine'],'actors':len(data['chars']),'frames':rows}
    Path(output).write_text(json.dumps(report,indent=2),encoding='utf-8');print('CHARACTER_PROTECTION_VALIDATED',report)


if __name__=='__main__':
    if '--' in sys.argv:export_geometry(*sys.argv[sys.argv.index('--')+1:])
    else:validate(*sys.argv[1:])
