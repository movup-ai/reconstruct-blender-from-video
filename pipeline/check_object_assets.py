"""Reopen and validate generated assets and the Stage 2 export contract in Blender.

blender -b --factory-startup -P pipeline/check_object_assets.py
Reports are written under output/. Optional -- --warm-cache populates ignored export caches.
"""
import argparse
import json
import math
from pathlib import Path
import runpy
import sys
import tempfile

import bpy
import bmesh

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'pipeline'))
import assets_lib


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--warm-cache',action='store_true',help='Save validated exports in the Stage 2 cache')
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
    folder=ROOT/'assets/objects'
    manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    metadata=json.loads((folder/'objects.json').read_text(encoding='utf-8'))
    catalog=assets_lib.catalog({'object_assets':str(folder)})
    assert set(manifest)==set(catalog),'Catalog and asset manifest differ'
    report=[]
    for kind,expected in manifest.items():
        path=folder/expected['file']
        bpy.ops.wm.open_mainfile(filepath=str(path),load_ui=False)
        objects=list(bpy.context.scene.objects)
        assert len(objects)==1 and objects[0].type=='MESH',(kind,'Unexpected objects')
        ob=objects[0]
        assert not ob.hide_render and ob.visible_get(),(kind,'Hidden mesh')
        assert ob.name==kind and ob.get('front_axis')=='-Y' and ob.asset_data is not None
        coords=[ob.matrix_world@v.co for v in ob.data.vertices]
        assert all(math.isfinite(x) for v in coords for x in v),(kind,'Invalid vertex')
        lo=[min(v[i] for v in coords) for i in range(3)]
        hi=[max(v[i] for v in coords) for i in range(3)]
        assert abs(lo[2])<1e-5 and abs(lo[0]+hi[0])<1e-5 and abs(lo[1]+hi[1])<1e-5,(kind,'Origin contract')
        assert all(abs(ob.location[i])<1e-6 and abs(ob.scale[i]-1)<1e-6 for i in range(3))
        assert len(ob.data.polygons)==expected['faces'] and 0<len(ob.data.polygons)<3000
        assert all(abs((hi[i]-lo[i])-expected['size_m'][i])<1e-4 for i in range(3))
        bm=bmesh.new();bm.from_mesh(ob.data)
        assert all(edge.is_manifold for edge in bm.edges),(kind,'Open/nonmanifold solid')
        assert all(face.calc_area()>1e-10 for face in bm.faces),(kind,'Degenerate face')
        bm.free()
        # Use the actual exporter that Stage 2 invokes, then its normaliser/placer.
        with tempfile.TemporaryDirectory() as tmp:
            export=Path(tmp)/'asset.json'
            sys.argv=['blender','--',str(export)]
            runpy.run_path(str(ROOT/'pipeline/asset_export.py'),run_name='__main__')
            raw=json.loads(export.read_text(encoding='utf-8'))
            model=assets_lib.normalise(raw)
        assert len(model['f'])==expected['faces']
        placed=assets_lib.place(model,[1,2,3],.3,model['size'])
        assert all(math.isfinite(float(x)) for v in placed for x in v)
        assert catalog[kind]['on_floor']==metadata[kind]['on_floor']
        if args.warm_cache:
            stat=path.stat()
            model.update(stamp=[stat.st_size,stat.st_mtime_ns],type=kind,source=path.name,objects=raw['objects'])
            cache=folder/'_cache';cache.mkdir(exist_ok=True)
            (cache/(kind+'.json')).write_text(json.dumps(model),encoding='utf-8')
        report.append({'type':kind,'faces':len(model['f']),'size_m':model['size'],'passed':True})
    output=ROOT/'output/object_asset_validation.json'
    output.write_text(json.dumps({'passed':True,'count':len(report),'assets':report},indent=2),encoding='utf-8')
    print('ASSET_VALIDATION_PASSED',len(report),flush=True)


if __name__=='__main__':main()
