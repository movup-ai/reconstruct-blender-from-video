"""Build the editable, low-poly Stage 2 object library with Blender.

blender -b --factory-startup -P pipeline/build_object_assets.py -- [--overwrite]
Existing assets and user metadata overrides are preserved unless --overwrite is set.
"""
import argparse
import json
import math
from pathlib import Path
import sys

import bpy
from mathutils import Vector

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'assets/objects'
PREVIEWS = ROOT / 'output/object_asset_previews'


def box(name, size, at, dark=False):
    bpy.ops.mesh.primitive_cube_add(size=1, location=at)
    ob=bpy.context.object; ob.name=name; ob.dimensions=size
    ob.data.materials.append(material(dark))
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    return ob


def material(dark=False):
    name='Inset grey' if dark else 'Blockout grey'
    m=bpy.data.materials.get(name)
    if m is None:
        m=bpy.data.materials.new(name); v=.22 if dark else .68
        m.diffuse_color=(v,v,v,1); m.roughness=1
    return m


def cyl(name, radius, height, at, axis='Z', dark=False, vertices=12):
    rotation={'Z':(0,0,0),'X':(0,math.pi/2,0),'Y':(math.pi/2,0,0)}[axis]
    bpy.ops.mesh.primitive_cylinder_add(vertices=vertices,radius=radius,depth=height,location=at,rotation=rotation)
    ob=bpy.context.object;ob.name=name;ob.data.materials.append(material(dark));return ob


def cone(name, r1, r2, height, at, dark=False):
    bpy.ops.mesh.primitive_cone_add(vertices=10,radius1=r1,radius2=r2,depth=height,location=at)
    ob=bpy.context.object;ob.name=name;ob.data.materials.append(material(dark));return ob


def beam(name, a, b, width=.04, depth=None):
    a,b=Vector(a),Vector(b)
    ob=box(name,(width,depth or width,(b-a).length),(a+b)/2)
    ob.rotation_euler=(b-a).to_track_quat('Z','Y').to_euler();return ob


def vessel(name, height, radius, profile=None, handle=False):
    # Connected inner and outer walls, with an actual open rim and closed bottom.
    profile=profile or [(radius*.80,0),(radius,height),(radius*.82,height),(radius*.65,.025)]
    n=12;verts=[(r*math.cos(i*2*math.pi/n),r*math.sin(i*2*math.pi/n),z) for r,z in profile for i in range(n)]
    faces=[]
    for j in range(len(profile)-1):
        for i in range(n):
            a=j*n+i;b=j*n+(i+1)%n;faces.append((a,b,b+n,a+n))
    faces+=[tuple(reversed(range(n))),tuple((len(profile)-1)*n+i for i in range(n))]
    mesh=bpy.data.meshes.new(name);mesh.from_pydata(verts,[],faces);mesh.update()
    ob=bpy.data.objects.new(name,mesh);bpy.context.collection.objects.link(ob);ob.data.materials.append(material())
    if handle:
        bpy.ops.mesh.primitive_torus_add(major_segments=12,minor_segments=4,location=(radius*.95,0,height*.53),rotation=(math.pi/2,0,0),major_radius=height*.24,minor_radius=height*.055)
        bpy.context.object.name='Handle';bpy.context.object.data.materials.append(material())


def legs(width, depth, top, count=4):
    for x in (-width/2+.065,width/2-.065):
        for y in (-depth/2+.065,depth/2-.065):box('Leg',(.07,.07,top),(x,y,top/2))


def table(width=1.3,depth=.75,height=.76):
    legs(width,depth,height-.07);box('Top',(width,depth,.07),(0,0,height-.035))


def sofa(width=2.05,depth=.9,height=.85):
    box('Base',(width,depth,.23),(0,0,.22))
    box('Seat',(width-.22,depth-.15,.16),(0,-.035,.415))
    box('Back',(width,.16,height-.28),(0,depth/2-.08,(height+.28)/2))
    for x in (-width/2+.07,width/2-.07):box('Arm',(.14,depth,.37),(x,0,.505))
    legs(width-.1,depth-.1,.15)


def chair(office=False,stool=False):
    if office:
        cyl('Pedestal',.045,.4,(0,0,.25));cyl('Hub',.11,.07,(0,0,.09))
        for i in range(5):
            a=i*math.tau/5;end=(.31*math.cos(a),.31*math.sin(a),.07)
            beam('Star base',(0,0,.10),end,.035);cyl('Caster',.045,.035,(end[0],end[1],.045),axis='X',dark=True,vertices=8)
    else:legs(.46,.46,.43)
    box('Seat',(.48,.47,.07),(0,0,.465))
    if not stool:box('Back',(.46,.07,.46),(0,.20,.72))


def cabinet(width=1.1,height=1.35,depth=.48,drawers=0):
    box('Body',(width,depth,height),(0,0,height/2))
    if drawers:
        for i in range(drawers):
            z=(i+.5)*height/drawers
            box('Drawer',(width-.035,.025,height/drawers-.035),(0,-depth/2-.015,z))
            box('Handle',(.20,.04,.025),(0,-depth/2-.045,z))
    else:
        for x in (-width/4,width/4):
            box('Door',(width/2-.025,.025,height-.04),(x,-depth/2-.015,height/2))
            box('Handle',(.025,.04,.15),(x*.15,-depth/2-.045,height*.55))


def shelf(width=1.0,height=1.8,depth=.32,levels=5):
    for x in (-width/2+.025,width/2-.025):box('Side',(.05,depth,height),(x,0,height/2))
    box('Back',(width,.025,height),(0,depth/2-.0125,height/2))
    for i in range(levels):box('Shelf',(width,depth,.04),(0,0,.02+i*(height-.04)/(levels-1)))


def frame(width=1,height=.8,depth=.07,mirror=False,window=False):
    for x in (-width/2+.035,width/2-.035):box('Frame side',(.07,depth,height),(x,0,height/2))
    for z in (.035,height-.035):box('Frame rail',(width,depth,.07),(0,0,z))
    box('Panel',(width-.13,.015,height-.13),(0,.015,height/2),True)
    if window:
        box('Mullion',(.035,depth*.75,height),(0,-.005,height/2))
        box('Crossbar',(width,depth*.75,.035),(0,-.005,height/2))


def railing(width=2,height=1,depth=.055):
    for i in range(9):box('Upright',(.035,depth,height),(width*(i/8-.5),0,height/2))
    for z in (.15,height):box('Rail',(width+.05,depth,.045),(0,0,z))


def car(kind='car'):
    w,l,h={'car':(1.8,4.2,1.45),'truck':(2.1,5.6,2.5),'van':(1.95,4.7,2.05),'bus':(2.4,8.0,3.0)}[kind]
    box('Body',(w,l,.65),(0,0,.70))
    if kind=='truck':
        box('Cargo',(w,l*.57,h-.9),(0,l*.19,(h+.9)/2));box('Cab',(w,l*.30,1.25),(0,-l*.33,1.28))
    else:box('Cabin',(w*.86,l*.65,h-.9),(0,.05,(h+.9)/2))
    box('Windshield',(w*.74,.025,.42),(0,-l*(.49 if kind=='truck' else .32),h-.38),True)
    for x in (-w/2,w/2):
        for y in (-l*.31,l*.31):cyl('Wheel',.32,.16,(x,y,.32),axis='X',dark=True)
    for x in (-w*.32,w*.32):box('Headlight',(.25,.03,.15),(x,-l/2-.015,.73),True)


def build(kind):
    if kind in ('sofa','armchair'):sofa(width=1 if kind=='armchair' else 2.05)
    elif kind=='ottoman':box('Cushion',(.70,.55,.28),(0,0,.31));legs(.62,.47,.17)
    elif kind in ('chair','office_chair','stool'):chair(kind=='office_chair',kind=='stool')
    elif kind in ('table','coffee_table','desk','pool_table'):
        table(*{'table':(1.3,.75,.76),'coffee_table':(1.05,.60,.42),'desk':(1.4,.7,.75),'pool_table':(2.2,1.2,.8)}[kind])
        if kind=='desk':box('Drawer pedestal',(.38,.60,.58),(.43,.02,.37))
        if kind=='pool_table':
            for x in (-1.06,1.06):box('Cushion rail',(.08,1.2,.08),(x,0,.84))
            for y in (-.56,.56):box('Cushion rail',(2.2,.08,.08),(0,y,.84))
    elif kind=='bench':table(1.5,.42,.46);box('Back',(1.5,.065,.40),(0,.18,.70))
    elif kind=='bed':
        box('Frame',(1.6,2.1,.22),(0,0,.23));box('Mattress',(1.55,2,.22),(0,0,.45))
        box('Headboard',(1.65,.09,.9),(0,1.02,.45))
        for x in (-.38,.38):box('Pillow',(.62,.38,.10),(x,.68,.61))
    elif kind in ('cabinet','wardrobe','chest_of_drawers','fridge'):
        cabinet(*{'cabinet':(1.1,1.3,.48,0),'wardrobe':(1.5,2.1,.6,0),'chest_of_drawers':(.9,1.05,.48,4),'fridge':(.72,1.8,.7,0)}[kind])
        if kind=='fridge':box('Freezer seam',(.70,.025,.02),(0,-.385,1.35),True)
    elif kind in ('shelf','bookshelf'):shelf(height=.9 if kind=='shelf' else 1.8,levels=3 if kind=='shelf' else 5)
    elif kind in ('counter','kitchen_island'):
        cabinet(1.8,.86,.7);box('Worktop',(1.88,.78,.06),(0,0,.89))
    elif kind=='sink':
        box('Back ledge',(.7,.12,.06),(0,.20,.18));box('Front rim',(.7,.06,.06),(0,-.23,.18))
        for x in (-.32,.32):box('Side rim',(.06,.5,.06),(x,0,.18))
        box('Basin bottom',(.59,.39,.04),(0,-.01,.02));box('Back wall',(.59,.04,.15),(0,.17,.10))
        box('Front wall',(.59,.04,.15),(0,-.19,.10))
        for x in (-.28,.28):box('Basin side',(.04,.38,.15),(x,0,.10))
        cyl('Tap',.018,.23,(.16,.20,.33));beam('Spout',(.16,.20,.43),(.16,.04,.43),.028)
    elif kind in ('stove','oven'):
        cabinet(.65,.9,.62);box('Oven glass',(.49,.025,.40),(0,-.37,.39),True)
        for x in (-.16,.16):
            for y in (-.15,.15):cyl('Burner',.10,.015,(x,y,.91),dark=True)
    elif kind in ('tv','monitor'):
        frame(1.1 if kind=='tv' else .58,.65 if kind=='tv' else .36,.06)
        box('Stand',(.20,.15,.13),(0,0,-.065));box('Foot',(.40,.24,.025),(0,0,-.14))
    elif kind in ('lamp','floor_lamp','table_lamp'):
        h=.6 if kind=='table_lamp' else 1.6
        cyl('Base',.18,.04,(0,0,.02));cyl('Stem',.025,h-.2,(0,0,(h-.2)/2))
        cone('Shade',.24,.13,.27,(0,0,h-.135))
    elif kind=='box':box('Box',(.65,.55,.55),(0,0,.275))
    elif kind in ('bin','barrel'):
        vessel('Container',.85,.30)
        if kind=='barrel':
            for z in (.12,.72):cyl('Hoop',.305,.035,(0,0,z),dark=True)
    elif kind in ('vase','pot','cup'):
        if kind=='vase':vessel('Vase',.45,.15,[(.09,0),(.15,.17),(.10,.31),(.065,.45),(.05,.45),(.085,.31),(.13,.17),(.07,.025)])
        elif kind=='pot':vessel('Plant pot',.35,.23)
        else:vessel('Cup',.105,.044,handle=True)
    elif kind in ('picture','mirror','window'):frame(mirror=kind=='mirror',window=kind=='window')
    elif kind=='door':
        box('Panel',(.88,.06,2.05),(0,0,1.025));cyl('Handle',.03,.07,(.32,-.055,1.0),axis='Y',dark=True)
        for x in (-.49,.49):box('Jamb',(.08,.13,2.15),(x,.015,1.075))
        box('Lintel',(1.06,.13,.08),(0,.015,2.11))
    elif kind in ('curtain','blind'):
        if kind=='curtain':
            for i in range(12):cyl('Fold',.065,2.0,((i-5.5)*.12,0,1),vertices=6)
        else:
            for i in range(14):box('Slat',(1.3,.06,.055),(0,0,.04+i*.075))
    elif kind in ('pillar','pole','streetlight','traffic_light'):
        h=2.6 if kind=='pillar' else 3.2
        cyl('Column',.22 if kind=='pillar' else .055,h,(0,0,h/2))
        if kind=='pillar':
            box('Base',(.55,.55,.12),(0,0,.06));box('Capital',(.55,.55,.12),(0,0,h-.06))
        elif kind in ('streetlight','traffic_light'):
            beam('Arm',(0,0,h),(.5,0,h),.06)
            box('Head',(.35,.22,.13) if kind=='streetlight' else (.22,.18,.65),(.5,0,h-.15),True)
            if kind=='traffic_light':
                for z in (h-.36,h-.15,h+.06):cyl('Signal',.06,.02,(.5,-.105,z),axis='Y',dark=True)
    elif kind=='stairs':
        for i in range(6):box('Step',(1.2,.28,.18*(i+1)),(0,(i-2.5)*.28,.09*(i+1)))
    elif kind in ('railing','fence'):railing(height=1.5 if kind=='fence' else 1.0)
    elif kind=='awning':
        ob=box('Canopy',(2.0,.95,.07),(0,0,.35));ob.rotation_euler.x=math.radians(15)
        box('Valance',(2.0,.07,.20),(0,-.46,.16))
    elif kind in ('car','truck','van','bus'):car(kind)
    elif kind=='wall':box('Wall',(3.0,.18,2.6),(0,0,1.3))
    elif kind=='fireplace':
        for x in (-.5,.5):box('Surround',(.22,.38,1.0),(x,0,.5))
        box('Mantel',(1.35,.48,.12),(0,0,1.06));box('Hearth',(1.35,.60,.10),(0,-.05,.05))
        box('Firebox',(.80,.04,.77),(0,.15,.47),True)
    elif kind=='bathtub':
        box('Bottom',(1.6,.72,.12),(0,0,.06))
        for x in (-.77,.77):box('End',(.10,.8,.52),(x,0,.26))
        for y in (-.35,.35):box('Side',(1.6,.10,.52),(0,y,.26))
    elif kind=='microwave':
        box('Case',(.52,.40,.30),(0,0,.15));box('Glass',(.37,.025,.22),(-.04,-.21,.15),True)
        box('Handle',(.02,.04,.18),(.14,-.24,.15))
    elif kind in ('washer','dishwasher'):
        cabinet(.60,.85,.60)
        if kind=='washer':cyl('Drum window',.20,.035,(0,-.36,.41),axis='Y',dark=True)
        else:box('Grip',(.38,.04,.025),(0,-.37,.73))
    elif kind=='radiator':
        for i in range(10):box('Fin',(.055,.15,.65),((i-4.5)*.07,0,.425))
        for z in (.15,.70):cyl('Pipe',.025,.70,(0,0,z),axis='X')
    elif kind=='shower':
        box('Tray',(.9,.9,.07),(0,0,.035))
        for x in (-.43,.43):
            for y in (-.43,.43):box('Frame',(.035,.035,2.0),(x,y,1.0))
        box('Back',(.90,.04,2.0),(0,.43,1.0));beam('Shower pipe',(.25,.38,1.0),(.25,.38,1.9),.025)
        beam('Head arm',(.25,.38,1.9),(.25,.08,1.9),.025);cyl('Head',.08,.025,(.25,.08,1.88))
    elif kind in ('booth','arcade_machine'):
        cabinet(1.0 if kind=='booth' else .65,2.0,.72)
        box('Screen',(.68 if kind=='booth' else .50,.03,.75),(0,-.43,1.35),True)
        if kind=='arcade_machine':box('Control deck',(.65,.35,.10),(0,-.48,.85))
    elif kind in ('tree','pine_tree'):
        cyl('Trunk',.15,1.2,(0,0,.60),vertices=8)
        for r,h,z in ((1.05,1.5,1.8),(.80,1.35,2.45),(.50,1.2,3.0)):cone('Foliage',r,0,h,(0,0,z))
    elif kind=='bush':
        for x,y,z,s in ((0,0,.45,.65),(-.4,0,.35,.45),(.4,.08,.35,.45)):
            bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2,radius=s,location=(x,y,z))
            bpy.context.object.name='Shrub crown';bpy.context.object.scale.z=.7;bpy.context.object.data.materials.append(material())
    else:raise ValueError(kind)


KINDS = '''sofa armchair ottoman chair office_chair stool table coffee_table pool_table desk bench bed
cabinet wardrobe chest_of_drawers fireplace arcade_machine shelf bookshelf counter kitchen_island sink stove oven fridge
tv monitor lamp floor_lamp table_lamp box bathtub microwave washer dishwasher radiator shower booth bin barrel vase pot cup
door window curtain blind picture mirror pillar pole streetlight traffic_light stairs railing fence awning car truck van bus wall
tree pine_tree bush'''.split()

HANGING={'shelf','sink','tv','monitor','microwave','vase','pot','cup','table_lamp','window','curtain','blind','picture','mirror','awning'}


def save_asset(kind):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    build(kind)
    meshes=[o for o in bpy.context.scene.objects if o.type=='MESH']
    bpy.ops.object.select_all(action='DESELECT')
    for ob in meshes:ob.select_set(True)
    bpy.context.view_layer.objects.active=meshes[0]
    bpy.ops.object.join();ob=bpy.context.object;ob.name=kind;ob.data.name=kind+'_mesh'
    bpy.ops.object.transform_apply(location=True,rotation=True,scale=True)
    pts=[v.co for v in ob.data.vertices]
    low=Vector(tuple(min(v[i] for v in pts) for i in range(3)));high=Vector(tuple(max(v[i] for v in pts) for i in range(3)))
    shift=Vector(((low.x+high.x)/2,(low.y+high.y)/2,low.z))
    for v in ob.data.vertices:v.co-=shift
    bpy.context.scene.unit_settings.system='METRIC'
    bpy.context.scene.unit_settings.scale_length=1
    ob['asset_type']=kind;ob['front_axis']='-Y';ob['units']='metres'
    ob.asset_mark();ob.asset_data.description='Simple editable blockout: '+kind.replace('_',' ')
    size=high-low
    assert min(size)>0 and len(ob.data.polygons)<3000
    bpy.context.preferences.filepaths.save_version=0
    scene=bpy.context.scene;scene.render.engine='BLENDER_WORKBENCH'
    scene.display.shading.light='STUDIO';scene.display.shading.color_type='MATERIAL'
    scene.display.shading.show_shadows=False;scene.display.shading.show_cavity=True
    scene.display.shading.background_type='WORLD';scene.world=bpy.data.worlds.new('Neutral');scene.world.color=(.12,.12,.12)
    # Save the reusable asset before adding a preview camera: files contain meshes only.
    path=OUT/(kind+'.blend')
    bpy.ops.wm.save_as_mainfile(filepath=str(path),compress=True)
    bpy.ops.object.camera_add();cam=bpy.context.object
    target=Vector((0,0,size.z/2));cam.location=target+Vector((4,-6,3.8)).normalized()*max(size)*3
    cam.rotation_euler=(target-cam.location).to_track_quat('-Z','Y').to_euler();cam.data.type='ORTHO';cam.data.ortho_scale=max(size)*1.5
    scene.camera=cam;scene.render.resolution_x=256;scene.render.resolution_y=256;scene.render.resolution_percentage=100
    scene.render.image_settings.file_format='PNG';scene.render.filepath=str(PREVIEWS/(kind+'.png'))
    bpy.ops.render.render(write_still=True)
    return {'size_m':[round(v,4) for v in size],'vertices':len(ob.data.vertices),'faces':len(ob.data.polygons),'file':path.name}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--overwrite',action='store_true')
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
    OUT.mkdir(parents=True,exist_ok=True);PREVIEWS.mkdir(parents=True,exist_ok=True)
    manifest_path=OUT/'manifest.json';manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    metadata_path=OUT/'objects.json';metadata=json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    for kind in KINDS:
        if args.overwrite or not (OUT/(kind+'.blend')).exists():manifest[kind]=save_asset(kind)
        metadata.setdefault(kind,{'prompt':kind.replace('_',' '),'on_floor':kind not in HANGING})
        print('ASSET_READY',kind,flush=True)
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    metadata_path.write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
    print('LIBRARY_READY',len(manifest),flush=True)


if __name__=='__main__':main()
