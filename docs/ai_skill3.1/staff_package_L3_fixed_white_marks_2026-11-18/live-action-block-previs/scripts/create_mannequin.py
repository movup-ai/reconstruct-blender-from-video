"""Create a new L3 project from the supplied production asset; no old V5 geometry."""
import bpy,sys,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
out=Path(a.output).expanduser().resolve()
if out.exists():raise FileExistsError('Refusing to overwrite '+str(out))
bpy.ops.wm.open_mainfile(filepath=str(ROOT/'assets/L3_standard_fixed_white_marks.blend'),load_ui=False)
sys.path.insert(0,str(ROOT/'scripts'))
from l3_checks import validate_scene
validate_scene(1,1)
out.parent.mkdir(parents=True,exist_ok=True);bpy.context.preferences.filepaths.save_version=0;bpy.ops.wm.save_as_mainfile(filepath=str(out));print('MANNEQUIN_CREATED_L3',str(out),flush=True)
