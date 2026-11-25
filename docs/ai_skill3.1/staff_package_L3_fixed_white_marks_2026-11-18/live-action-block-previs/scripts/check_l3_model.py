"""Run in a separate Blender process. Read-only input; writes a JSON report."""
import bpy,sys,json,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from l3_checks import validate_scene
p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--report',required=True);p.add_argument('--first',type=int);p.add_argument('--last',type=int)
a=p.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
source=Path(a.input).resolve();report=Path(a.report).resolve()
if report==source or report.exists():raise FileExistsError('Choose a new report path.')
bpy.ops.wm.open_mainfile(filepath=str(source),load_ui=False)
try:result=validate_scene(a.first,a.last)
except Exception as e:
    report.parent.mkdir(parents=True,exist_ok=True);report.write_text(json.dumps(dict(passed=False,error=str(e)),ensure_ascii=False,indent=2),encoding='utf-8');raise
report.parent.mkdir(parents=True,exist_ok=True);report.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print('L3_MODEL_CHECK_PASSED',flush=True)
