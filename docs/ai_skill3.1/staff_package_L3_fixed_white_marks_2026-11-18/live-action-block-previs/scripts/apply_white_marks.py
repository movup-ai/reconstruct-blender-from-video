"""Run in a NEW Blender background process, input -> distinct output."""
import bpy, sys, argparse, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from white_marks import apply_white_marks

parser = argparse.ArgumentParser()
parser.add_argument('--input', required=True)
parser.add_argument('--output', required=True)
args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else [])
source = Path(args.input).resolve()
output = Path(args.output).resolve()
if output == source or output.exists():
    raise FileExistsError('Choose a new output file; original is preserved.')
bpy.ops.wm.open_mainfile(filepath=str(source))
changes = apply_white_marks()
# Existing embedded build scripts may contain old cyan template paths. Keep them
# as historical records, never pretend that running them recreates this revision.
for text in bpy.data.texts:
    if text.name in ['Construction script', 'Production script']:
        text.name = 'HISTORICAL_before_white_fix_' + text.name
text = bpy.data.texts.new('Current mark correction')
text.write('2026-11-18: eye/front stripe and chest stripe changed to neutral white. '
           'Existing motion/camera/geometry retained. Historical scripts are not current rebuild procedures.\n'
           + json.dumps(changes, ensure_ascii=False, indent=2))
bpy.context.scene['white_mark_revision'] = '2026-11-18 neutral emission RGB 1,1,1, global exposure retained'
output.parent.mkdir(parents=True, exist_ok=True)
bpy.context.preferences.filepaths.save_version = 0
bpy.ops.wm.save_as_mainfile(filepath=str(output))
output.with_suffix('.white-check.json').write_text(json.dumps(dict(input_name=source.name, output_name=output.name, material_srgb=[255,255,255], changed=changes), ensure_ascii=False, indent=2), encoding='utf-8')
print('WHITE_MARKS_SAVED', str(output), len(changes), flush=True)
