# reconstruction/ — Scene data and reconstruction entry point

| File | Description |
| --- | --- |
| `scene.json` | Actual scene data (schema `ba_whitemodel_scene` 1.0), exported by `pipeline/scene_export.py` in the same Blender process after saving the main project |
| `reconstruct.py` | Reads `scene.json`, rebuilds the shot in Blender, and saves a new project |
| `README.md` | This guide |

## Contents of scene.json

- **Scene basis**: `scene` stores meters, Blender's right-handed coordinate system (+Z up, XYZ Euler angles in radians), frame range, frame rate, resolution, active camera, and render settings including color management and world color.
- **Objects and hierarchy**: `collections` records collections and their parents; `objects` records names, types, collections, parents, rest transforms, visibility, and parent inverse matrices.
- **Geometry and appearance**: all mesh vertices, faces, material slots per face, and smoothing flags; `materials` maps material names to colors. No textures are needed.
- **Characters and motion**: `characters` records labels, colors, heights, root and joint empties, and mesh parts. Root position and heading, pelvis height, and XYZ joint rotations are animated. Mesh parts follow the corresponding joints; white head and chest marker faces use `WM_marker` material.
- **Animation**: `animation` stores keys as object → property → channel → `[frame, value, interpolation]`, including characters, cameras, visibility changes, and moving props.
- **Camera**: `camera` records the camera object, focal length, sensor width, and shot ranges. Position and rotation have keys every frame.
- **Lighting**: render settings, world data, and supported light objects are exported when present. Workbench scenes use studio lighting.

## Usage

Use Blender; the saved version is in the `exporter` field of `scene.json`. No plugins, network access, or additional dependencies are required.

```sh
blender -b --factory-startup -P reconstruct.py -- scene.json rebuilt.blend
```

Output: `rebuilt.blend`, a new file. Keep it separate from the main project at `model_3d/{package}.blend`.

## Known limitations

- Reconstruction restores exported objects, geometry, material colors, hierarchy, and animation. Rig constraints, drivers, and armature bones are not reconstructed; use the main project to retain those controls. Non-scene data such as Blender's UI layout is not included.
- The main `.blend` is the authoritative delivery. This guide and script support recipient review and reconstruction.
