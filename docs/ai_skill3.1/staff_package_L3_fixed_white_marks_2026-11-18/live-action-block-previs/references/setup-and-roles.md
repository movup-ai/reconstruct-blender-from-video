# Software, plugins, and staff roles

## Environment reported by the source package

FlowHMR-Windows 0.3.6; Windows; RTX 4060 8 GB; CUDA 12.1; Blender 5.2.3 LTS. These are the package author's reported test versions, not a guarantee for other versions or a new verification of those claims.

FlowHMR and its models estimate motion. Blender handles ten-part characters, cameras, scenes, animation editing, and rendering. FlowHMR's body proportions differ from this mannequin, so exported motion still requires retargeting and image-space correction.

## Motion-capture workstation

1. Obtain the complete portable package from the [maintainer's FlowHMR-Windows releases](https://github.com/xiaoerofficial021-collab/FlowHMR-Windows/releases). This workflow was written for 0.3.6. Extract and launch `FlowHMR.exe`.
2. Choose the GPU backend in the Runtime panel. The source package's version-specific instructions list CUDA 12.1 for RTX 30/40 and CUDA 12.8 for RTX 50. Follow the relevant release's instructions for other hardware. The portable package includes base Python; staff need not first assemble another Python environment.
3. Confirm base models in the Models panel. Obtain public models through the software's documented sources. Import account-gated models such as SMPL+H from packages acquired through an authorized personal/company account. Existing third-party models on one computer are not automatically authorized for redistribution.
4. Test recognition, generation, export, and Blender import on a short clip first. Initial model loading can be slow; a still-loading interface is not evidence of completed generation.

The supplied guide reports 0.3.6 minimums of Windows 10/11 64-bit, 16 GB RAM, and a dedicated 6 GB GPU, with 32 GB RAM and 8+ GB VRAM recommended. It reports a trial on a 4060 8 GB. CPU support has different performance. These remain historical package statements rather than current hardware advice.

## Are the original Blender plugins required?

The documented route is FlowHMR output -> continuous correction -> L3 native rig and fixed-proportion rigid binding. The reported trials used no extra Blender retargeting plugin.

| Original archive | Purpose / limit | Used in reported trials? |
|---|---|---|
| FlowHMR_Rigify_Retarget_v0.1.0.zip | Rigify retargeting; verify plugin/Blender compatibility separately | No |
| FlowHMR_2_MMR_v1.3.1.zip | MMR route; not automatically the ten-part mannequin route | No |
| MikuMikuRig-5.2.2.zip | MMD/MMR workflow; does not infer motion from source video | No |

Staff using this mannequin do not need every archive simply for completeness. For another rig/manual plugin route, select tools for the target skeleton. This package did not validate those plugins against its reported Blender 5.2.3 environment.

## Helper scripts

Run template and mark-repair scripts with Blender's Python; a separate bpy installation is unnecessary. Run in a new background process after replacing example paths:

```powershell
& 'C:\path\to\blender.exe' --background --disable-autoexec --python 'package\live-action-block-previs\scripts\apply_white_marks.py' -- --input 'original.blend' --output 'new_project_white_marks.blend'
```

Output must be a new path. Scripts refuse to overwrite existing files. Mark repair recognizes this template's integrated materials, not arbitrary characters. Expect `WHITE_MARKS_SAVED`, then reopen the result. It preserves motion, camera, proportions, and lights and does not rerender the video.

To create an independent project from the supplied L3:

```powershell
& 'C:\path\to\blender.exe' --background --disable-autoexec --python 'package\live-action-block-previs\scripts\create_mannequin.py' -- --output 'new_template.blend'
```

A generated template is already included and can normally be appended directly. Helper scripts do not generate motion from video.

## L3 template scope

The production template includes native controls and fixed stripe binding. Opening, rendering, and ordinary bone transformations need no extra retargeting plugin. Install FlowHMR only on the capture workstation. Complete automatic retargeting of new motion is not included. Verify native Rigify panels/snapping against compatible versions when needed. Opening a template does not automatically align arbitrary FBX.
