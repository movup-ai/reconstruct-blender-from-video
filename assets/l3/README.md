# Supplied L3 runtime character

Source: `docs/ai_skill3.1/staff_package_L3_fixed_white_marks_2026-11-18/live-action-block-previs/assets/`.

- `L3_standard.blend`: complete production asset, copied without geometry modification.
- `character.json`: numerical fitting joints, rest rotations, rigid body parts, and source template hash. Derived from the actual evaluated native rig with its object scale, then normalized to a 1.75 m reference and +Y forward.
- `model-profile.json`: original model-space dimensions, topology, white-face indices, and rigid reference edges used for validation.
- `semantic-palette.json`: exact semantic sRGB values and color policies.

Rebuild explicitly with Blender:

```text
blender --background --factory-startup --disable-autoexec --python pipeline/build_l3_asset.py
```

The runtime does not depend on the docs folder. Keep the template and descriptor together; the renderer verifies their hash. See `docs/L3_PREVIS.md` for integration and validation scope.
