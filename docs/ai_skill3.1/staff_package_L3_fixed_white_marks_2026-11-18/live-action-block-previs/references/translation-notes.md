# English localization record

Translated on 2026-11-25. The package now uses English filenames, guides, configuration text, image labels, and Blender collection names. The five helper scripts retain their original behavior; create_mannequin.py now references the English template filename. The standalone handbook was rebuilt from its English source chapters.

## Provenance and validation

Both Blender assets were resaved in Blender 5.3.0 to rename the collection Characters. Geometry, vertex weights, face material assignments, and rest-rig fingerprints were unchanged. Original and localized file hashes are recorded in translation-blender-checks.json. model-profile.json and provenance.json distinguish original_source_sha256 from the current source_sha256. The historical validation-l3.json still describes the source package's earlier tests. translation-validation-l3.json records the fresh 60-frame template check. New project creation and white-material patching also succeeded in separate temporary files. No source footage was reconstructed in this translation task.

## Image localization

The built-in image_gen tool localized assets/client-palette.png and assets/L3_white_marks_stability_check.png. These are AI-edited reference illustrations. They are not unchanged source pixels or new experimental evidence. Use semantic-palette.json for exact RGB values and Blender validation reports for geometry evidence. The other six reference images were inspected and retained unchanged.

Final prompt set:

1. Palette: Translate all Chinese chart text into English. Preserve the ten swatches, color order, card layout, white background, and Jerry Chen attribution watermarks. Title: Blockout semantic color palette. Keep character colors consistent across shots. Label protagonist/red, second/blue, third/green, fourth/yellow, foreground/brown, background/gray, distant interactive object/pink, nearby interactive object/orange, light source/white, other special target/purple. Preserve the modeling principle: simplify geometry while retaining placement, occlusion, interaction, and camera-perspective relationships. English only.
2. L3 checks: Translate only Chinese labels in the supplied technical contact sheet. Preserve model poses, stripes, brightness, panels, and composition. Title: L3 standard: original proportions, fixed pure-white eye and chest stripes. Columns: Front standard; Side standard; Head turn: stripe rotates rigidly with head; Dark scene: stripes dim with exposure. Detail labels: Eye stripe detail and Chest stripe detail. Footer: Fixed dimensions refer to model proportions; natural perspective changes from distance, head turns and occlusion are retained. English only.

The package author reported historical software versions and FlowHMR trials. Translation does not independently verify those historical claims or install the described tools. See ../../ANALYSIS.md relative to the skill root, or ../../../ANALYSIS.md from this references directory, for app integration findings.
