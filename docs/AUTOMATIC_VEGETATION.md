# Automatic vegetation backgrounds

The default local background builder detects and fits vegetation without annotations, an API key, Codex, or Claude. User annotations remain optional corrections. Select Prepare to run automatic background discovery; Render background only applies the saved object annotations without refitting people.

Automatic discovery uses the installed SAM 3 checkpoint for both instance segmentation and text queries for `bush`, `shrub`, and `hedge`. It reuses one loaded checkpoint and image encoding. Concept masks must meet confidence, area, and geometry checks; duplicate masks are removed. These masks can recover shrubs that the semantic model labels as grass or trees.

The semantic fallback retains individual SAM foliage instances. Measured height above the ground, image aspect, and trunk shape distinguish low shrubs, hedges, and upright trees. Missing depth pixels remain part of the silhouette when enough reliable depth supports the object. Indoor generic plants retain the potted-plant prototype.

Bush and hedge geometry stays inside its requested width, height, and depth, including small or flat shrubs. Fitting jointly adjusts position, dimensions, and orientation against the detected silhouette and reliable depth. The exported geometry uses the same deterministic seed as the fitter. Later views are checked before they can move the initial anchor.

If an automatic shrub fits poorly, the builder tries the alternative bush/hedge prototype once. It keeps the alternative only when mask overlap improves. Recovery is bounded and uses local geometry and mask comparisons. It does not request user assistance or call an external language model. Invalid scene-mask caches are archived and regenerated automatically.

`object_alignment.json` records silhouette overlap, center and size errors, detected-bound errors, and validation status. `environment_profile.json` records model loading, SAM inference, classification, instance separation, fitting, recovery, and verification timings. Nested timings are not additive. A failed validation remains a reported failure; a prototype is an approximation and is not guaranteed to reproduce every plant exactly.

Advanced configuration can disable the extra concept queries with `sam3_vegetation_prompts: false`. Changing it invalidates the environment dependency without invalidating human fitting. SAM mask caches also include the query list in their keys.

Regression checks: `.venv\Scripts\python.exe -m unittest tests.test_bush_auto tests.test_object_fit -q`. Tests cover automatic classification, connected-foliage separation, sparse-depth silhouettes, text-mask acceptance, invalid-cache recovery, prototype dimensions, fitter/export consistency, and the bounded recovery path. They do not require a GPU; actual model inference must also be checked on installed hardware.
