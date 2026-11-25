# Limb depth corrections

Open a prepared clip and use **Limb depth timeline**, below the held-object timeline.

1. Pause the source video at an overlap and add a depth range.
2. Select the character and left/right upper arm or forearm.
3. Choose **In front of** or **Behind**, then the other character's arm segment or torso.
4. Set the inclusive start/end frames with the current-frame buttons. Left/right refer to the character's anatomy. Clearance defaults to 8 cm between the segment centre lines at their closest projected points.
5. Press **Apply depth corrections**. Inspect the updated pose overlay and correction report. Use **Finish** to render the solid blockout and update the delivery.

Use separate ranges as the order changes. Remove a range and apply again to undo it. Save preserves ranges without applying them; a subsequent Re-check or Finish also applies pending depth edits.

The explicit `depth` job reads the saved scene. It never falls back to tracking, body estimation, camera estimation, depth inference, or automatic prop detection—even if other settings have changed. An unprepared clip fails with a preparation message. Existing cached prop attachments are repositioned on corrected hands. Other pending review changes remain pending for the ordinary pipeline.

The solver adjusts shoulder/elbow rotations of both selected arms together. It preserves roots, camera, actor scale, bone lengths, and other joints. It uses cached fitted projections as image-alignment targets, anatomical limits, self-collision penalties, temporal correction regularization, and short transitions around the range boundaries. Ranges respect character visibility and shot-local track instances. Repeated application starts from the saved original rotations rather than accumulating changes.

This is a local numerical correction, with no new model or cloud service. A single video does not uniquely specify hidden limb positions. Some requested orders cannot be reached without degrading image alignment or exceeding joint limits. The report lists unresolved frame constraints and counts frames skipped when a character is invisible. Review the solid render before accepting the result; the fitting preview is a wire overlay, not a depth ground-truth check.

Regression tests cover a synthetic reversed overlap, fixed roots and camera, unchanged non-arm joints, front/behind controls, reversal/removal, invalid and conflicting ranges, inference-free job execution, dependency isolation, and UI actions. An isolated cached office scene smoke test is saved locally under `output/limb_depth_office`; its test range is not a ground-truth annotation of the source clip.
