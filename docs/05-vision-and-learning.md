# 05 — Camera views, nighttime hypotheses and learning

Cameras are fixed at locations with multiple recurring views. The current increment supports operator-created/renamed groups, revisioned image assignment, assignment history, and run-time view snapshots. Automatic grouping by scene geometry/landmarks, batch split/merge tools, and view-change detection remain unimplemented. Background and ROI validity must be specific to view and condition. Large shifts/zoom changes require review.

## Nighttime pipeline proposal

Original image → exposure/quality assessment → native-coordinate ROI → vehicle/small-object branch + light-component branch + per-view persistent-background evidence → compatible evidence fusion → separate candidates and confirmed vehicles. Enhancement is derived input, not independent corroboration. Never assume every new bright pixel or pair is a vehicle. Model glare/reflections/wet-road effects and avoid absorbing stationary vehicles into background.

A vehicle can appear as one unresolved bright region, two nearby headlights, then a visible body. Save light-component hypotheses and grouping evidence separately from whole-vehicle predictions. Still counts remain independent. Only continuous video evidence can support a track through those transitions. A pair becoming visible must not double the vehicle/event count. Distance/type/color remain unknown without supporting evidence/calibration.

## Offline chat

Interpret a request such as red pickup truck as a typed query with explicit class/attributes/exclusions/ROI/time/uncertainty rules. Show the interpretation before queueing. No generated-code execution, online API, or invented color from headlight-only/monochrome input.

## Correction lifecycle

Append every edit with source checksum/run/view/model/operator/time and revision. Preserve raw predictions. Latest approved correction defines a current decision, while prior revisions remain audit evidence. View reassignment and ROI edits are configuration changes, not object labels. Bulk approval is not human ground truth. Partial review does not label every unmarked pixel/object as negative. Preserve raw Unicode OCR plus normalized variants and country/layout hypotheses.

Training selection requires explicit human review, approved data roles and source retention. Keep mission boundaries; no automatic cross-mission pooling. No trigger threshold is approved. Each adapter must state real training/refit/calibration capabilities. Model candidates preserve base checkpoint/dataset IDs/settings/resources/metrics and failures. Compare with a held-out, camera/time-disjoint evaluation set and report regressions, including light-only and each country. No automatic promotion merely because training succeeded; retain rollback.

The current increment includes a local CPU YOLOX-S vehicle-proposal adapter and an experimental light-candidate/background path; see [real vehicle detection and review](10-vehicle-detection.md). These are not validated traffic counts, and no independent accuracy benchmark is available. The night-light head has a separate explicit-label refit path. YOLOX fine-tuning, the complete training/evaluation/promotion lifecycle, and production model acceptance remain unimplemented.
