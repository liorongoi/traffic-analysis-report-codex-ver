# 04 — UI workflow and wireframes

Confirmed: Hebrew and English, full RTL/LTR. Mission setup records input type, one analysis mode, location/timezone, source/output folders, recursion, processing preference, and export-format preference. The wizard includes a bundled map of supported countries, map-click coordinate selection, an IANA-zone assignment, and UTC-to-local/DST preview. Local scan windows, persistent continuous-monitor pause/resume controls, automatic inference batch status, durable in-app alerts, manual camera-view groups with individual or selected-image reassignment, selected-image splitting into groups, and confirmed whole-group merges with append-only group/assignment history are available. A merge retains source/target region history and old run snapshots; current assignments move to the chosen target, whose regions govern later analysis. Versioned view-specific inclusion/exclusion polygons, an audited acknowledge-and-rerun flow for affected runs, contextual bilingual operator help, and left/right review shortcuts that respect keyboard focus and unsaved edits are available. Still-image object search uses a constrained offline grammar, displays the typed interpretation and exact saved-run scope, and requires operator confirmation before searching saved evidence; unsupported color and fine-subtype attributes remain unknown. Geometry-based view proposals are available but require held-out camera acceptance. Object search supports explicit weekdays, ISO dates, 24-hour ranges, and named views from the selected saved run; open-ended schedule/region phrasing and video chat remain unimplemented. Browser desktop notifications require browser permission, and native Windows delivery remains unaccepted. The image-review toolbar includes an explicit bulk sign-off/export action whose confirmation shows the run, count, and fixed snapshot boundary; it is kept distinct from individual review, includes only arrivals within the confirmed boundary, and records that it is not independently verified ground truth. Still-image and video plate modes share a crop approval screen with source name, detector score, output folder, frame PTS/time base, and UTC only for verified origins. Decisions are revisioned, and no text is extracted.

```
Mission library
  [New mission] [Open mission] [Monitoring status] [Notifications] [Help]
       ↓
1. Mission setup: name | image/video | country | location/timezone map
   Source + recursion | one-time/continuous | output | XLSX/CSV
   Filename parse preview: raw name → UTC minute → historical local time
       ↓
2. Configure: exactly one analytical mode
   View groups [representatives / manual rename / reassign]
   View-specific inclusion/exclusion polygons [normalized original-image points / revision history]
   Schedules | query interpretation / plate-crop approve-reject screen
   Model availability + resource budget + validation summary
       ↓
3. Run & review: durable progress, actual errors, pending review
   Image table ↔ investigation viewer
   Video events ↔ timestamped player and track/clip inspector
       ↓
4. Approve/export: explicit result snapshot and destination
       ↓
5. Learning: correction inventory → eligible dataset → candidate training
   → independent evaluation → controlled promotion/rollback
```

Video timing correction: a reviewer enters a start and end offset inside the immutable event interval. Each requested boundary snaps to the nearest saved analyzed frame assigned to that same track; ties choose the lower frame index. The correction is appended to review history and is retained through later accept/reject actions. Playback, clip exports, and event-table exports use the corrected interval and preserve the original interval. This supports trimming only; track merge and split behavior remains undecided and unimplemented.

Investigation viewer: original image on the main canvas; machine/added/removed/candidate overlays visually distinct; zoom/pan; rectangle addition; class/light-group edits; visible source UTC and local timestamp; Save/Approve; Previous/Next and position. Native-pixel coordinates survive browser scaling.

Nighttime table shows confirmed vehicle count separately from unresolved light-based candidates. Do not draw an invented whole-vehicle box around an unresolved light and present it as measured body extent.

Bulk action says Approve without individual review and export, previews exact run/result IDs, and excludes later arrivals. Invalid/unknown timestamps and unsupported formats remain visible. Empty, loading, interrupted, unauthorized and model-unavailable states need distinct messages.

Offline timezone map must use bundled geometry/tzdb/fonts, with named-zone search and historical offset/DST preview. No live tile or timezone API. The map is implemented for the supported countries. The contextual Help panel documents verified still-image import, UTC capture-time parsing, timezone windows, view/ROI revision behavior, human review, exports, and unsupported modes. Full browser/operator acceptance remains pending.
