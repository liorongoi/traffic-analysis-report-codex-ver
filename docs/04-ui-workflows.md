# 04 — UI workflow and wireframes

Confirmed: Hebrew and English, full RTL/LTR. Mission setup records input type, one analysis mode, location/timezone, source/output folders, recursion, processing preference, and export-format preference. The wizard includes a bundled offline world map with country outlines and approximate IANA timezone regions, map-click coordinate selection, global GeoNames city/country search, installed IANA zone search, an editable IANA-zone assignment, and UTC-to-local/DST preview. City names and coordinates come from the local GeoNames index; zone names come from the workstation's installed timezone database. The timezone overlay is approximate and is not authoritative for legal or political boundaries. Local scan windows, persistent continuous-monitor pause/resume controls, automatic inference batch status, durable in-app alerts, manual camera-view groups with individual or selected-image reassignment, selected-image splitting into groups, and confirmed whole-group merges with append-only group/assignment history are available. A merge retains source/target region history and old run snapshots; current assignments move to the chosen target, whose regions govern later analysis. Versioned view-specific inclusion/exclusion polygons, an audited acknowledge-and-rerun flow for affected runs, contextual bilingual operator help, and left/right review shortcuts that respect keyboard focus and unsaved edits are available. Still-image object search uses a constrained offline grammar, displays the typed interpretation and exact saved-run scope, and requires operator confirmation before searching saved evidence; the detector does not infer color or fine subtype, but reviewers can record visually verified values and search filters those values while leaving unlabeled items uncertain. Geometry-based view proposals are available but require held-out camera acceptance. Object search supports explicit weekdays, ISO dates, 24-hour ranges, and named views from the selected saved run; open-ended schedule/region phrasing and video chat remain unimplemented. Browser desktop notifications require browser permission, and native Windows delivery remains unaccepted. In continuous still-image monitoring, a missing imported path and a byte-identical image at a new path appear as a reconciliation candidate. An operator confirms or declines the link using the reviewer field; links are append-only, reversible by a later unlink event, and never merge or suppress either capture record. The Video Traffic run form requires one or more supported vehicle classes; the selected labels are saved in the run configuration and fine subtypes such as semi-trailers remain unknown. The image-review toolbar includes an explicit bulk sign-off/export action whose confirmation shows the run, count, and fixed snapshot boundary; it is kept distinct from individual review, includes only arrivals within the confirmed boundary, and records that it is not independently verified ground truth. Still-image and video plate modes share a crop approval screen with source name, detector score, output folder, frame PTS/time base, and UTC only for verified origins. Decisions are revisioned, and no text is extracted.

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
   Video target classes [bicycle / car / motorcycle / bus / truck]
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

Video timing correction: a reviewer enters a start and end offset inside the immutable event interval. Each requested boundary snaps to the nearest saved analyzed frame assigned to that same track; ties choose the lower frame index. The correction is appended to review history and is retained through later accept/reject actions. Playback, clip exports, and event-table exports use the corrected interval and preserve the original interval. Reviewers can merge one track event into another event in the same run, or split a track at a saved analyzed frame that has observations on both sides. Merges project one logical event; splits create two logical child events. Original event rows remain immutable, corrections can be cleared by a later audit entry, and exports preserve the source event IDs.

Investigation viewer: original image on the main canvas; machine/added/removed/candidate overlays visually distinct; zoom/pan; rectangle addition; class/light-group edits; visible source UTC and local timestamp; Save/Approve; Previous/Next and position. Native-pixel coordinates survive browser scaling.

Nighttime table shows confirmed vehicle count separately from unresolved light-based candidates. Do not draw an invented whole-vehicle box around an unresolved light and present it as measured body extent.

Bulk action says Approve without individual review and export, previews exact run/result IDs, and excludes later arrivals. Invalid/unknown timestamps and unsupported formats remain visible. Empty, loading, interrupted, unauthorized and model-unavailable states need distinct messages.

Offline timezone map uses 239 bundled world country shapes, 419 approximate IANA timezone regions, and a GeoNames-derived index of 34,188 cities across 244 country codes, with coordinates, region names, alternate spellings, and IANA zones. Search and historical offset/DST preview use local assets and the runtime timezone database. No external tile, geocoding, or timezone service is used. City search covers the indexed places only; timezone boundaries are approximate. The contextual Help panel documents verified still-image import, UTC capture-time parsing, timezone windows, view/ROI revision behavior, human review, exports, and unsupported modes. Full browser/operator acceptance remains pending.
