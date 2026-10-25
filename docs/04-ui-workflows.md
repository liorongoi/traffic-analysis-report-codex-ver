# 04 — UI workflow and wireframes

Confirmed: Hebrew and English, full RTL/LTR. Mission setup records input type, one analysis mode, location/timezone, source/output folders, recursion, processing preference, and export-format preference. The wizard includes a bundled map of supported countries, map-click coordinate selection, an IANA-zone assignment, and UTC-to-local/DST preview. Local scan windows, persistent continuous-monitor pause/resume controls, automatic inference batch status, durable in-app alerts, manual camera-view groups with individual or selected-image reassignment, versioned view-specific inclusion/exclusion polygons, an audited acknowledge-and-rerun flow for affected runs, contextual bilingual operator help, and left/right review shortcuts that respect keyboard focus and unsaved edits are available. Still-image object search uses a constrained offline grammar, displays the typed interpretation and exact saved-run scope, and requires operator confirmation before searching saved evidence; unsupported color and fine-subtype attributes remain unknown. Geometry-based view proposals are available but require held-out camera acceptance. Object search supports explicit weekdays, ISO dates, 24-hour ranges, and named views from the selected saved run; open-ended schedule/region phrasing, native OS notifications, video chat, and video plate-crop review are unimplemented. Still-image and video plate modes share a crop approval screen with source name, detector score, output folder, frame PTS/time base, and UTC only for verified origins. Decisions are revisioned, and no text is extracted.

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

Investigation viewer: original image on the main canvas; machine/added/removed/candidate overlays visually distinct; zoom/pan; rectangle addition; class/light-group edits; visible source UTC and local timestamp; Save/Approve; Previous/Next and position. Native-pixel coordinates survive browser scaling.

Nighttime table shows confirmed vehicle count separately from unresolved light-based candidates. Do not draw an invented whole-vehicle box around an unresolved light and present it as measured body extent.

Bulk action says Approve without individual review and export, previews exact run/result IDs, and excludes later arrivals. Invalid/unknown timestamps and unsupported formats remain visible. Empty, loading, interrupted, unauthorized and model-unavailable states need distinct messages.

Offline timezone map must use bundled geometry/tzdb/fonts, with named-zone search and historical offset/DST preview. No live tile or timezone API. The map is implemented for the supported countries. The contextual Help panel documents verified still-image import, UTC capture-time parsing, timezone windows, view/ROI revision behavior, human review, exports, and unsupported modes. Full browser/operator acceptance remains pending.
