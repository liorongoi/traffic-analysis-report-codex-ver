# 04 — UI workflow and wireframes

Confirmed: Hebrew and English, full RTL/LTR. The current review preview switches languages; the complete wizard remains planned.

```
Mission library
  [New mission] [Open mission] [Monitoring status] [Notifications] [Help]
       ↓
1. Mission setup: name | image/video | country | location/timezone map
   Source + recursion | one-time/continuous | output | XLSX/CSV
   Filename parse preview: raw name → UTC minute → historical local time
       ↓
2. Configure: exactly one analytical mode
   View groups [representatives / split / merge / reassign]
   Inclusion/exclusion polygons | schedules | query interpretation/ALPR options
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

Investigation viewer: original image on the main canvas; machine/added/removed/candidate overlays visually distinct; zoom/pan; rectangle addition; class/OCR/light-group edits; visible source UTC and local timestamp; Save/Approve; Previous/Next and position. Native-pixel coordinates survive browser scaling.

Nighttime table shows confirmed vehicle count separately from unresolved light-based candidates. Do not draw an invented whole-vehicle box around an unresolved light and present it as measured body extent.

Bulk action says Approve without individual review and export, previews exact run/result IDs, and excludes later arrivals. Invalid/unknown timestamps and unsupported formats remain visible. Empty, loading, interrupted, unauthorized and model-unavailable states need distinct messages.

Offline timezone map must use bundled geometry/tzdb/fonts, with named-zone search and historical offset/DST preview. No live tile or timezone API. This map and all interactive screens remain planned.
