# Initial implementation evidence — 2026-10-24

Superseded for inference/review status by [real detector verification](10-vehicle-detection.md). The limitations below describe the earlier import-only milestone.

## Confirmed constraints

English and Hebrew UI, full bidirectional layout. Current-user Windows installation without elevation. Local and authorized network-share sources. One active user. Independent mission databases, no mission pooling or SpectraScope dependencies.

## Executed checks

- 21 Python unittest checks passed using neutral Python 3.12 on this Mac: capture-time parsing/precision, invalid dates, historical DST, proposed memory tiers, source snapshot independence/tamper detection, append-only reviews, revision conflicts, holdout protection, mission isolation, geometry validation, traversal/symlink rejection, failed import rollback, own-workspace rejection, HTTP page/assets, session-token requirement, origin/Host rejection, mission creation and CSV header export.
- `node --check web/app.js` passed, including after the final nonmodal unsaved-change adjustment.
- Imported 143 user-supplied Fintraffic samples with manifest checksums. Source images remain in Downloads; independent copies and provenance are in the new project workspace.
- Browser showed 143 table rows, a decoded 1280-pixel-wide source image, Hebrew `lang=he` / `dir=rtl`, and correct Europe/Helsinki historical local time.
- Browser rectangle draw created a candidate entry. Removing it removed the overlay. Test marks were not saved as human truth.
- Native JavaScript confirmation interrupted browser automation. Replaced the navigation confirmation with an inline unsaved-changes message and explicit discard button. That final navigation flow still needs a fresh browser interaction check. Do not claim a complete UI acceptance pass.

## Deliberate limitations

The Fintraffic sample set has no human ground-truth labels. No precision/recall/F1, night detection accuracy, throughput, GPU compatibility or video performance has been measured. Annotation geometry is stored as normalized original-image coordinates. Manual camera-view groups and individual/selected-image assignments, versioned view-specific inclusion/exclusion polygons, append-only history, proposal filtering, immutable run snapshots of mission/source/model/schedule/view/ROI configuration, and warnings when an affected view's current ROI geometry differs from a saved run are implemented. An operator can acknowledge a changed ROI and create an audited rerun using the same saved input images and model settings; model drift blocks this shortcut. Automatic view discovery/grouping remains future work.

Manual browser smoke on 2026-10-25 used a disposable local mission and two generated JPEGs: the mission imported both images, a camera group was created and refreshed in the list, and the selected-image batch assignment updated its image count to two. The browser console showed no errors after fixing the form refresh. This checks the local browser workflow only; it does not establish native Windows or packaged deployment acceptance.

No actual Windows runtime was staged, setup EXE compiled/extracted, target installed, native runtime checked or no-admin policy exercised. Installer scripts are source only. Network-share behavior and production LAN authentication remain unverified/unimplemented.

The initial failed sample import was followed by a corrected successful 143-image import; failure evidence was retained. Initial test failures identified an INSERT placeholder mismatch, which was fixed before the successful suite.

## Next implementation gates

1. Continuous still-image intake runs in a backend polling thread, waits five seconds for unchanged file size/mtime, persists per-path checkpoints and status, avoids importing the same path/checksum twice, validates complete JPEG/PNG decoding, records invalid paths, reports source paths that disappear without deleting imported snapshots, and stores replacements in new immutable source runs. Each imported still-image General Traffic batch creates a source-scoped inference run; durable pending batches retry when another worker is active, link to their inference run across restart, and show pending/queued/completed/failed status in the UI. Operators can pause/resume intake; new-batch, completion, and failure alerts persist until acknowledged in the in-app alert region. The mission wizard loads pinned local country geometry, selects country/coordinates and an IANA zone, and previews UTC instants in the selected zone. Mission scan windows support weekdays, date ranges, multiple local-time windows, overnight windows, and IANA DST behavior; each inference run snapshots the rules and excludes out-of-window captures. Manual view groups, single-image and explicitly selected multi-image assignment with per-image history, and versioned ROI polygons are available and tested; batch assignment rejects invalid selections without partial writes. Runs are flagged when current ROI geometry for an assigned view differs from the saved snapshot; operators can acknowledge the exact drift state and create a same-input rerun, with append-only records. Watchdog filesystem notifications wake scans while periodic reconciliation remains active, with visible polling fallback when unavailable. Native OS notifications, rename matching/reconciliation, a worldwide timezone polygon atlas, automatic view discovery, and broader source-health recovery remain pending. CSV/XLSX output-folder snapshots stream in bounded batches and reconcile interrupted exports at startup.
2. Extend the existing local CPU detector and experimental night-light proposals only after collecting authorized, independently reviewed day/night references; measure accuracy and count error before treating outputs as operational results.
3. Add video/ALPR adapters and protected learning pipeline.
4. Build and test complete Windows offline media under a current-user account, including missing dependencies and process lifecycle checks.

## Current verification update — 2026-10-25

The current suite contains 105 Python checks. Recent increments add deterministic source/output path-overlap checks and observer-adapter tests for file-event wakeup, recursive watch selection, watch reconfiguration on pause/settings changes, unconfigured-source handling, and periodic polling fallback. The local `.venv` does not include Watchdog, so these tests use an injected observer test double; actual macOS/Windows/native and authorized network-share notification delivery has not been verified. `watchdog==6.0.0` is now an explicit application and offline runtime dependency, and the Windows staging validator rejects a runtime missing it. `node --check web/app.js` and `git diff --check` passed.

## Video timestamp-decoder update — 2026-10-25

Added `av==18.1.0` and a bounded incremental PyAV/FFmpeg decoder. It yields one RGB image at a time, enforces a decoded-pixel limit, checks stream/frame validity and timestamp ordering, and keeps each frame's integer PTS plus its exact rational time base. Absolute UTC is returned only when a caller supplies an aware recording origin. A generated VFR Matroska fixture with PTS `0, 40, 160, 200` at time base `1/1000` verified unequal frame gaps (`40, 120, 40` ms); no frame-index/FPS interpolation was used. Five decoder-specific tests and the full 105-test Python suite passed; `node --check web/app.js` and `git diff --check` passed. This validates the decode/timestamp foundation only; it does not validate customer codec compatibility, recording UTC origin, video runs, chunking/tracking, events or clips, nor Windows decode performance. Importing both PyAV and OpenCV in this macOS test process printed duplicate bundled FFmpeg Objective-C class warnings; native runtime interaction still needs target validation.
