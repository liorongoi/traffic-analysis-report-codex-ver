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

The Fintraffic sample set has no human ground-truth labels. No precision/recall/F1, night detection accuracy, throughput, GPU compatibility or video performance has been measured. Annotation geometry is stored as normalized original-image coordinates. Manual camera-view groups and assignments, versioned view-specific inclusion/exclusion polygons, append-only history, proposal filtering, and inference-run snapshots are implemented; automatic view discovery, batch tools, and validity/invalidation review remain future work.

No actual Windows runtime was staged, setup EXE compiled/extracted, target installed, native runtime checked or no-admin policy exercised. Installer scripts are source only. Network-share behavior and production LAN authentication remain unverified/unimplemented.

The initial failed sample import was followed by a corrected successful 143-image import; failure evidence was retained. Initial test failures identified an INSERT placeholder mismatch, which was fixed before the successful suite.

## Next implementation gates

1. Continuous still-image intake now runs in a backend polling thread, waits five seconds for unchanged file size/mtime, persists per-path checkpoints and status, avoids importing the same path/checksum twice, validates complete JPEG/PNG decoding, records invalid paths, and stores replacements in new immutable source runs. It does not run inference automatically. Mission scan windows support weekdays, date ranges, multiple local-time windows, overnight windows, and IANA DST behavior; each inference run snapshots the rules and excludes out-of-window captures. Manual view groups and versioned ROI polygons are available and tested. Filesystem notifications, operator alerts, robust rename/missing-file reconciliation, automatic view discovery, ROI invalidation review, and broader source-health recovery remain pending. CSV/XLSX output-folder snapshots stream in bounded batches and reconcile interrupted exports at startup.
2. Extend the existing local CPU detector and experimental night-light proposals only after collecting authorized, independently reviewed day/night references; measure accuracy and count error before treating outputs as operational results.
3. Add video/ALPR adapters and protected learning pipeline.
4. Build and test complete Windows offline media under a current-user account, including missing dependencies and process lifecycle checks.
