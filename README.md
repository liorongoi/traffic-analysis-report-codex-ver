# Traffic Analysis — independent offline application

## Working inference and review preview (2026-10-24)

This is a **separate project**, outside SpectraScope. Its source, workspace, database, launch port and installer identity are independent.

Implemented:

- Browser review interface with English/Hebrew switching, RTL/LTR layout and directionally isolated timestamps/filenames.
- Mission setup stores input type, one analysis mode, location/time zone, source/output paths, recursion, processing preference and export formats in a per-mission SQLite configuration. The backend validates configured directories and prevents image import or inference for unsupported mission types/modes.
- Import batches remain separate runs; non-recursive image imports honor the saved mission setting.
- Explicit camera filename parser (`DDMMYYYY-HHmm-suffix`, UTC minute precision) and explicit Fintraffic manifest importer (UTC second precision).
- Recursively import camera images. Fintraffic imports verify every image against its provided SHA-256 manifest. Originals stay untouched; independent mission snapshots are verified again when viewed.
- Image table and previous/next review; draw/remove normalized image rectangles; distinguish confirmed vehicles, candidates, lights and background lights.
- Append-only review revisions, operator, evidence role and completeness. Optimistic revision checks prevent overwriting concurrent edits; held-out image roles cannot become training roles. An explicit bulk sign-off path previews the saved run, exact image count, and fixed row boundary, requires operator confirmation, exports only that snapshot, and labels it `bulk_unreviewed` rather than independent ground truth.
- Traceable CSV snapshots written to the configured output folder with unique filenames, source IDs/checksums, revision state, and a snapshot hash. Machine proposals remain explicitly unconfirmed until a human reviews them.
- Loopback development host with a session token for writes, host/origin checks and self-only browser asset policy.
- Real local YOLOX-S vehicle inference, separately labelled experimental night-light proposals, durable progress/cancellation, immutable original predictions and saved correction lineage. See [vehicle detection and testing](docs/10-vehicle-detection.md).
- Bounded PyAV/FFmpeg video decoding that yields one RGB frame at a time and preserves each frame's exact integer PTS and rational time base; generated VFR regression coverage confirms uneven timestamp gaps. Absolute UTC still requires a supplied recording origin.
- Policy-configured, same-recording track assembly with JSON-compatible chunk checkpoints. Ambiguous associations remain uncertain; thresholds must be supplied explicitly and are not production-validated.
- Incremental video analysis connects the spawned decoder to the detector and tracker, reports per-frame position/progress, supports cancellation, and leaves absolute UTC unset without an explicit origin. The spawned decoder process isolates bundled FFmpeg libraries from OpenCV.
- Per-mission video evidence copies, immutable run snapshots, atomic track checkpoints/event writes, idempotent events, and append-only event-review records. Video Traffic runs now use a supervised child worker, persist model/tracker settings, share the single local processing slot with image inference, and expose source/run/event-review APIs. The browser supports source registration with UTC-origin provenance, run settings and lifecycle controls, paginated event review, integrity-checked byte-range playback, and event seeking, and event clip exports. Exports use explicit operator-entered pre/post context, encode video-only H.264 MP4, and record event/run/source lineage plus a SHA-256 output hash in the mission ledger.
- Windows **preview installer source**, current-user directory, `PrivilegesRequired=lowest`, app-local runtime, current-user shortcuts, offline checksums and no download fallback.

## Run the preview

Requires Python 3.11+ and installed IANA timezone data. On Windows, the eventual private runtime must include `tzdata`; installation must not download it.

```sh
PYTHONPATH=src .venv/bin/python -m road_traffic serve --workspace workspace/sample-review --port 8765
```

Open http://127.0.0.1:8765 in a browser. The development preview binds **only to loopback**. It is not the authenticated LAN production host.

In this workspace, use the neutral development runtime at `/Users/lioron/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3` if system Python is older than 3.11.

Fintraffic samples supplied by the user are imported locally into `workspace/sample-review`. They are not installer defaults. The initial failed development import is retained as a separate empty mission/run record for diagnosis; select the mission containing 143 images.

## Tests

```sh
PYTHONPATH=src python -m unittest discover -s tests -v
```

See [verification record](docs/09-verification.md). The 156-test Python suite covers the current API and processing workflows, including operator-confirmed offline search, video workflow, host inventory, source integrity, and byte-range checks. Additional real-model and browser evidence is recorded in docs/10-vehicle-detection.md. These checks do not establish detection accuracy or native Windows compatibility.

## Not implemented yet

Interrupted inference runs preserve per-image results and can resume unfinished inputs under the original immutable run configuration. Continuous still-image intake runs in the backend; it fully decodes JPEG/PNG files up to the current 24-megapixel limit before import, reports invalid paths, and lists missing source paths without deleting imported evidence. Each newly imported continuous batch automatically queues its own still-image General Traffic inference run, retries while the single local worker is busy, persists batch status across restart, and keeps all predictions pending human review. Operators can persistently pause/resume intake, and new-batch/completion/failure events are stored until shown in the in-app alert region. Watchdog filesystem notifications wake scans early; periodic reconciliation remains active, with an explicit polling-fallback status when notifications cannot be established. The mission wizard includes a bundled offline map for the five supported countries; map clicks select country and coordinates, populate the IANA zone, and preview UTC-to-local time across historical DST transitions. Local-time scanning windows are applied to inference runs; operators can create camera-view groups and revision per-image or selected-image assignments. View-specific, validated inclusion/exclusion polygons are revisioned and snapshotted into inference runs; changed ROI geometry flags affected runs, and operators can acknowledge the exact change and launch an audited rerun over the same inputs while model settings match. CSV and XLSX exports write Unicode-preserving, non-overwriting snapshots into the configured output folder and record their checksums in the mission ledger. Export rows are written in bounded batches. Overnight windows use the selected start weekday; boundaries are start-inclusive/end-exclusive; daylight saving uses the mission's IANA zone. Filesystem notifications, native OS alerts, rename/reconciliation policy, held-out camera-view grouping acceptance, multi-view background generation, continuous video monitoring, validated night traffic counting/tracking, authorized target-country plate-localization acceptance (OCR and recognition are excluded), open-ended natural-language schedule/region parsing (explicit weekdays, ISO dates, 24-hour windows, and named saved views are supported), YOLOX fine-tuning/automatic model promotion and authenticated LAN operation remain outstanding. Experimental light candidates are not validated detections. See [bundled map data provenance](docs/third-party-data.md).

Video source/run/event storage is connected to a supervised child worker and authenticated loopback APIs for source registration, run start/status/cancel/resume, and append-only event review. Orphaned workers are reconciled as failed resumable runs while preserving frame checkpoints. The browser provides video source registration, run configuration and lifecycle controls, paginated event review, integrity-checked playback and event seeking, and H.264 MP4 event-clip export with explicit pre/post context and a mission-local integrity ledger. Representative-media/UTC acceptance remains incomplete; see [the requirements fit review](docs/11-requirements-fit-review.md).

The current development inference runtime is the independently staged `.venv`. `requirements-inference-macos.lock.txt` records its packages; installed runtime code never invokes pip.

The on-demand host report reads CPU, RAM, GPU/driver inventory, runtime provider names, and free disk capacity, then measures a small temporary write+fsync and immediate read on the application-data volume. The read may be served from the OS cache, so this is a local comparison aid rather than a sustained-storage benchmark. Provider listing does not prove GPU execution. Windows target hardware and GPU execution remain unverified.

The Windows setup EXE has **not** been built or tested. It requires a complete Windows x64 runtime and timezone pack staged independently, Inno Setup compilation, extracted-payload inspection and a native no-admin/offline Windows test. The preview uses an existing browser; the final application/browser-engine packaging is still outstanding. Do not use the setup source as proof that the finished offline application exists.

## Requirements and design

The verbatim request is in `docs/requirements-source.txt`. Design documents 01–08 describe the intended complete product; [current verification status](docs/09-verification.md) and [vehicle detection evidence](docs/10-vehicle-detection.md) record what has actually been implemented and tested. The [requirements fit review](docs/11-requirements-fit-review.md) summarizes current coverage and the remaining gaps.

## Night separation and correction learning

See [night review and training status](docs/11-night-separation-and-learning.md). Fifty tests pass. The main mission has zero saved corrections, so no user-trained candidate exists yet. The separate light head supports explicit-label refits; YOLOX fine-tuning is unsupported.
