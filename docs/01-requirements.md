# 01 — Confirmed requirements and traceability

The full source request is requirements-source.txt, preserved verbatim. This table groups requirements without removing their source detail. Status: F = first foundation increment; P = planned, not implemented. Test identifiers are acceptance obligations; only executed tests count as evidence.

This table is the initial foundation-stage mapping, not a current completeness claim. For the current state of each requirement area, use [the requirements fit review](11-requirements-fit-review.md); for current vehicle/night inference evidence, use [document 10](10-vehicle-detection.md). Later implementation evidence supersedes foundation-era statuses where they differ.

| ID | Requirement | Status | Verification |
|---|---|---|---|
| ENV-01 | Independent application, offline Windows workstation, local/LAN browser access, one active user | P | Native Windows disconnected and two-computer session test |
| ENV-02 | No runtime downloads/CDN/cloud services; staging separate | P | Installer/network-denial audit |
| HW-01 | Detect 16/48 GB RAM and GPU support independently, preserve OS headroom | F/P | Budget unit tests; target measurements pending |
| MIS-01 | Mission name, input type, country, named timezone, source/destination, recursion, one-time/continuous, XLSX/CSV | F/P | Mission settings form, persistence, path validation, and unsupported-mode guards are implemented; continuous still-image intake creates one inference run per imported batch with durable queue/retry status; mission schedules, the bundled offline country/time-zone map, and CSV/XLSX output-folder export are implemented; target workstation acceptance remains pending |
| MIS-02 | Separate database per mission, independent runs, no output mixing | F | SQLite isolation/lineage tests |
| TIME-01 | Parse confirmed UTC filename and preserve minute precision | F | Valid/invalid filename tests |
| TIME-02 | Historical local times, DST, multiple overnight windows | F/P | IANA-zone image filtering supports date ranges, weekdays, multiple windows, overnight windows, and DST folds/gaps; start is inclusive and end exclusive. The bundled offline map selects the supported-country zone and coordinates and previews UTC instants in that zone; it uses supported-country boundaries rather than a worldwide timezone polygon atlas |
| ING-01 | Recursive optional monitoring, stability, restart ledger, notifications, immutable source versions | P | Backend polls continuous still-image missions, waits for stable files, decodes JPEG/PNG inputs before import, checkpoints path/version state, resumes after restart, records missing paths without deleting imported snapshots, and imports changed versions without replacing old snapshots. Operators can persistently pause/resume monitoring. Each imported batch queues a source-scoped inference run, retries durably while another worker is active, and exposes pending/queued/completed/failed batch counts. New-batch/completion/failure in-app alerts persist until acknowledged. Invalid files receive per-file errors in continuous mode and abort explicit import batches atomically. Filesystem notifications, native OS alerts, rename/reconciliation policy, and workstation acceptance remain pending |
| VIEW-01 | Fixed camera locations, recurring views; automatic proposals and manual split/merge/reassign | P | Manual view groups, rename, per-image revisioned assignment, representative/count listing, and run snapshots are implemented. Automatic view proposals, batch split/merge workflows, and held-out grouping evaluation remain pending |
| ROI-01 | Original-image-coordinate inclusion/exclusion polygons with validation and versioning | P | Normalized original-image geometry, exclusion precedence, immutable region history, run snapshots and HTTP API are implemented; automatic invalidation review remains pending |
| IMG-01 | Count each still independently; no unique-flow claim from irregular captures | P | Inference inputs are filtered per image using each capture's local schedule; per-image benchmark/report verification remains pending |
| CHAT-01 | Offline query interpretation → typed specification → operator confirmation | P | Allowed/unsupported/unobservable-attribute cases |
| PLATE-01 | Iran/Lebanon/Israel/Latvia ALPR with Unicode, raw/normalized values, uncertain results | P | Country-specific authorized ground truth |
| NIGHT-01 | Separate light components, candidates and confirmed vehicles; background by view | P | Nighttime candidate and false-alarm tests |
| NIGHT-02 | One unresolved light → paired headlights → visible body; do not double count | P | Annotated nighttime sequence and still-image grouping cases |
| VID-01 | Bounded decoding, real PTS, chunk-continuous tracking, deduplicated events | P | VFR/chunk-boundary/occlusion/long-video tests |
| RUN-01 | Durable states/progress/errors; pause/resume/cancel independent of browser | F/P | Per-image prediction commits preserve progress; startup reconciliation detects missing workers; failed/stopped runs resume the same immutable input/config and skip saved predictions. Recovery/resume tests pass; native Windows process-liveness handling remains untested |
| REVIEW-01 | Image/event tables, rectangles, false positives, classes/OCR, navigation, approvals | F/P | Audit store tests; manual rectangle review preview implemented; actual inference/video/OCR review pending |
| REVIEW-02 | Bulk approval clearly unreviewed, snapshot export excludes later arrivals | F/P | Label-eligibility tests; export workflow pending |
| LEARN-01 | Preserve every correction, revision/conflict resolution, provenance | F | Append-only correction and lineage tests |
| LEARN-02 | Real adapter capabilities, protected splits, versioned candidates, controlled promotion | P | Training fixture/evaluation/rollback acceptance |
| EXPORT-01 | XLSX/CSV and clips, Unicode/leading zeros, no truncation/overwrite, snapshot lineage | P | CSV/XLSX files save with unique names, source IDs/checksums, snapshot hashes, and a mission-local export ledger; rows stream in bounded batches, and XLSX sheets split at the Excel row limit; video clips and broader recovery testing remain pending |
| UX-01 | Modern responsive wizard/review/progress/monitoring/Help; Hebrew/English RTL/LTR | P | Mission setup, bilingual scan-window editor, monitor status, and review are partial; Help and full browser/operator acceptance remain pending |
| DEP-01 | Complete offline media, small app updates, no implicit admin/Docker/WSL | P | Native clean-install/update/rollback tests |

No detection accuracy or throughput target is approved. Foundation tests do not establish model performance or Windows compatibility.
