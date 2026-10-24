# 03 — Architecture and data model

## Workstation topology

Authorized local/LAN browser → authenticated local web backend → mission service → durable per-mission queue → isolated worker processes. Backend owns databases, evidence storage and exports. A browser disconnect does not own worker lifetime. LAN binding is explicit, not enabled by an unauthenticated development server. No separate Windows Server, cloud, Docker or WSL is required.

Use spawn-compatible worker processes on Windows. Load only a selected adapter/model pack. Framework imports must be lazy. Missing weights/adapters produce an explicit unavailable state, never random detections. Model runtimes and immutable weights are separate from writable checkpoints.

## Per-mission storage

`<workspace>/missions/<UUID>/mission.sqlite3`, with private evidence, checkpoints and exports directories beside it. Source and output paths are never used as database names. A registry must not aggregate analytical results. Transactions and optimistic revisions protect edits, even with one operator using multiple browser tabs.

Foundation entities: mission configuration; runs and immutable configuration snapshots; versioned sources; run/source membership; append-only original predictions; append-only corrections with previous-revision linkage; review/reference eligibility; revisioned camera-view assignments; local scan windows; source-monitor checkpoints; and export snapshots. Inference snapshots include the mission configuration, source-run metadata, model configuration, schedule rules, view assignments and effective ROI. The video pipeline runs PyAV/FFmpeg in a spawned decoder process and sends one RGB frame at a time over a backpressured pipe to detection and tracking, avoiding co-loading PyAV/OpenCV FFmpeg dylibs. It enforces a maximum decoded pixel count, retains exact integer PTS/time base, reports frame progress, checks cancellation between frames, and closes active tracks at EOF. A policy-configured tracker carries same-recording identities through serialized checkpoints and marks ambiguous matches uncertain; thresholds have no defaults and must be explicit. Per-mission video sources are copied immutably with SHA-256, each run freezes source/origin/model/tracker configuration, checkpoints and closed-track event writes commit atomically, event IDs are run-scoped and idempotent, and event review revisions are append-only. These persistence primitives are not yet connected to the background video worker/API/UI. Video event UTC values derive from exact PTS/time-base plus explicit provenance-backed UTC origin; a missing origin is unresolved and must never be replaced by filesystem time or frame-index/FPS arithmetic. Future work adds worker/service integration, restartable processing, event playback/review UI, clip exports, automatically proposed views, native notifications, dataset manifests, and the full training/model lifecycle.

Every correction requires valid mission/run/source lineage and a real operator identity. It records before/after evidence, type, geometry/text/class when relevant, label role, review completeness and model/preprocessing provenance. Application roles must prevent direct arbitrary database manipulation. Bulk approvals are retained but are not eligible human references.

## Durable execution

Run transitions: queued → running → completed/failed/cancelled; running ↔ paused only where adapters support checkpointing. Completed/failed runs are immutable historical executions; retry creates a new run. Browser status is fetched from persisted events/progress. Each planned run must declare its input snapshot and algorithm version.

Current still-image General Traffic runs execute through a durable local worker using a staged CPU YOLOX-S model pack; model boxes and experimental night-light outputs remain unconfirmed proposals. Per-image progress, cancellation/resume, and worker-exit reconciliation are implemented. Video decoding/tracking/events, country-specific plate OCR, YOLOX training and controlled model promotion remain unimplemented; a separate mission-local light-correction candidate can be refit only from explicit eligible labels.


Continuous still-image monitoring uses backend-owned filesystem notifications to wake scans sooner when available. Periodic full reconciliation remains authoritative and visibly serves as fallback when notification setup fails or the source is unavailable. The watcher is dynamically reconfigured when mission monitoring settings change.
