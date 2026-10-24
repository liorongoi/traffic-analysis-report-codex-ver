# 03 — Architecture and data model

## Workstation topology

Authorized local/LAN browser → authenticated local web backend → mission service → durable per-mission queue → isolated worker processes. Backend owns databases, evidence storage and exports. A browser disconnect does not own worker lifetime. LAN binding is explicit, not enabled by an unauthenticated development server. No separate Windows Server, cloud, Docker or WSL is required.

Use spawn-compatible worker processes on Windows. Load only a selected adapter/model pack. Framework imports must be lazy. Missing weights/adapters produce an explicit unavailable state, never random detections. Model runtimes and immutable weights are separate from writable checkpoints.

## Per-mission storage

`<workspace>/missions/<UUID>/mission.sqlite3`, with private evidence, checkpoints and exports directories beside it. Source and output paths are never used as database names. A registry must not aggregate analytical results. Transactions and optimistic revisions protect edits, even with one operator using multiple browser tabs.

Foundation entities: mission configuration; runs and immutable configuration snapshots; versioned sources; run/source membership; append-only original predictions; append-only corrections with previous-revision linkage; review/reference eligibility; revisioned camera-view assignments; local scan windows; source-monitor checkpoints; and export snapshots. Inference snapshots include the mission configuration, source-run metadata, model configuration, schedule rules, view assignments and effective ROI. A bounded PyAV/FFmpeg decoder now yields one RGB frame at a time, enforces a maximum decoded pixel count, and retains each decoded frame's raw integer PTS and rational time base. Video event UTC values must derive from that PTS/time-base pair plus an explicit recording UTC origin; retain the raw pair because UTC datetime values have microsecond precision. A missing origin is an explicit unresolved state and must never be replaced by filesystem time or frame-index/FPS arithmetic. Future work adds video source/run persistence, chunk-continuous tracks/events, clip exports, automatically proposed view groups, native notifications, dataset manifests, and the complete training/model lifecycle.

Every correction requires valid mission/run/source lineage and a real operator identity. It records before/after evidence, type, geometry/text/class when relevant, label role, review completeness and model/preprocessing provenance. Application roles must prevent direct arbitrary database manipulation. Bulk approvals are retained but are not eligible human references.

## Durable execution

Run transitions: queued → running → completed/failed/cancelled; running ↔ paused only where adapters support checkpointing. Completed/failed runs are immutable historical executions; retry creates a new run. Browser status is fetched from persisted events/progress. Each planned run must declare its input snapshot and algorithm version.

Current still-image General Traffic runs execute through a durable local worker using a staged CPU YOLOX-S model pack; model boxes and experimental night-light outputs remain unconfirmed proposals. Per-image progress, cancellation/resume, and worker-exit reconciliation are implemented. Video decoding/tracking/events, country-specific plate OCR, YOLOX training and controlled model promotion remain unimplemented; a separate mission-local light-correction candidate can be refit only from explicit eligible labels.


Continuous still-image monitoring uses backend-owned filesystem notifications to wake scans sooner when available. Periodic full reconciliation remains authoritative and visibly serves as fallback when notification setup fails or the source is unavailable. The watcher is dynamically reconfigured when mission monitoring settings change.
