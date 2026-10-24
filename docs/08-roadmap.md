# 08 — Phased roadmap and acceptance

| Phase | Deliverable | Exit evidence | Current state |
|---|---|---|---|
| 0 | Requirements, unknowns, architecture/data model, UI wireframes, vision/learning, benchmarks, hardware/deployment and test roadmap | Reviewable documents linked from README | Authored |
| 1 | Pure domain foundation: timestamps/schedules, mission DBs/run states, prediction/correction history, hardware budgets, CLI | Isolated unit/integration tests, no fake inference | Timestamp parsing, source snapshots and review revisions tested; scheduling/predictions/worker recovery still planned |
| 2 | Mission setup, authenticated web UI, offline TZ map, source retention/ingestion, monitoring/recovery, view and ROI review | Browser tests, partial-copy/restart cases, representative files | Partial: mission settings and mode guards are saved and tested; directory monitoring, offline map, complete source health/retention, views and ROI remain planned |
| 3 | Licensed/staged pretrained vehicle + nighttime adapters and reports | Real held-out day/night/headlight/camera benchmarks | Partial: local CPU YOLOX-S proposals and experimental night-light proposals run; independent accuracy evidence and approved representative data remain outstanding |
| 4 | Incremental video/PTS/tracking/events/player/clips | Long/VFR/occlusion/chunk-boundary tests | Not implemented |
| 5 | Country-specific ALPR and exports | Iran/Lebanon/Israel/Latvia authorized tests and Unicode round-trips | Needs data and compatible adapters |
| 6 | Full review UI, eligible datasets, training candidates/evaluation/promotion/rollback | Correction persistence, protected splits, real refit evidence | Ledger foundation only |
| 7 | Complete offline Windows delivery and LAN acceptance | Clean native target install, no network, update/rollback, 16/48 GB and GPU/CPU results | Unverified; no installer built |

Do not mark later phases complete because a UI prototype, model citation, synthetic test or package hash passes. Quantitative accuracy/throughput acceptance is not set. CI must run pure tests, integration/browser tests, model adapter tests only with real packs and native target tests separately. Missing dependencies/hardware/data are explicit skips, not passes. Record failures and limitations in reports.

Required recovery cases: half-copied media; modified source evidence; failed model import; resumed video chunk without duplicate events; task cancellation; stale review revision; unavailable source/output path; full disk/export interruption; changed package manifest; rollback to prior model. For UTC minute precision test local schedule boundaries and DST transitions, including local overnight anchoring.
