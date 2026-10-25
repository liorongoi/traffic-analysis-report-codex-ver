# 02 — Open decisions

## Awaiting operator input

- Confirmed: bilingual Hebrew/English with RTL/LTR. Filenames, timestamps and mixed-script OCR require directional isolation.
- Received: 143 Fintraffic JPG samples with manifest checksums and timestamps. Still needed: representative videos, codecs, durations, video UTC origin, daily volume and expected history size.
- Web research (2026-10-25): Zenseact Open Dataset (ZOD) is a candidate for local UTC-timed driving footage: 1,473 sequences are 20 seconds each, camera JPGs are captured at about 10 Hz, and GNSS/IMU data carries UTC timestamps. The dataset is CC BY-SA 4.0, but download access requires approval; the sequence is images plus sensor metadata, not a ready-made MP4. A local test clip could be assembled only after approved access and frame-to-UTC alignment is checked. It is dashcam rather than fixed-camera footage. NATIX is another gated, non-commercial dashcam candidate with frame-matched GPS metadata; its sample files and timestamp offsets still require approval and local verification, and its license prohibits redistribution. AI City 2019 better matches fixed intersection/highway cameras and gives synchronization offsets, but requires a participation agreement and does not establish an absolute UTC origin. Zenodo's traffic-evaluation MP4 filenames contain apparent dates/times, but the record does not declare a license or timezone. No candidate has yet been acquired and validated as an acceptance fixture; do not treat undocumented filename times as UTC or use unclear-rights media.
- Confirmed: installation must not require administrator rights. Source folders may be local or authorized network shares; credential handling and network-disconnection behavior require Windows testing.
- Review/export retention and immutable evidence storage budget; no source deletion policy is approved.
- Source rename reconciliation: identical bytes may represent distinct observations, so checksum equality alone cannot authorize merging or suppressing an import. A manual link policy must identify the source snapshot/path evidence and preserve both capture records.
- Country-specific plate variants and authorized ground truth for all four countries.
- Nighttime candidate sensitivity, definition of a confirmed vehicle, and acceptable false-positive/miss/count-error rates.
- Target processing latency/throughput and required video correction operations. Clip context is explicitly entered per export (0–600 seconds before/after); no unstated default is applied.
- Training evidence thresholds, dataset reuse policy and model-promotion criteria. No inherited SpectraScope thresholds.

## Proposed engineering choices (not business facts)

- Python domain/worker layer, FastAPI API and a lightweight browser UI; dependencies will be pinned during a separate staging step.
- Local per-mission SQLite with a sole backend writer and transactions. A small global mission registry contains only IDs/paths, not merged mission results.
- Half-open schedule intervals [start, end), local weekday referring to the start day; overnight end times belong to the next day. All-day windows require an explicit option, not equal start/end times.
- Exclusion masks override inclusion masks. Camera-view/ROI changes create configuration versions; they do not rewrite reviewed runs.
- One processing task at a time initially, with inference/review priority over background training. Real target profiling may justify more concurrency.
