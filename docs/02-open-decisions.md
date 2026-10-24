# 02 — Open decisions

## Awaiting operator input

- Confirmed: bilingual Hebrew/English with RTL/LTR. Filenames, timestamps and mixed-script OCR require directional isolation.
- Received: 143 Fintraffic JPG samples with manifest checksums and timestamps. Still needed: representative videos, codecs, durations, video UTC origin, daily volume and expected history size.
- Web search (2026-10-25): NATIX Multi-Camera Driving Dataset is the strongest located timestamped MP4 candidate. Its card describes approximately one-minute dashcam clips, per-camera frame-matched GPS CSVs with ISO-8601 timestamps, trip `startEpochMs`/`endEpochMs` Unix times, and an IANA timezone. The page says its public sample is about 20 minutes across six complete trips, but access to files requires approval and credentials. The NATIX Data RAIL-NC license limits use to non-commercial purposes and prohibits redistribution; never commit its footage or share it onward. It is driving footage, not fixed traffic-camera footage. The timestamp strings' explicit UTC offsets, clip-to-trip mapping and actual frame alignment still need inspection on an approved local copy; the card warns GPS/video synchronization may be off by 0–1 seconds, and up to 3 seconds in extreme cases. The AI City Challenge 2019 dataset better matches fixed intersection/highway cameras and provides synchronization offsets, but its rules require a participation agreement and describe relative offsets, not a verified absolute UTC origin. The Zenodo Dataset for Traffic Evaluation has road-monitoring MP4s with date/time-looking filenames, but the record does not declare a license or timezone. None is yet an acquired/validated acceptance fixture; do not use the latter two until terms/metadata are confirmed.
- Confirmed: installation must not require administrator rights. Source folders may be local or authorized network shares; credential handling and network-disconnection behavior require Windows testing.
- Review/export retention and immutable evidence storage budget; no source deletion policy is approved.
- Source rename reconciliation: identical bytes may represent distinct observations, so checksum equality alone cannot authorize merging or suppressing an import. A manual link policy must identify the source snapshot/path evidence and preserve both capture records.
- Country-specific plate variants and authorized ground truth for all four countries.
- Nighttime candidate sensitivity, definition of a confirmed vehicle, and acceptable false-positive/miss/count-error rates.
- Target processing latency/throughput, clip pre/post context, and required video correction operations.
- Training evidence thresholds, dataset reuse policy and model-promotion criteria. No inherited SpectraScope thresholds.

## Proposed engineering choices (not business facts)

- Python domain/worker layer, FastAPI API and a lightweight browser UI; dependencies will be pinned during a separate staging step.
- Local per-mission SQLite with a sole backend writer and transactions. A small global mission registry contains only IDs/paths, not merged mission results.
- Half-open schedule intervals [start, end), local weekday referring to the start day; overnight end times belong to the next day. All-day windows require an explicit option, not equal start/end times.
- Exclusion masks override inclusion masks. Camera-view/ROI changes create configuration versions; they do not rewrite reviewed runs.
- One processing task at a time initially, with inference/review priority over background training. Real target profiling may justify more concurrency.
