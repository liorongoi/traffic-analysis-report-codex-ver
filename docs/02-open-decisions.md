# 02 — Open decisions

## Awaiting operator input

- Confirmed: bilingual Hebrew/English with RTL/LTR. Filenames, timestamps and mixed-script OCR require directional isolation.
- Received: 143 Fintraffic JPG samples with manifest checksums and timestamps. Still needed: representative videos, codecs, durations, video UTC origin, daily volume and expected history size.
- Confirmed: installation must not require administrator rights. Source folders may be local or authorized network shares; credential handling and network-disconnection behavior require Windows testing.
- Review/export retention and immutable evidence storage budget; no source deletion policy is approved.
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
