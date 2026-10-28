# 02 — Open decisions and acceptance inputs

This register tracks decisions and evidence that code alone cannot supply. The user asked us to use supplied research materials for technical learning and not investigate reuse rights. This project does not perform further reuse-rights research; representative media and target-machine details must come from the user or the deployment team.

## Settled product decisions

- The app is an offline-first, local road-traffic review and analysis tool, with opt-in authorized LAN access. There is no application sign-in flow.
- The UI is bilingual Hebrew/English and supports RTL/LTR.
- Plate work includes localization, image-chip extraction, optional local OCR suggestions, and separate human crop/text approval. Each new crop snapshots the mission's operator-configured country as context and labels its basis; this is not image-derived country or layout inference. Never infer vehicle identity. The user superseded the earlier crop-only instruction and authorized a usable local LPR algorithm. Do not add plate-type, town, governorate, or identity inference.
- The supplied Iran and Lebanon plate charts are qualitative visual guidance only. They are not camera samples or OCR ground truth.
- Video identity correction uses corrected identities: a merge creates one corrected logical event; a split creates child events; original records remain auditable.
- Original sources and append-only human review history are preserved. No automatic deletion or retention period has been approved.
- Installation is current-user only, requires no administrator privileges, and must present one installer.

## Inputs required to close acceptance

- **Representative video and time evidence:** supply fixed-camera recordings with codecs, duration/frame-rate details, and independently verified UTC-to-PTS origin. The 143 received Fintraffic JPGs provide still-image/timestamp examples, but do not exercise video tracking, playback, or clips. Existing technical leads include MIT Traffic's stationary-camera clips without verified UTC, and VIDETEC-2's infrastructure-camera clips with UTC sensor timestamps but no verified camera-video-to-PTS mapping. Moving-camera UTC datasets can exercise timestamp parsing but cannot replace fixed-camera acceptance.
- **Plate benchmark media:** supply real camera crops and independently confirmed plate boxes/text for Iran, Lebanon, Israel, and Latvia, with day/night, distance, blur, occlusion, and plate-layout variety. The supplied charts do not measure localization or OCR accuracy. The GLPD PDF is a research paper, not its image archive; its 60/20/20 split and near-duplicate grouping at 0.90 normalized edit similarity are technical guidance, not RoadLoom results.
- **Target workstation:** run install, offline launch, model inference, monitoring, recovery, and authorized-LAN checks on the intended Windows 16/48 GB configurations and actual GPU/CPU. CI package verification is not a substitute for this hardware acceptance.
- **Product thresholds:** specify acceptable vehicle detection/count error, false positives/misses, tracking duplicate rate, latency/throughput, nighttime behavior, and model-promotion evidence. No numeric thresholds are approved, so the app must continue to label predictions as proposals.
- **Retention and storage:** specify retention duration and storage budget before any automatic cleanup or source-deletion behavior is added. Current behavior preserves evidence.
- **LAN operations:** verify the intended authorized network shares, credential handling, browser access, and expected behavior during share/network outages on the target Windows setup.

## Implemented behavior that remains subject to acceptance

- Still-image mission setup, import, monitoring, inference queueing, review, ROI/view assignment, plate chips/OCR suggestions, and exports are implemented with automated coverage; real camera accuracy and full operator acceptance remain open.
- Video decode, tracking, event review, corrected merge/split identities, playback, and clip/event export are implemented with synthetic coverage; representative-video, codec, UTC alignment, long/VFR, occlusion, and native Windows acceptance remain open.
- The learning system supports explicit night-light-head fitting and evaluation, protected image evidence roles, and controlled promotion/rollback for that adapter. YOLOX fine-tuning and numeric promotion rules remain unimplemented; eligible representative training data is required before adding or validating a plate/vehicle training adapter.
- The GLPD-informed plate report is exploratory: it reports exact match, normalized edit similarity, character error, a focus reference, and separate one-crop summaries grouped by exact string and by transitive near-duplicate verified-text groups at similarity 0.90. These groups reduce repeated-string weighting but do not create train/validation/test partitions or held-out accuracy evidence.
