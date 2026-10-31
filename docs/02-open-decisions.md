# 02 — Open decisions and acceptance inputs

This register tracks decisions and evidence that code alone cannot supply. The user asked us to use supplied research materials for technical learning and not investigate reuse rights. This project does not perform further reuse-rights research; representative media and target-machine details must come from the user or the deployment team.

## Settled product decisions

- The app is an offline-first, local road-traffic review and analysis tool, with opt-in authorized LAN access. There is no application sign-in flow.
- The UI is bilingual Hebrew/English and supports RTL/LTR.
- Plate work is limited to plate-region localization, image-chip extraction, and separate human crop approve/reject decisions. The latest user instruction supersedes the earlier LPR request: do not recognize or export plate text and never infer vehicle identity, plate type, town, or governorate. Mission country remains operator-configured context, not image-derived identification.
- The supplied Iran and Lebanon plate charts are qualitative visual guidance only, not camera samples or localization ground truth.
- Video identity correction uses corrected identities: a merge creates one corrected logical event; a split creates child events; original records remain auditable.
- Original sources and append-only human review history are preserved. No automatic deletion or retention period has been approved.
- Installation is current-user only, requires no administrator privileges, and must present one installer.

## Inputs required to close acceptance

- **Representative video and time evidence:** provide fixed-camera recordings with codecs, duration/frame-rate details, and independently verified UTC-to-PTS origin. The 143 received Fintraffic JPGs provide still-image/timestamp examples, but do not exercise video tracking, playback, or clips. The [Zenodo Dataset for Traffic Evaluation](https://zenodo.org/records/14041920) supplied a 266-frame monitored-road integration smoke; its timestamped filenames do not identify a timezone or establish the UTC-to-PTS mapping, and no matching vehicle ground truth is listed. The [GRAM-RTM dataset](https://gram.web.uah.es/data/datasets/rtm/index.html) describes a manually annotated fixed-camera sequence, but its download link redirected to Microsoft sign-in here, so no files were fetched. MIT Traffic has stationary-camera clips without verified UTC, and VIDETEC-2 has UTC sensor timestamps without verified camera-video-to-PTS mapping. Moving-camera UTC datasets can exercise timestamp parsing but cannot replace fixed-camera acceptance.
- **Plate benchmark media:** provide representative camera images with independently labeled plate boxes for Iran, Lebanon, Israel, and Latvia, covering day/night, distance, blur, occlusion, and plate layouts. The supplied charts do not measure localization quality. The GLPD PDF is research context, not its image archive or RoadLoom acceptance evidence.
- **Target workstation:** run install, offline launch, model inference, monitoring, recovery, and authorized-LAN checks on the intended Windows 16/48 GB configurations and actual GPU/CPU. CI package verification is not a substitute for this hardware acceptance.
- **Product thresholds:** specify acceptable vehicle detection/count error, false positives/misses, tracking duplicate rate, latency/throughput, nighttime behavior, and model-promotion evidence. No numeric thresholds are approved, so the app must continue to label predictions as proposals.
- **Retention and storage:** specify retention duration and storage budget before any automatic cleanup or source-deletion behavior is added. Current behavior preserves evidence.
- **LAN operations:** verify the intended authorized network shares, credential handling, browser access, and expected behavior during share/network outages on the target Windows setup.

## Implemented behavior that remains subject to acceptance

- Still-image mission setup, import, monitoring, inference queueing, review, ROI/view assignment, plate-chip extraction and crop approval, and exports have automated coverage; real camera accuracy and full operator acceptance remain open.
- Video decode, tracking, event review, corrected merge/split identities, playback, and clip/event export are implemented with synthetic coverage; representative-video, codec, UTC alignment, long/VFR, occlusion, and native Windows acceptance remain open.
- The learning system supports explicit night-light-head fitting and evaluation, protected image evidence roles, and controlled promotion/rollback for that adapter. The optional YOLOX-S fine-tuning CLI consumes protected training/validation COCO splits and exports a local ONNX pack for operator import and selection on future still-image and video runs; each run freezes its model ID/hash, and video resume resolves the run’s original model even after the mission default changes. `scripts/stage-yolox-training.py` packages a preprepared CUDA-enabled portable Windows runtime, YOLOX 0.3.0 source, trainer CLI and baseline into a separate hashed transfer bundle without downloads or authentication. This bundle stays outside the single core installer. The offline detection evaluator and baseline/candidate report comparator are implemented, but representative held-out detector scoring is not yet available. Actual CUDA training on the target, numerical promotion criteria and representative reviewed data remain unverified.
- Plate acceptance will measure localization precision/recall and crop approval/rejection against camera images with human-labeled plate boxes; no text-recognition metric or text label is part of the workflow.
