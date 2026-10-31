# 06 — Model research and evaluation plan

## Primary research starting points (not approved packs)

| Candidate | Technical fit | Remaining evidence |
|---|---|---|
| BDD100K https://github.com/bdd100k/bdd100k | Diverse driving detection/tracking, including night | Moving-camera domain differs from fixed traffic cameras |
| UA-DETRAC https://faculty.ucmerced.edu/mhyang/papers/cviu2020_detrac.pdf | Fixed-view traffic, night/rain/occlusion, broad vehicle classes | Broad truck/other labels do not validate fine vehicle subtypes |
| AI City https://www.aicitychallenge.org/ai-city-challenge-dataset-access/ | Fixed-camera tracking and multi-camera traffic | Absolute UTC provenance and target-camera plate coverage need verification |
| ExDark https://github.com/cs-chan/Exclusively-Dark-Image-Dataset | Supplementary low-light images | Still images alone do not validate continuous video tracking |

The current review increment uses a local COCO-pretrained YOLOX-S ONNX pack from the official OpenCV Zoo distribution; see [vehicle detection evidence](10-vehicle-detection.md) for version, hash, model provenance and functional checks. This pack has not been accepted as an accuracy benchmark or production model. The candidates below are technical research starting points, not selected dependencies. Plate chips are retained for operator approval or rejection. Camera samples with human-labeled plate boxes are needed to measure crop localization precision and recall in Iran, Lebanon, Israel, and Latvia. Do not invoke hosted Roboflow or any online production API.

The user-provided Iranian plate reference image shows private, disability, public-vehicle, diplomatic, police, army, government, IRGC, taxi, and free-trade-zone designs. Use it as qualitative guidance for collecting chip examples and layout coverage, not as labeled acceptance data; do not classify plate types or infer vehicle identity. For operator-provided Iranian crop-review samples, include private-car layouts and visible plate variants where available. Public references differ slightly on physical size (520×110 mm versus 520×113 mm), so chip extraction must not enforce a fixed physical-size/aspect-ratio rule. See [Wikipedia's Iran plate overview](https://en.wikipedia.org/wiki/Vehicle_registration_plates_of_Iran) and the [American Dynamics plate guide](https://docs.johnsoncontrols.com/americandynamics/r/American-Dynamics/en-US/VideoEdge-6.2.0-License-Plate-Guide/A/6.2.0/Middle-East). The reference image itself is not included in the repository or treated as acceptance data.

The user-provided Lebanese plate-reference images show private-vehicle examples and public, governmental, and service variants, including differing colors and commercial/motorcycle layouts. Use them as qualitative guidance for collecting operator-provided samples, not as labeled acceptance data. The app extracts image chips for operator review and does not identify plate type or vehicle identity. These images are visual references, not labeled acceptance data, and are not included in the repository.

### Global License Plate Dataset paper (technical reference)

The user-provided [Global License Plate Dataset paper](https://arxiv.org/abs/2405.10949) reports a multi-country collection with plate text, boxes, corner points and segmentation masks. Its evaluation reports whole-plate exact-match accuracy and normalized edit distance. The paper describes 60/20/20 train/validation/test partitions and groups identical or near-identical plate strings at a 0.90 normalized edit-similarity threshold to reduce leakage across those partitions. The paper's text-recognition metrics and image archive are outside this project's plate-chip workflow. Only its plate-box and split-leakage discussion may inform future localization evaluation; no text recognition or text-label workflow is planned. The paper also describes variance of the Laplacian as a blur measure; crop review records and displays this raw value as an optional focus hint. There is no pass/fail cutoff because a useful threshold depends on camera, scale, and capture conditions. The PDF supplied in this task is the paper; its image archive is not present in the workspace.

## Video integration candidates

- [Open Skoltech Car (OS:Car)](https://newpavlov.github.io/oscar-website/) documents frame filenames with Unix timestamps in microseconds and assembled WebM sequences. It is moving-car winter-road footage; frame-to-UTC alignment still needs verification and it does not represent a fixed traffic camera.
- [Zenseact Open Dataset (ZOD)](https://zod.zenseact.com/) describes UTC GNSS/IMU with sampled dashcam images. Frame-to-time alignment needs verification; the moving-camera view is not fixed-camera acceptance evidence.
- [AI City 2019](https://www.aicitychallenge.org/2019-data-sets/) better matches fixed traffic cameras and supplies relative synchronization offsets, but does not establish absolute UTC.
- The [Zenodo traffic-evaluation videos](https://zenodo.org/records/14041920) describe monitored-road video but do not provide verified UTC origin in the public record.


### Additional source review (2026-10-26)

- The [University of Bath RoundaboutHD record](https://researchdata.bath.ac.uk/1574/) describes 40 minutes of 4K/15 fps footage from four fixed-position traffic cameras, with vehicle tracking annotations. The record does not document an absolute UTC origin or video-PTS-to-UTC mapping.
- The [AI City 2019 dataset page](https://www.aicitychallenge.org/2019-data-sets/) describes synchronized fixed-camera videos and per-video start offsets, but does not establish absolute UTC capture time.
- The [Zenodo Traffic Evaluation v2 record](https://zenodo.org/records/14041920) includes long monitored-road videos, but its public record does not provide verified capture-time provenance.

These sources are technical leads only. Their camera view, actual files, labels, codecs, and time mapping would need to be inspected before treating any sequence as test evidence.

### Additional clip search (2026-10-27)

- The [2K Dashcam Traffic Surveillance Video Corpus](https://huggingface.co/datasets/thundarstrom/traffic-surveillance-video-corpus) describes 50-minute drives but currently reports no files. It is dashcam footage and does not document a UTC-to-video-PTS mapping.
- The [TIme Square Intersection dataset](https://personal.ie.cuhk.edu.hk/~ccloy/downloads_qmul_TISI_dataset.html) provides GMT-named image sequences at approximately 10 Hz rather than video clips.
- The official [AAU MTID page](https://vap.aau.dk/mtid/) links to a multi-view traffic dataset hosted through Kaggle; timezone-aware capture/PTS mapping is not established in the public description.

No candidate has been acquired or accepted as a representative fixed-camera/verified-UTC fixture. No further reuse-rights research is part of this project task.

## Selection and evidence

Compare a simple baseline with compatible pretrained vehicle/small-object detectors, a light/background branch, tracking, and plate-chip crop proposals. For plates, record crop precision/recall and human approval/rejection rates on operator-provided, country-specific box labels. Preserve camera/view/source separation for held-out localization evaluation. No target-country accuracy is currently measured, and plate identity lookup is out of scope. Record versions/commit/weight SHA/taxonomy/input needs/CPU and GPU memory. Choose based on technical fit and measured evidence. A conversation model interprets queries; it does not produce ground-truth pixel detections.

Frozen dataset manifests: source identifier, retrieval date, file checksums, split IDs, camera/view/time groups, label schema and transformation versions. Keep train/validation/test disjoint by related captures, recordings, and near-duplicate plate strings. Background warm-up rules must be declared; no hidden use of future test frames or labels.

Report vehicle Precision/Recall/F1/AP, per-image count error, candidate false alarms/acceptance, tracking identity consistency/duplicate events, plate-chip crop precision/recall and human approval/rejection, throughput/latency/peak RAM/VRAM/storage. Break down by view/country/condition/profile. Do not use algorithm scores as measured accuracy. Acceptance thresholds await operator agreement.

## Offline detection benchmark commands

Use a frozen RoadLoom COCO ZIP split to generate predictions and score them without network access:

```powershell
python scripts/predict-detection-benchmark.py --dataset-zip D:\\RoadLoom\\reviewed.zip --model-pack model-packs\\yolox-s --model-kind vehicle --split test --hardware-profile 16GB --output D:\\RoadLoom\\vehicle-predictions.json
python scripts/evaluate-detection-benchmark.py --ground-truth D:\\RoadLoom\\reviewed.zip --predictions D:\\RoadLoom\\vehicle-predictions.json --split test --report D:\\RoadLoom\\vehicle-report.json
```

For plate-region chips, set `--model-kind plate` and select the plate-localization model pack. The evaluator reports box precision/recall/F1, AP50 and mAP50:95, false alarms per image, per-image count error, available country/condition/view/hardware strata, model initialization time, end-to-end split throughput, per-image median/p95 latency, peak process RSS, and model/dataset/prediction storage bytes. Peak VRAM is reported as unavailable when the selected ONNX provider does not expose a portable measurement. An optional per-image metadata JSON can supply `country`, `condition`, `camera_view`, and `hardware_profile`. The vehicle prediction command measures the standalone YOLOX detector; it does not measure the separate night-light hypotheses, background logic, scan masks, ROI restriction, road masks, or complete traffic pipeline. Plate scoring measures region localization only and never reads plate text. These commands calculate metrics but do not establish accuracy until run against a complete, independently labeled representative test split.

## Sequence-level tracking benchmark

Use paired JSON manifests with one sequence per recorded camera and one object list per frame. Ground-truth objects use `track_id`, `category_id`, and an original-frame `bbox` in `[x, y, width, height]` pixel coordinates. The prediction manifest can come directly from a completed RoadLoom run; the exporter converts the saved normalized boxes using the source's recorded dimensions and retains model/source hashes. Prediction and ground-truth files must contain identical sequence and frame IDs.

Example ground-truth shape:

```json
{"sequences":[{"sequence_id":"camera-a.mp4","metadata":{"country":"Lebanon","condition":"night","camera_view":"north-1","hardware_profile":"conservative"},"frames":[{"frame_id":0,"objects":[{"track_id":"vehicle-1","category_id":"car","bbox":[120,80,64,42]}]}]}]}
```

Export saved predictions and score the two files:

```powershell
python scripts/export-video-tracking-run.py --database D:\\RoadLoom\\missions\\MISSION-ID\\mission.sqlite3 --run-id RUN-ID --output D:\\RoadLoom\\tracking-predictions.json
python scripts/evaluate-tracking-benchmark.py --ground-truth D:\\RoadLoom\\tracking-ground-truth.json --predictions D:\\RoadLoom\\tracking-predictions.json --iou-threshold 0.5 --report D:\\RoadLoom\\tracking-report.json
```

The report includes class-aware per-frame box precision/recall, ID switches, fragmentations, MOTA, global-assignment ID precision/recall/IDF1, track-count error, and the share of ground-truth tracks split across multiple predicted IDs. That last value is a duplicate-track indicator, not a universal duplicate-event rate; event-level acceptance still needs a frozen definition and representative annotated sequences. Matching is within one sequence only: the evaluator never assigns identities across cameras or recordings. It calculates metrics and supplies no acceptance thresholds.

### Public video candidates for development fixtures

- The [Highway Vehicle Surveillance Dataset (HVSD)](https://data.mendeley.com/datasets/zdgmdg7p25/1) describes day, night, and rainy highway videos with annotated boxes and vehicle IDs. Its public record does not document absolute capture-UTC to video-PTS origin, so it may inform detector/tracker exercises but cannot close time acceptance.
- The [MIT Traffic Dataset](https://mmlab.ie.cuhk.edu.hk/datasets/mit_traffic/index.html) provides 90 minutes from a stationary 720×480 camera split into 20 clips. The published ground truth covers sampled pedestrians, not vehicle tracks, and the page does not establish absolute UTC origin.
- [RoundaboutHD](https://researchdata.bath.ac.uk/1574/) describes 40 minutes from four fixed-position 4K/15fps cameras with 512 annotated vehicle identities and an 8 GB archive. Its page does not document absolute capture UTC or PTS alignment; cross-camera identity scoring is outside this evaluator.

These are development leads only. Target-camera acceptance still needs representative footage, independently verified frame/time provenance, labeled plate boxes, and product-approved thresholds.
