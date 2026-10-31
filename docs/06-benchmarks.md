# 06 — Model research and evaluation plan

## Primary research starting points (not approved packs)

| Candidate | Technical fit | Remaining evidence |
|---|---|---|
| BDD100K https://github.com/bdd100k/bdd100k | Diverse driving detection/tracking, including night | Moving-camera domain differs from fixed traffic cameras |
| UA-DETRAC https://faculty.ucmerced.edu/mhyang/papers/cviu2020_detrac.pdf | Fixed-view traffic, night/rain/occlusion, broad vehicle classes | Broad truck/other labels do not validate fine vehicle subtypes |
| AI City https://www.aicitychallenge.org/ai-city-challenge-dataset-access/ | Fixed-camera tracking and multi-camera traffic | Absolute UTC provenance and target-camera plate coverage need verification |
| ExDark https://github.com/cs-chan/Exclusively-Dark-Image-Dataset | Supplementary low-light images | Still images alone do not validate continuous video tracking |
| [Nighttime Vehicle Detection Dataset](https://github.com/ntnu-arl/vehicles-nighttime) | 10,913 grayscale road images with vehicle boxes; includes static roadside ranges suitable for nighttime still-image detection checks | No temporal tracks or UTC labels; sampled labels need box-bound validation, and a small smoke is not target-camera acceptance |

An explicitly ranged COCO converter and class-agnostic detection scoring support this dataset without inventing vehicle subtypes or silently repairing its boxes. A small smoke over 57 images from two static roadside ranges scored the baseline at P/R/F1 0.141/0.197/0.165, AP50 0.067, and 0.91 count MAE at IoU 0.50. This is a preliminary warning about the detector-only nighttime result, not a product acceptance score; see [verification details](09-verification.md) and the saved [report](../reports/nighttime-nvd-yolox-s-smoke.json).

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

### Additional source review (2026-10-31)

- [ForeSpeed](https://github.com/ampedsoftware/forespeed) describes 322 CCTV videos of 14 physical vehicle passes across six camera systems, preserving presentation timestamps for variable-frame-rate videos where available. The repository reports per-pass speed ground truth in the paper appendix. One anonymous 864 KB HEVC Lorex pass now completes the local YOLOX-S/tracker pipeline (40 frames; 59 proposals; two track fragments); it provides a small real-CCTV codec/integration smoke, not a dense traffic or accuracy benchmark. A visible recorder timestamp and filename time exist, but their timezone and mapping to the first PTS have not been independently verified. See [`docs/09-verification.md`](09-verification.md).
- [Intersection Monitoring](https://zenodo.org/records/7727882) describes two fixed infrastructure-camera image sequences at 25 fps with synchronized 20 Hz GNSS vehicle traces and camera calibration. The published time field is HHMMSSss; the record does not establish a date, timezone, or absolute UTC origin, and the GNSS trace covers the test vehicle rather than dense traffic labels. It could support a narrow geometry/timing smoke check after image-to-log alignment, but it is not representative traffic-accuracy or UTC acceptance data.
- [Constellation](https://doi.org/10.1007/s11263-026-03029-1) describes 13,314 human-annotated still frames from one fixed, high-elevation Manhattan intersection across day, night, and weather conditions, with vehicle/pedestrian boxes and time-separated splits. It could support a coarse still-image vehicle-box smoke benchmark after checking anonymous access and coordinate conventions; its labels do not distinguish RoadLoom vehicle subtypes, its public description does not establish absolute UTC-to-PTS mapping or provide a video acceptance fixture, and it says plates are not readable from this view.
- The [Bellevue Traffic Video Dataset](https://github.com/City-of-Bellevue/TrafficVideoDataset) describes approximately 101 hours of 1280x720, 30 Hz fixed traffic-camera footage from five intersections. It is a useful continuous-video candidate for decode/tracking smoke tests, but the repository does not document absolute UTC capture times, video PTS mapping, or matching frame-level labels.
- The [UHD database for smart cities and transport](https://zenodo.org/records/10663664) provides downloadable 4K MP4 clips, including a static-camera roundabout clip. It could exercise high-resolution decoding and detector scaling, but the record does not provide UTC-to-PTS provenance or traffic ground truth. The [DLR Urban Traffic record](https://zenodo.org/records/15754836) exposes a small MP4 preview alongside time-indexed trajectories and other intersection data; the public record does not establish that the preview is synchronized to the trajectory timestamps or maps them to absolute UTC. A 912-frame preview smoke completed, but visual inspection found rendered simulation imagery and YOLOX-S emitted no vehicle proposals at the normal smoke threshold; this validates pipeline execution only, not model or traffic accuracy (details in [`docs/09-verification.md`](09-verification.md)).
- The [University of Glasgow CCTV traffic record](https://researchdata.gla.ac.uk/2043/) describes fixed pole-mounted 1280x720 H.264/MP4 video at 25 fps over 14-hour days, but explicitly says the video is not available for research sharing; it is not an obtainable fixture.
- The [AI City Challenge 2021 data](https://www.aicitychallenge.org/2021-data-and-evaluation/) describes synchronized multi-camera traffic recordings, but access is limited to challenge participants and the public description does not document an absolute UTC-to-PTS origin. It is therefore not an account-free fixture for this project.

These sources do not close the need for authorized representative fixed-camera recordings with independently verified UTC-to-PTS origin and independently labeled traffic events. No candidate has been acquired or accepted as the representative acceptance fixture. Public fixed-camera videos without UTC or labels may still be useful for narrow decode and tracking smoke checks, but must not be described as accuracy or UTC acceptance evidence.

No further reuse-rights research is part of this project task.

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

To compare a fine-tuned detector with the selected baseline, first run the prediction and evaluation commands above for each model against the **same frozen `test` split**, using the same category schema and operating IoU. Then compare the JSON reports:

```powershell
python scripts/compare-detection-benchmarks.py --baseline-report D:\RoadLoom\vehicle-report.json --candidate-report D:\RoadLoom\candidate-report.json --output D:\RoadLoom\model-comparison.json
```

The comparator rejects reports with different ground-truth hashes, image/category schemas, detector kinds, or test splits. It reports candidate-minus-baseline precision, recall, F1, AP, count error, and per-category and available metadata-stratum changes, with regressions listed for review. It does not apply acceptance thresholds or promote/select the candidate; selection remains an operator decision.

For the full baseline/candidate workflow in one command, use the same protected test ZIP with both model packs:

```powershell
python scripts/compare-detection-model-packs.py --dataset-zip D:\RoadLoom\reviewed.zip --baseline-model-pack model-packs\yolox-s --candidate-model-pack D:\RoadLoom\candidate\roadloom-model-pack --baseline-threshold 0.18 --candidate-threshold 0.21 --hardware-profile 16GB --output-dir D:\RoadLoom\candidate-comparison
```

Choose each confidence threshold using validation data before evaluating on the held-out test role. The command runs each model over the same test images, saves separate prediction and evaluation JSON files, then writes `comparison.json`. Its output directory must be new or empty to avoid replacing earlier reports. Vehicle tiling is on by default for both packs; `--no-tiles` disables it for both. Optional image metadata supplies country, day/night condition, and camera view for stratified comparisons. The report remains review-only and never promotes a model.

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

The report includes class-aware (or explicitly selected class-agnostic) per-frame box precision/recall, ID switches, fragmentations, MOTA, global-assignment ID precision/recall/IDF1, track-count error, and the share of ground-truth tracks split across multiple predicted IDs. That last value is a duplicate-track indicator, not a universal duplicate-event rate; event-level acceptance still needs a frozen definition and representative annotated sequences. Matching is within one sequence only: the evaluator never assigns identities across cameras or recordings. It calculates metrics and supplies no acceptance thresholds.

For [RoundaboutHD](https://researchdata.bath.ac.uk/1574/), the repository includes an offline converter from the published SCT row format (`frame_id object_id xmin ymin xmax ymax`) to the same tracking manifest. Supply the exact video frame count and dimensions. Frame numbering defaults to zero-based; set `--frame-offset` explicitly if a prepared clip begins at another source frame. The converter emits every frame, including empty frames, namespaces track IDs per camera, checks pixel boxes, and records each annotation file's SHA-256.

```powershell
python scripts/convert-roundabouthd-tracking.py --labels "c001=D:\RoundaboutHD\imagesc001\SCT\imagesc001_SCT_GT.txt" --frame-count 9000 --width 3840 --height 2160 --output D:\RoadLoom\c001-ground-truth.json
python scripts/export-video-tracking-run.py --database D:\RoadLoom\missions\MISSION-ID\mission.sqlite3 --run-id RUN-ID --sequence-id c001 --output D:\RoadLoom\c001-predictions.json
python scripts/evaluate-tracking-benchmark.py --ground-truth D:\RoadLoom\c001-ground-truth.json --predictions D:\RoadLoom\c001-predictions.json --class-agnostic --report D:\RoadLoom\c001-report.json
```

The public SCT labels do not supply vehicle subtype categories, so `--class-agnostic` deliberately scores localization and identity without penalizing car/truck/bus label differences. The report records this matching mode. Use this only when the selected ground truth has no subtype labels; the default evaluation remains class-aware. Process the corresponding full video and verify its actual frame count/dimensions before scoring. RoundaboutHD's public archive is 8 GB and has no documented UTC-to-PTS origin; conversion support does not mean its video was fetched, evaluated, or accepted as UTC evidence.

### Public video candidates for development fixtures

- The [Zenodo Dataset for Traffic Evaluation](https://zenodo.org/records/14041920) provides monitored-road H.264 clips; filenames include date/time strings and the record lists 9.4 GB of files. One 266-frame clip has already passed the local decoder, detector, tracker, and durable-worker path (see [video smoke evidence](09-verification.md)). The record does not identify the filename timezone or verify the UTC-to-PTS origin, and this clip has no matching vehicle ground truth in the record. It is integration evidence only.
- The [UHD database focus on smart cities and smart transport](https://zenodo.org/records/10663664) includes `Sc10_Traffic roundabout_static camera.mp4`, a downloadable 3840×2160 HEVC sample with 500 frames at 50 fps. All frames decoded locally with increasing PTS; the bundled YOLOX-S ran with tiling on one frame and produced 69 unscored proposals, which the tracker accepted as one-frame observations. This confirms a 4K HEVC decode and single-frame model/tracker smoke only; it has no verified absolute UTC origin or matching vehicle boxes and does not establish detector quality or sustained throughput (see [verification](09-verification.md)).
- The official [GRAM Road-Traffic Monitoring dataset](https://gram.web.uah.es/data/datasets/rtm/index.html) describes three manually annotated traffic sequences with vehicle IDs; its Urban1 sequence is a fixed surveillance view at 600×360 and 25 fps. The listed archive links redirected to Microsoft sign-in in this session. No account was used and no files were fetched, so it remains an access-gated lead rather than project evidence.
- The [Highway Vehicle Surveillance Dataset (HVSD)](https://data.mendeley.com/datasets/zdgmdg7p25/1) describes day, night, and rainy highway videos with annotated boxes and vehicle IDs. Its record does not document absolute capture-UTC to video-PTS origin. The record page shows a sign-in/register prompt; an account-free download was not verified, and no files were fetched.
- The [MIT Traffic Dataset](https://mmlab.ie.cuhk.edu.hk/datasets/mit_traffic/index.html) provides 90 minutes from a stationary 720×480 camera split into 20 clips. The published ground truth covers sampled pedestrians, not vehicle tracks, and the page does not establish absolute UTC origin.
- [RoundaboutHD](https://researchdata.bath.ac.uk/1574/) describes 40 minutes from four fixed-position 4K/15fps cameras with 512 annotated vehicle identities and an 8 GB archive. Its page does not document absolute capture UTC or PTS alignment; cross-camera identity scoring is outside this evaluator.
- The [ForeSpeed dataset](https://github.com/ampedsoftware/forespeed) publishes 322 CCTV videos of 14 physical vehicle passes recorded across different camera models and formats. Its README says original video timing, including variable-frame-rate PTS where available, is preserved, and the paper appendix contains reference speeds. This is a useful candidate for codec, PTS, and event-timing regression fixtures, but the staged passes are not continuous traffic monitoring, and the published description does not establish capture UTC mapped to each video's PTS. No files have been fetched or evaluated.

These are development leads only. Target-camera acceptance still needs representative footage, independently verified frame/time provenance, labeled plate boxes, and product-approved thresholds.


### Public-data access check — 2026-10-31

- A [Roboflow Universe Lebanese Plates Detection project](https://universe.roboflow.com/gg-6ss5j/lebanese-plates-detection) lists a 328-image plate-box dataset version. The page's sample detector uses a credentialed hosted API; an account-free dataset export was not verified, no images or labels were fetched, and the page's published metrics are not RoadLoom evidence. It remains a possible localization-data lead only.
- The [IR-LPR project](https://github.com/mut-deep/IR-LPR) describes 20,967 Iranian car images with whole-plate and character detection annotations. Its whole-car test archive is listed at 2.33 GB and the isolated plate-image test archive at 53.8 MB; both are linked through Google Drive. Opening the plate test archive prompts for Google sign-in. Under the user's no-account preference, no files were downloaded. If later available without login, whole-plate boxes could support region-only Iranian crop scoring by discarding all character labels; they would not establish fixed-camera accuracy or cover Lebanon, Israel, or Latvia.
- The official [I-24 MOTION data documentation](https://github.com/I24-MOTION/I24M_documentation) describes products from a fixed highway camera network. The [system paper](https://arxiv.org/abs/2301.11198) describes video frames with PTS timing, but the public products surfaced in this search are trajectories, not raw camera recordings; they cannot serve as RoadLoom decoder or plate-crop fixtures. Existing fixed-camera candidates remain insufficient for verified UTC-to-PTS and target-country plate acceptance.

A short anonymous ForeSpeed Eufy T2P1 sample provides a narrow plate-crop integration fixture: its 2304×1296 frame 450 yielded one 40×22-pixel crop candidate at score 0.218 with 768-pixel overlapping tiles and threshold 0.18, while the earlier full-frame pass at 0.30 found none. See [verification](09-verification.md). This is a threshold/tile-path smoke only; it supplies no labeled boxes, acceptance metrics, or country coverage.
