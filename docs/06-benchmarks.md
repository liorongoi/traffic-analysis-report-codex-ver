# 06 — Model research and evaluation plan

## Primary research starting points (not approved packs)

| Candidate | Technical fit | Remaining evidence |
|---|---|---|
| BDD100K https://github.com/bdd100k/bdd100k | Diverse driving detection/tracking, including night | Moving-camera domain differs from fixed traffic cameras |
| UA-DETRAC https://faculty.ucmerced.edu/mhyang/papers/cviu2020_detrac.pdf | Fixed-view traffic, night/rain/occlusion, broad vehicle classes | Broad truck/other labels do not validate fine vehicle subtypes |
| AI City https://www.aicitychallenge.org/ai-city-challenge-dataset-access/ | Fixed-camera tracking and multi-camera traffic | Absolute UTC provenance and target-camera plate coverage need verification |
| ExDark https://github.com/cs-chan/Exclusively-Dark-Image-Dataset | Supplementary low-light images | Still images alone do not validate continuous video tracking |

The current review increment uses a local COCO-pretrained YOLOX-S ONNX pack from the official OpenCV Zoo distribution; see [vehicle detection evidence](10-vehicle-detection.md) for version, hash, model provenance and functional checks. This pack has not been accepted as an accuracy benchmark or production model. The candidates below are technical research starting points, not selected dependencies. Plate chips are retained alongside optional OCR suggestions; operator-provided camera samples with transcription labels are needed to measure character error and exact-match rates. Camera samples with plate-box labels are needed to evaluate crop proposals in Iran, Lebanon, Israel, and Latvia. Do not invoke hosted Roboflow or any online production API.

The user-provided Iranian plate reference image shows private, disability, public-vehicle, diplomatic, police, army, government, IRGC, taxi, and free-trade-zone designs. Use it as qualitative guidance for collecting chip examples and layout coverage, not as OCR training or evaluation data; do not classify plate types or infer vehicle identity. The shipped PP-OCRv3 Arabic recognizer includes Arabic/Persian letters and Latin/digits; it is general text OCR and has not been validated on Persian-script plates. For operator-provided Iranian crop-review samples, include the broad private-car layout and visible special-color variants when available. Public references differ slightly on physical size (520×110 mm versus 520×113 mm), so chip extraction must not enforce a fixed physical-size/aspect-ratio rule. See [Wikipedia's Iran plate overview](https://en.wikipedia.org/wiki/Vehicle_registration_plates_of_Iran) and the [American Dynamics plate guide](https://docs.johnsoncontrols.com/americandynamics/r/American-Dynamics/en-US/VideoEdge-6.2.0-License-Plate-Guide/A/6.2.0/Middle-East). The reference image itself is not included in the repository or treated as acceptance data.

The user-provided Lebanese plate-reference images show private-vehicle examples and public, governmental, and service variants, including differing colors and commercial/motorcycle layouts. Use them as qualitative guidance for collecting operator-provided samples, not as labeled acceptance data. Local OCR suggestions include Latin and Arabic/Persian letters with digits, but are not validated for Lebanese plate text; the app does not identify plate type or vehicle identity. These images are visual references, not labeled acceptance data, and are not included in the repository.

### Global License Plate Dataset paper (technical reference)

The user-provided [Global License Plate Dataset paper](https://arxiv.org/abs/2405.10949) reports a multi-country collection with plate text, boxes, corner points and segmentation masks. Its evaluation reports whole-plate exact-match accuracy and normalized edit distance. The paper describes 60/20/20 train/validation/test partitions and groups identical or near-identical plate strings at normalized edit distance 0.9 or above to reduce leakage across those partitions. Its technical lessons inform a mission-local exploratory report for OCR suggestions against human-confirmed plate text: crop-level whole-plate exact match, normalized character similarity, corpus character error rate (Levenshtein edits divided by reference characters), and a deterministic one-crop-per-normalized-verified-string summary so repeated observations of the exact same verified string do not receive extra weight. This current summary does not perform the paper's near-duplicate clustering and is not a train/validation/test split. If a plate-recognition training/evaluation split is added, it must group near-identical labels before assigning partitions and keep the test partition untouched. The paper also describes variance of the Laplacian as a blur measure; crop review records and displays this raw value as an optional focus hint. There is no pass/fail cutoff because a useful threshold depends on camera, scale, and capture conditions. Both OCR views remain exploratory, are not protected holdout measurements, and can still reflect selection bias. The PDF supplied in this task is the paper; its image archive is not present in the workspace.

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

Compare a simple baseline with compatible pretrained vehicle/small-object detectors, a light/background branch, tracking, and plate-chip crop proposals. For plates, record crop precision/recall and human approval/rejection rates on operator-provided, country-specific box labels. Measure OCR suggestions against independently human-confirmed plate text using whole-plate exact match, normalized character edit similarity, and corpus character error rate (Levenshtein edits divided by reference characters), separately from crop precision/recall. Preserve protected evaluation splits by related plate strings before claiming generalization. No target-country accuracy is currently measured, and plate identity lookup is out of scope. Record versions/commit/weight SHA/taxonomy/input needs/CPU and GPU memory. Choose based on technical fit and measured evidence. A conversation model interprets queries; it does not produce ground-truth pixel detections.

Frozen dataset manifests: source identifier, retrieval date, file checksums, split IDs, camera/view/time groups, label schema and transformation versions. Keep train/validation/test disjoint by related captures, recordings, and near-duplicate plate strings. Background warm-up rules must be declared; no hidden use of future test frames or labels.

Report vehicle Precision/Recall/F1/AP, per-image count error, candidate false alarms/acceptance, tracking identity consistency/duplicate events, plate-chip crop precision/recall and human approval/rejection, throughput/latency/peak RAM/VRAM/storage. Break down by view/country/condition/profile. Do not use algorithm scores as measured accuracy. Acceptance thresholds await operator agreement.
