# 06 — Model research and evaluation plan

## Primary research starting points (not approved packs)

| Candidate | Scope | Required gate |
|---|---|---|
| BDD100K https://github.com/bdd100k/bdd100k | Diverse driving detection/tracking, night | Data license differs from BSD code; commercial use needs appropriate rights; moving-camera domain mismatch |
| UA-DETRAC https://faculty.ucmerced.edu/mhyang/papers/cviu2020_detrac.pdf | Fixed-view traffic, night/rain/occlusion, broad classes | Verify current data access and terms; broad truck/other labels do not validate semitrailers |
| AI City https://www.aicitychallenge.org/ai-city-challenge-dataset-access/ | Vehicle tracking and language retrieval | Select an actually available track and check its specific terms |
| ExDark https://github.com/cs-chan/Exclusively-Dark-Image-Dataset | Supplementary low-light images | Current terms require written commercial permission; third-party photo rights separate |

The current review increment uses a local COCO-pretrained YOLOX-S ONNX pack from the official OpenCV Zoo distribution; see [vehicle detection evidence](10-vehicle-detection.md) for version, hash, license notes, and functional checks. This pack has not been accepted as an accuracy benchmark or production model. The candidates below remain research starting points, not approved datasets or dependencies. Plate chips are retained alongside optional OCR suggestions; authorized camera samples with transcription labels are needed to measure character error and exact-match rates. Authorized camera samples with plate-chip box labels are needed to evaluate crop proposals in Iran, Lebanon, Israel, and Latvia. Do not invoke hosted Roboflow or any online production API.

The user-provided Iranian plate reference image shows private, disability, public-vehicle, diplomatic, police, army, government, IRGC, taxi, and free-trade-zone designs. Use it as qualitative guidance for collecting chip examples and layout coverage, not as OCR training or evaluation data; do not classify plate types or infer vehicle identity. The shipped PP-OCRv3 Arabic recognizer includes Arabic/Persian letters and Latin/digits; it is general text OCR and has not been validated on Persian-script plates. Persian digit accuracy remains unverified. For authorized Iranian crop-review samples, include the broad private-car layout and visible special-color variants when available. Public references differ slightly on physical size (520×110 mm versus 520×113 mm), so chip extraction must not enforce a fixed physical-size/aspect-ratio rule. See [Wikipedia's Iran plate overview](https://en.wikipedia.org/wiki/Vehicle_registration_plates_of_Iran) and the [American Dynamics plate guide](https://docs.johnsoncontrols.com/americandynamics/r/American-Dynamics/en-US/VideoEdge-6.2.0-License-Plate-Guide/A/6.2.0/Middle-East). The reference image itself is not included in the repository or treated as acceptance data.

The user-provided Lebanese plate-reference images show private-vehicle examples and public, governmental, and service variants, including differing colors and commercial/motorcycle layouts. Use them as qualitative guidance for collecting authorized samples, not as labeled acceptance data. Local OCR suggestions include Latin and Arabic/Persian letters with digits, but are not validated for Lebanese plate text; the app does not identify plate type or vehicle identity. These images are visual references, not labeled acceptance data, and are not included in the repository.

### Global License Plate Dataset paper (technical reference)

The user-provided [Global License Plate Dataset paper](https://arxiv.org/abs/2405.10949) reports a multi-country collection with plate text, boxes, corner points and segmentation masks. Its evaluation reports whole-plate exact-match accuracy and normalized edit distance, and its split construction groups identical or near-identical plate strings to limit leakage. Its technical lessons now inform a mission-local exploratory report for OCR suggestions against human-confirmed plate text. The report includes crop-level whole-plate exact match, normalized character similarity, and corpus character error rate (Levenshtein edits divided by reference characters), plus a deterministic one-crop-per-normalized-verified-string summary so repeatedly observed plate strings do not receive extra weight. The paper also describes variance of the Laplacian as a blur measure; the crop review now records and displays this raw value as an optional focus hint. No pass/fail cutoff is set because a useful threshold depends on camera, scale, and capture conditions. Both OCR views remain exploratory, are not protected holdout measurements, and can still reflect selection bias. The linked image archive was unavailable locally when checked, so its samples are not part of this implementation.

## Video integration candidates

- [Open Skoltech Car (OS:Car)](https://newpavlov.github.io/oscar-website/) documents CC BY 4.0 data, frame files named with Unix timestamps in microseconds, and assembled WebM sequences. The site says only a small portion is uploaded. This moving-car winter-road dataset may support a parser/pipeline smoke fixture after inspection of an actually available sequence and verification of frame-to-UTC alignment; it does not satisfy fixed-camera acceptance.
- [Zenseact Open Dataset (ZOD)](https://zod.zenseact.com/) offers UTC GNSS/IMU with sampled dashcam images under CC BY-SA 4.0, but access requires approval and frame-to-time alignment must be verified.
- [AI City 2019](https://www.aicitychallenge.org/2019-data-sets/) better matches fixed traffic cameras and supplies relative synchronization offsets, but requires a participation agreement and does not establish absolute UTC.
- The [Zenodo traffic-evaluation videos](https://zenodo.org/records/14041920) have no declared license or timezone on the record and are therefore not cleared for use.


### Additional source review (2026-10-26)

- The [University of Bath RoundaboutHD record](https://researchdata.bath.ac.uk/1574/) describes 40 minutes of labeled 4K/15 fps footage from four fixed-position traffic cameras, with vehicle tracking annotations. Its data archive metadata says “Software: MIT License,” but does not separately state a data license; the record also does not document an absolute UTC origin or a video-PTS-to-UTC mapping. It is a strong technical candidate for later review, but neither reuse rights nor the required UTC alignment are established.
- The [AI City 2019 dataset page](https://www.aicitychallenge.org/2019-data-sets/) describes synchronized fixed-camera videos and per-video start offsets, but does not establish absolute UTC capture time; the dataset is gated by a challenge participation agreement.
- The [Zenodo Traffic Evaluation v2 record](https://zenodo.org/records/14041920) includes long monitored-road videos, but its license field and timezone/UTC origin are blank.

These sources do not clear the acceptance gate. Do not download, redistribute, or use them as accuracy evidence until the rights and time provenance are confirmed.

### Additional clip search (2026-10-27)

- The [2K Dashcam Traffic Surveillance Video Corpus](https://huggingface.co/datasets/thundarstrom/traffic-surveillance-video-corpus) page labels the dataset CC BY 4.0 and describes 50-minute drives, but currently reports that the dataset is empty. It is dashcam footage and does not document a UTC-to-video-PTS mapping. No files were available or downloaded.
- The [TIme Square Intersection dataset](https://personal.ie.cuhk.edu.hk/~ccloy/downloads_qmul_TISI_dataset.html) provides image sequences grouped by GMT-named folders at approximately 10 Hz, rather than video clips. Its page restricts use to research purposes. It is not suitable as a product acceptance fixture.
- The official [AAU MTID page](https://vap.aau.dk/mtid/) links to a multi-view traffic dataset hosted through Kaggle, but the public page does not state a reuse license or timezone-aware capture/PTS mapping. It is not cleared for use.

This search found no candidate that satisfies the authorized representative-video gate with verified UTC provenance. No media was downloaded.

No candidate has been acquired or accepted. Do not include media in the repository until its exact terms, provenance, and UTC alignment are confirmed.

## Selection and evidence

Compare a simple baseline with compatible pretrained vehicle/small-object detectors, a light/background branch, tracking, and plate-chip crop proposals. For plates, record crop precision/recall and human approval/rejection rates on operator-provided, country-specific box labels. Measure OCR suggestions against independently human-confirmed plate text using whole-plate exact match, normalized character edit similarity, and corpus character error rate (Levenshtein edits divided by reference characters), separately from crop precision/recall. Preserve protected evaluation splits by related plate strings before claiming generalization. No target-country accuracy is currently measured, and plate identity lookup is out of scope. Record versions/commit/weight SHA/taxonomy/input needs/CPU and GPU memory. Choose based on technical fit and measured evidence. A conversation model interprets queries; it does not produce ground-truth pixel detections.

Frozen dataset manifests: source URL, retrieval date, license decision, file checksums, split IDs, camera/view/time groups, label schema and transformation versions. Keep train/validation/test disjoint by related captures and recordings. Background warm-up rules must be declared; no hidden use of future test frames or labels.

Report vehicle Precision/Recall/F1/AP, per-image count error, candidate false alarms/acceptance, tracking identity consistency/duplicate events, plate-chip crop precision/recall and human approval/rejection, throughput/latency/peak RAM/VRAM/storage. Break down by view/country/condition/profile. Do not use algorithm scores as measured accuracy. Acceptance thresholds await operator agreement.
