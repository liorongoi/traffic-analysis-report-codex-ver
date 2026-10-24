# 06 — Model research and evaluation plan

## Primary research starting points (not approved packs)

| Candidate | Scope | Required gate |
|---|---|---|
| BDD100K https://github.com/bdd100k/bdd100k | Diverse driving detection/tracking, night | Data license differs from BSD code; commercial use needs appropriate rights; moving-camera domain mismatch |
| UA-DETRAC https://faculty.ucmerced.edu/mhyang/papers/cviu2020_detrac.pdf | Fixed-view traffic, night/rain/occlusion, broad classes | Verify current data access and terms; broad truck/other labels do not validate semitrailers |
| AI City https://www.aicitychallenge.org/ai-city-challenge-dataset-access/ | Vehicle tracking and language retrieval | Select an actually available track and check its specific terms |
| IR-LPR https://github.com/mut-deep/IR-LPR | Iranian plate images/characters | Verify underlying data/weight rights separately from repository code |
| ExDark https://github.com/cs-chan/Exclusively-Dark-Image-Dataset | Supplementary low-light images | Current terms require written commercial permission; third-party photo rights separate |

The current review increment uses a local COCO-pretrained YOLOX-S ONNX pack from the official OpenCV Zoo distribution; see [vehicle detection evidence](10-vehicle-detection.md) for version, hash, license notes, and functional checks. This pack has not been accepted as an accuracy benchmark or production model. The candidates below remain research starting points, not approved datasets or dependencies. Iran data does not validate Lebanon/Israel/Latvia OCR. Company-provided authorized camera samples and country-specific labels are essential. Do not invoke hosted Roboflow or any online production API.

## Selection and evidence

Compare a simple baseline with compatible pretrained vehicle/small-object detectors, a light/background branch, tracking, and country-specific OCR. Record versions/commit/weight SHA/license/taxonomy/input needs/CPU and GPU memory. Choose only after applicable rights and measured evidence. A conversation model interprets queries; it does not produce ground-truth pixel detections.

Frozen dataset manifests: source URL, retrieval date, license decision, file checksums, split IDs, camera/view/time groups, label schema and transformation versions. Keep train/validation/test disjoint by related captures and recordings. Background warm-up rules must be declared; no hidden use of future test frames or labels.

Report Precision/Recall/F1/AP, per-image count error, candidate false alarms/acceptance, tracking identity consistency/duplicate events, plate exact match/CER, throughput/latency/peak RAM/VRAM/storage. Break down by view/country/condition/profile. Do not use algorithm scores as measured accuracy. Acceptance thresholds await operator agreement.
