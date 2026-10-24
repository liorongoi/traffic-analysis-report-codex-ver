# Real vehicle detection and review — initial milestone

Current night separation and correction-head capabilities are documented in [the subsequent update](11-night-separation-and-learning.md). Training limitations below describe the initial milestone.

## What is running

A real COCO-pretrained YOLOX-S ONNX model from the official OpenCV Zoo distribution, served by a separate CPU worker. The directory declares Apache-2.0 licensing; the staged pack includes its LICENSE and upstream README. No model or package downloads occur in the inference adapter. The model SHA-256 is `c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063`.

Vehicle labels: car, truck, bus, motorcycle, bicycle. They are **unconfirmed model proposals**, not human ground truth. Detector scores are not calibrated accuracy. Threshold 0.18 is a review-oriented engineering setting, not an accuracy guarantee. The full frame and overlapping native-resolution crops feed the same model, followed by cross-class vehicle NMS; separate overlapping passes are not independent votes.

## Night and monochrome handling

Routing uses mean luminance below 95 **or** mean saturation below 5. This includes the bright monochrome night mode of this camera; it is not an astronomical day/night classifier. A separate experimental pass proposes novel bright components and possible pairs. Two lights in a pairing hypothesis are not two confirmed vehicles, and irregular still images are not tracked as the same vehicle across time.

For the explicit fixed Fintraffic view, up to 32 compatible, non-held-out reference images provide a saved persistence/median/MAD background. The target image is excluded when it is a reference. Similarity, source IDs/hashes, model hash and parameters are saved. Candidate novelty must exceed `max(25, 3*MAD)` above the reference median. Large images outside the bounded background budget do not learn this background. Other camera presets do not silently pool different views.

**Known limitations:** the pretrained detector misses unresolved and streaking vehicles at night. Reflections and illumination changes still create many light proposals. Background suppression may hide stationary vehicle lights; the unmodified RGB image remains the detector input, and users must inspect missed objects. This is an initial review tool, not a validated night traffic counter. No precision, recall or F1 can be asserted without independently reviewed references.

## Review workflow

1. Open the local app and select **Fintraffic sample review**.
2. The latest saved run is already populated. Use **Show images** for all, night/monochrome, detector-box or light-proposal views.
3. Use **Proposals** to inspect vehicle boxes separately from experimental light candidates.
4. In **Editable review**, change a proposal's type to **Confirmed vehicle**, correct its class, reject a false positive, or draw a missed rectangle. Enter the operator and select review completeness/evidence role, then **Save review revision**.
5. Switch **Overlay** to **Original predictions (read only)** to inspect the unchanged output. Saved reviews remain tied to their original inference run, including after a new run.
6. Use opacity/zoom for inspection; these do not change source pixels or saved scientific coordinates.
7. **Inference report** downloads configuration, original outputs, review lineage and source hashes as JSON. **Export CSV** exports the selected run with separate confirmed counts and original predictions. Rejected annotations remain in audit data and are excluded from counts.

Runs expose durable stage, done/total, errors and a progress bar. **Stop** cancels after the current image; cancellation and failed runs retain completed predictions. Closing the browser does not stop its worker. Automatic retraining is not implemented; no corrections are silently used for training.

## Executed evidence

- **143/143 real samples processed, zero per-image errors**, adapter `yolox-s-coco+night-lights-v3`.
- 90 frames routed to night/monochrome processing. 872 detector proposals and 4,874 light proposals. These counts do not establish object counts or accuracy.
- Median scoring time about **0.198 s/image**, p95 **0.620 s/image**, total scoring **41.18 s** on this Mac. Model loading and background preparation are additional. Windows and GPU performance are unverified.
- **35 unit/API regression tests passed**, including single/pair light logic, persistence suppression, immutable outputs, correction conflicts, holdouts, cross-mission isolation, selected reports and rejected-object export counts.
- Real-model functional check with Python outbound connections blocked: actual day/night samples processed; daytime sample produced 8 proposals; blank image produced 0. This is not an OS-level air-gap test.
- Real worker processed 2 actual samples using `python -I` and the private-runtime bootstrap.
- Browser test in a separate temporary QA workspace: real inference started/completed; progress appeared; stop reached cancelled; class/type edit, rejection and missed-object rectangle survived save/reload; original 8 predictions stayed unchanged/read-only. English/Hebrew direction switching passed. QA corrections have role `none` and are not training truth.
- Visual contact-sheet inspection confirmed useful daytime boxes and visible night misses. This qualitative inspection is not a labelled benchmark.

Final live browser checks also showed 143/143 processed, 8 saved daytime vehicle proposals, and 90 night/monochrome rows; a selected night frame exposed 21 light proposals. Screenshot evidence is saved in reports/. The browser download-event check timed out; the report endpoint was verified by API tests, but UI file download is not counted as passed.

Raw evidence: `reports/test-results-inference.txt`, `reports/real-model-functional-check.json`, `reports/isolated-worker-check.json`, `reports/ui-review-verification.json`, and `reports/fintraffic-real-inference.json`.

## Offline Windows status

Staging now requires the local model/license and NumPy, Pillow, OpenCV, ONNX Runtime and timezone packages in a complete private Windows runtime. Installer verification invokes a separate CPU-native model check, not a package manager. The isolated worker bootstrap avoids dependence on PYTHONPATH in embedded Python.

No Windows package has been built or installed in this increment. No-admin installation, Windows native libraries, network shares, GPU execution and fully disconnected target operation still require target validation. The Mac development environment and its dependency lock are not Windows installer payloads.
