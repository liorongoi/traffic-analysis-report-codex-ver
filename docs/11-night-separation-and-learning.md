# Night separation and correction learning — 25 October 2026

## Implemented

- Light proposals are separated into **headlight candidates**, **recurring/streetlight suspects**, **reflection/trail suspects**, and **uncertain lights**. These are explainable hypotheses, not independently measured classifications.
- Compact novel sources can be proposed singly or as a mutual-nearest pair. A pair is not two vehicles. A bright bridge discourages pairing streak fragments. Glare near a recurring source remains uncertain.
- Recurring lights remain available for review instead of being deleted. Recurrence may also describe stationary vehicle lights; it does not prove a streetlight. Elongation may describe either a reflection or an exposure trail.
- Original predictions, background references and earlier runs are preserved. New filters/colors prioritize review without changing source imagery.
- Reviewers can assign an explicit night class (headlight, streetlight, reflection, other), confidence 1–5, and evidence role. An unmodified model suggestion is not a label.
- A separate regularized multinomial light-patch classifier supports refitting from explicit training labels. **YOLOX is unchanged and is not fine-tuned.** The candidate is saved with its feature schema, fitting evidence, source hashes, parameters and hash. It can be selected for a run; after a current held-out evaluation, a recorded operator promotion makes it the default selected model for future runs in that mission. Base context rules remain selectable.
- Only latest, explicitly reviewed class decisions with confidence >=3 and role `training` enter fitting. Overlapping patches and copied-image evidence are deduplicated conservatively. Held-out hashes, same camera/hour groups and frames within 30 minutes of a held-out image are excluded. Validation is scored separately. Test labels are never fitted or used for validation; a separate explicit evaluation action scores their light patches once for an unchanged source/region manifest and stores an immutable report. A separate audited action then records promotion or rollback with candidate hash, evaluation, operator and rationale.
- The technical manual-fit minimum is four regions, two classes and two distinct image hashes. This is **not** an automatic training trigger or evidence of adequate model quality. Automatic correction-triggered training remains unimplemented in this independent traffic app.
- A correction head requires a verified single Fintraffic camera view; unknown/mixed views are rejected rather than inferred from opaque filenames. Other camera-view declarations remain future work.

## Real-sample execution

Latest run: `0d2714e1-6c02-4d62-a417-5d26fd513073`, adapter `yolox-s-coco+night-lights-v5`.

143/143 real Fintraffic images completed with no per-image processing errors. Ninety frames used the night/monochrome branch. The run produced 993 headlight hypotheses, 3,840 recurring/streetlight hypotheses, 1,861 reflection/trail hypotheses and 5,450 uncertain-light proposals. These totals describe review workload, **not vehicle counts or accuracy**. More recurring proposals are retained than in earlier versions, so comparing raw totals to previously suppressed output does not measure improvement.

Visual inspection found false headlight hypotheses near fixtures and on streaks. Those findings prompted the extra uncertainty and pairing checks. False positives and misses remain. Without independent real-image labels, Precision/Recall/F1 and improvement in real-world accuracy are unmeasured.

## Saved corrections: actual state

The main Fintraffic mission currently contains **zero saved reviews**. Its learning inventory is empty. Calling Train returns `needs_labels` and creates no checkpoint. No user-trained light model or fine-tuned detector has been produced. Earlier QA edits remain in an isolated temporary workspace with evidence role `none`; they were not used as real training truth.

A request for the location of any other correction workspace/export is pending. No SpectraScope databases, correction rules, imagery or models are imported into this application.

## How to supply useful labels

1. Select **Night / monochrome pass** and a proposal category.
2. Review the source image. Assign **Explicit night class** and **Review confidence** for the relevant object. Rejecting an ambiguous proposal alone does not assign a semantic night class.
3. Select **Training reference** and save. Keep separate images for **Validation holdout**; avoid nearby frames for training.
4. Choose **Train light correction candidate** once eligible examples exist. Inspect its report. With no validation labels, accuracy remains unmeasured.
5. Evaluate a completed candidate on the held-out labels, inspect the report, then use **Candidate promotion and rollback decisions** to record an operator decision and rationale. A promoted candidate is selected by default for later runs; choose **None · context rules only** to use the base path instead. Suggestions from the head remain uncalibrated and do not automatically become confirmed vehicles.

The learning panel now shows training/validation/test/exclusion counts, the selected candidate's artifact hash and per-class validation metrics, recent candidate status/failure history, and append-only promotion/rollback history. Test labels are counted but not shown to the fitter or used for validation. The selector defaults to the latest promoted candidate when one exists; selecting **None** returns to the base context rules. Each run records the selected artifact hash.

## Executed tests

At the night-v5 checkpoint, 50 focused tests passed, and the full suite later grew to 55 checks at that earlier point. On 2026-10-25, the current full suite passed 196 tests, including promotion evaluation gating, rollback to base rules, immutable decision history and HTTP delivery. Synthetic labels were confined to test workspaces and are not evidence of Fintraffic accuracy.

Browser verification in an isolated QA workspace confirmed that explicit night class and confidence survive save/reload and that review-only labels do not unlock training. The live main app showed 90 night/monochrome frames, separate recurring-light/headlight filters, and zero eligible training regions. Existing correction preservation and original read-only prediction tests also remain passing.

Evidence: `reports/night-v5-tests.txt`, `reports/night-v5-real-run.json`, `reports/inference-v5-worker.log`. Windows/GPU execution and installation remain unverified.
