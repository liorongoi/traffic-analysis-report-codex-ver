# Night separation and correction learning — 24 October 2026

## Implemented

- Light proposals are separated into **headlight candidates**, **recurring/streetlight suspects**, **reflection/trail suspects**, and **uncertain lights**. These are explainable hypotheses, not independently measured classifications.
- Compact novel sources can be proposed singly or as a mutual-nearest pair. A pair is not two vehicles. A bright bridge discourages pairing streak fragments. Glare near a recurring source remains uncertain.
- Recurring lights remain available for review instead of being deleted. Recurrence may also describe stationary vehicle lights; it does not prove a streetlight. Elongation may describe either a reflection or an exposure trail.
- Original predictions, background references and earlier runs are preserved. New filters/colors prioritize review without changing source imagery.
- Reviewers can assign an explicit night class (headlight, streetlight, reflection, other), confidence 1–5, and evidence role. An unmodified model suggestion is not a label.
- A separate regularized multinomial light-patch classifier supports refitting from explicit training labels. **YOLOX is unchanged and is not fine-tuned.** The candidate is saved with its feature schema, fitting evidence, source hashes, parameters and hash; it requires explicit selection for a later inference run.
- Only latest, explicitly reviewed class decisions with confidence >=3 and role `training` enter fitting. Overlapping patches and copied-image evidence are deduplicated conservatively. Held-out hashes, same camera/hour groups and frames within 30 minutes of a held-out image are excluded. Validation is scored separately; test labels are not scored or fitted by the training action.
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
5. Select the saved candidate under **Correction model for next run**, then start a new inference run. Suggestions from that head remain uncalibrated and do not automatically become confirmed vehicles.

## Executed tests

At this checkpoint, **50 tests passed**, including synthetic single/pair headlights, recurring lamps not paired as headlights, nearby fixture glare, elongated streaks, missing-background uncertainty, no-label training refusal, explicit refit/versioning, validation isolation, checkpoint tampering, review-only exclusion and real-worker application of a trained synthetic correction head. The current full suite has since grown to 55 passing Python checks after adding mission-configuration coverage. Synthetic labels were confined to test workspaces and are not evidence of Fintraffic accuracy.

Browser verification in an isolated QA workspace confirmed that explicit night class and confidence survive save/reload and that review-only labels do not unlock training. The live main app showed 90 night/monochrome frames, separate recurring-light/headlight filters, and zero eligible training regions. Existing correction preservation and original read-only prediction tests also remain passing.

Evidence: `reports/night-v5-tests.txt`, `reports/night-v5-real-run.json`, `reports/inference-v5-worker.log`. Windows/GPU execution and installation remain unverified.
