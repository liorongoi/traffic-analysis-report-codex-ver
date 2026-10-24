# Requirements fit review — 2026-10-24

This review compares the current project state with the verbatim product request in [requirements-source.txt](requirements-source.txt). It describes the implementation in this workspace, not the full requested production system. Evidence from [vehicle detection](10-vehicle-detection.md) supersedes earlier foundation-only statements where they conflict. The original product requirements remain controlling; a prototype or passing test does not waive them.

## Fit summary

The repository is a valid independent, still-image review and inference increment. It is not yet a production-ready match for the full request. It has a real local CPU vehicle-proposal adapter and an experimental light-candidate path, plus source snapshots, per-mission stores, revisioned review, and CSV/report output. Results are explicitly proposals and are not validated counts.

| Requirement area | Current fit | Evidence and remaining work |
|---|---|---|
| Independent application; offline-first design | Partial | Separate application and local staged inference are documented. Native Windows disconnected operation, complete offline packaging, and authenticated LAN deployment are unverified. |
| Requirements, architecture, workflows, roadmap | Partial | Documents 01–10 and the source prompt exist. Traceability status must reflect newer implementation evidence; unresolved product questions remain in 02. |
| Missions and per-mission persistence | Partial | Per-mission settings now save input type, analysis mode, location/time zone, source/output directories, recursion, one-time/continuous preference, and CSV/XLSX preference. Configured directories are checked; imports honor recursion; video and unsupported inference modes are blocked. The offline location map, durable monitoring, destination-folder export, XLSX, full configuration snapshots, and recovery policy remain incomplete. |
| UTC capture time and local-time handling | Partial | Confirmed image filename parsing, minute precision, and historical time-zone conversion are covered. Offline geographic time-zone map and full local scanning schedules are not implemented. Video absolute UTC origin remains unresolved. |
| Import and monitoring | Partial | Recursive image import, manifest verification, and independent snapshots exist. Durable directory monitoring, stability/reconciliation, restart recovery, notifications, changed-file history, and retention policy remain incomplete. |
| Camera views and regions | Not implemented | Current sample background handling is tied to one explicit fixed view; general grouping, reassignment history, split/merge, editable inclusion/exclusion polygons, and invalidation review are outstanding. |
| Still-image detection and review | Partial | Real CPU YOLOX-S proposals, experimental light proposals, immutable predictions, correction revisions, navigation, and confirmed-count separation exist. No labeled accuracy benchmark or validated traffic counter exists. General schedules, typed chat search, and review workflows for all requested attributes remain outstanding. |
| Nighttime analysis | Experimental | Light components and background suppression are proposals with documented failure modes. No confirmed vehicle inference from lights, validated night accuracy, or headlight-transition evaluation exists. |
| Video, tracking, events, clips | Not implemented | No video decoding, PTS handling, tracking, event review, clip export, or long-recording validation. |
| License plates | Not implemented | No country-specific plate detector/OCR or country-specific evaluation for Iran, Lebanon, Israel, and Latvia. |
| Corrections and learning | Partial | Append-only corrections, provenance, holdout protections, and a separate night-light refit path exist. YOLOX training, complete dataset selection/splitting, candidate evaluation, promotion, rollback, and video corrections are outstanding. Bulk approval is not training ground truth. |
| Exports | Partial | CSV and JSON inference reports exist. XLSX, video-event/clip exports, robust large-export handling, and complete snapshot/export audit workflow remain outstanding. |
| Hardware adaptation and resource control | Partial | A conservative RAM-budget helper and bounded inference/background work are documented. Target 16/48 GB measurements, hardware/backend probing, supported GPU execution, and verified CPU fallback on Windows remain untested. |
| UX, languages, security | Partial | The mission setup form now records the required configuration and labels unavailable modes. English/Hebrew direction switching and still-image review have evidence. Monitoring controls, contextual help, authorized LAN authentication, and full operator acceptance remain outstanding. |
| Installer and updates | Not implemented for production | Windows installer source exists. No finished setup EXE, native current-user install, clean offline install, update/rollback preservation, or target workstation acceptance has been demonstrated. |
| Acceptance and benchmark evidence | Partial | Foundation/API and real-model functional checks are documented. No independently labeled precision/recall/F1, count error, OCR, tracking, target hardware performance, or numerical product thresholds are available. |

## Alignment issues to keep visible

- The README and `docs/10-vehicle-detection.md` describe the current real inference increment. Older planning text in documents 01, 05, 06, 08, and 09 must not claim that no adapter exists or that the detector is already accepted as a production capability.
- The 143-image sample run proves that the adapter processed those files. It does not prove detection accuracy, night performance, traffic counts, Windows compatibility, GPU support, or a complete offline deployment.
- The full request explicitly makes several inputs unresolved: representative video/codecs and duration, daily volume/history size, installation permissions, clip context, training/reuse policy, and numerical acceptance thresholds. Implementation must preserve these as open decisions rather than silently inventing values.
- The next product gates are: finish mission configuration/monitoring/views/schedules; validate detection on independently labeled, authorized camera data; implement video and country-specific plate paths; complete correction/training lifecycle; then validate the packaged offline Windows/LAN deployment on target hardware.

## Review method and limits

This is a documentation/source-state review using repository instructions, the README, requirements and design documents, and current implementation evidence. It does not claim a fresh test run or independent benchmark. Existing test and functional evidence is listed in `docs/10-vehicle-detection.md` and `docs/09-verification.md`; consult those records for their exact scope and limitations.
