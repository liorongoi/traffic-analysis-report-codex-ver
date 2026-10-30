# Plate-region localization and chip extraction

This pack contains one local YOLOX-tiny model that proposes plate-region boxes. RoadLoom saves each detected region as a crop beside source, timestamp, detector-confidence, and checksum lineage. The operator approves or rejects each crop. The application does not recognize plate letters or digits, search by plate text, infer vehicle identity, or make model output a confirmed plate without operator review.

The model and its checksum are pinned in `manifest.json`. Operation performs no runtime downloads. The bundled detector is a proposal aid; localization accuracy on the target cameras remains unvalidated. The supplied Iran and Lebanon charts are visual guidance, not real-camera test data.

The detector weights are Apache-2.0. Its publisher notice says some community-contributed training photos were used without consent obtained; source imagery is not redistributed. See `NOTICE` for provenance.
