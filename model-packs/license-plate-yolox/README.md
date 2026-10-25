# Plate-region detector pack

This pack contains only the YOLOX-tiny plate-region detector. The application uses its bounding boxes to save plate image crops for human review. It does not load, bundle, or call OCR, text recognition, normalization, or identity lookup.

The detector weights are Apache-2.0 and SHA-256 pinned in `manifest.json`. `LICENSE` and `NOTICE` are retained from the upstream model pack. The upstream notice says its training set includes community-contributed images for which consent was not obtained; the source images are not included here. Target-country performance for Iran, Lebanon, Israel, and Latvia has not been independently evaluated. Treat detections as unconfirmed proposals and approve or reject each crop.

Upstream: [open-lpr model pack](https://huggingface.co/faisalthaheem/open-lpr-models). Only `plate_yolox_tiny_640.onnx` is used. The OCR model and character dictionary are deliberately absent.
