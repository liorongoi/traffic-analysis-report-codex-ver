# Plate localization and local OCR pack

This pack contains a YOLOX-tiny plate-region detector and a CPU ONNX CTC text recognizer. The app saves each detected plate chip, displays an editable OCR suggestion, and records chip approval separately from text acceptance, correction, rejection, or unreadable decisions. OCR text is never treated as verified without an operator action.

The recognizer is an ONNX conversion of PaddleOCR's official [`arabic_PP-OCRv3_mobile_rec` model](https://huggingface.co/PaddlePaddle/arabic_PP-OCRv3_mobile_rec), using the [documented ONNX conversion](https://huggingface.co/tobiichioriguchi/arabic_PP-OCRv3_mobile_rec_onnx) and the [official Arabic dictionary](https://github.com/PaddlePaddle/PaddleOCR/blob/main/ppocr/utils/dict/arabic_dict.txt). Its 161-character dictionary includes Latin, Arabic/Persian letters, ASCII digits, and Arabic-Indic digits. Model and dictionary hashes, source revisions, and sizes are pinned in `manifest.json`. Inference uses the installed CPU ONNX Runtime only; no network access or runtime substitution is allowed.

PaddleOCR reports 73.55% exact recognition accuracy on a general Arabic-text benchmark. That is not plate-text accuracy. Iranian Persian digit forms and OCR quality for Iran, Lebanon, Israel, and Latvia have not been independently evaluated. Inspect each chip and manually correct or reject suggestions. No vehicle identity lookup is performed.

The detector weights are Apache-2.0. Its publisher notice says some community-contributed training photos were used without consent obtained; source imagery is not redistributed. See `NOTICE` and the per-artifact manifest for provenance.
