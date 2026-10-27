"""Real model-pack smoke checks; these do not claim country accuracy."""
import unittest
from pathlib import Path

from PIL import Image

from road_traffic.plate_review import (
    GlobalPlateTextRecognizer, OCR_GLOBAL_MODEL_SHA256,
    _global_latin_ocr_applicable,
)


ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / 'model-packs' / 'license-plate-yolox'


class GlobalPlateOcrTests(unittest.TestCase):
    def test_latin_plate_recognizer_loads_pinned_onnx_and_returns_reviewable_suggestion(self):
        recognizer = GlobalPlateTextRecognizer(PACK)
        result = recognizer.read(Image.new('RGB', (180, 54), (240, 240, 240)))
        self.assertEqual(result['model_sha256'], OCR_GLOBAL_MODEL_SHA256)
        self.assertIsInstance(result['text'], str)
        self.assertTrue(0 <= result['confidence'] <= 1)
        self.assertTrue(set(result['text']) <= set(recognizer.alphabet))

    def test_latin_only_suggestion_is_not_used_for_iranian_missions(self):
        for country in ('Iran', 'IR', 'IRN', 'ایران', 'إيران'):
            with self.subTest(country=country):
                self.assertFalse(_global_latin_ocr_applicable(country))
        for country in ('Lebanon', 'Israel', 'Latvia', ''):
            with self.subTest(country=country):
                self.assertTrue(_global_latin_ocr_applicable(country))


if __name__ == '__main__':
    unittest.main()
