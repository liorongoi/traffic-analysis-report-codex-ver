import contextlib
import sys
import tempfile
from pathlib import Path
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from road_traffic import road_segmentation


class RoadSegmentationTests(unittest.TestCase):
    def test_local_loader_never_requests_remote_model_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'config.json').write_text('{}', encoding='utf-8')
            (root / 'model.safetensors').write_bytes(b'fixture')
            (root / 'runtime').mkdir()
            calls = []

            class Model:
                def to(self, device):
                    self.device = device
                    return self

                def eval(self):
                    self.evaluated = True
                    return self

            model = Model()

            class ModelFactory:
                @staticmethod
                def from_pretrained(path, **kwargs):
                    calls.append(('model', path, kwargs))
                    return model

            class ProcessorFactory:
                @staticmethod
                def from_pretrained(path, **kwargs):
                    calls.append(('processor', path, kwargs))
                    return object()

            torch = SimpleNamespace(float32='float32', inference_mode=contextlib.nullcontext)
            transformers = SimpleNamespace(Sam3Model=ModelFactory, Sam3Processor=ProcessorFactory)
            road_segmentation._load_sam.cache_clear()
            try:
                with (patch.object(sys, 'path', list(sys.path)) as module_path,
                        patch.dict(sys.modules, {'torch': torch, 'transformers': transformers}),
                        patch.dict('os.environ', {'ROAD_TRAFFIC_SAM3_DIR': str(root)})):
                    loaded = road_segmentation._load_sam(str(root))
            finally:
                road_segmentation._load_sam.cache_clear()
            self.assertIs(loaded[1], model)
            self.assertTrue(model.evaluated)
            self.assertEqual([item[0] for item in calls], ['model', 'processor'])
            self.assertTrue(all(item[2]['local_files_only'] for item in calls))

    def test_windows_model_pack_uses_user_data_outside_the_installer(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            default = root / 'user-data' / 'RoadTrafficAnalysis' / 'Models' / 'sam3.1-unsloth'
            default.mkdir(parents=True)
            override = root / 'external-drive' / 'sam-pack'
            with patch.dict('os.environ', {'LOCALAPPDATA': str(root / 'user-data'),
                    'ROAD_TRAFFIC_SAM3_DIR': ''}):
                self.assertEqual(road_segmentation._model_dir(root / 'installed-app'), default)
            with patch.dict('os.environ', {'LOCALAPPDATA': str(root / 'user-data'),
                    'ROAD_TRAFFIC_SAM3_DIR': str(override)}):
                self.assertEqual(road_segmentation._model_dir(root / 'installed-app'), override)

    def test_text_prompt_masks_become_review_only_normalized_road_polygons(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[60:100, 20:80] = 1
        scores = np.array([.8], dtype=np.float32)

        class Inputs(dict):
            def __init__(self):
                super().__init__(original_sizes=SimpleNamespace(tolist=lambda: [[100, 100]]))

        class Processor:
            def __call__(self, **kwargs):
                self.prompt = kwargs['text']
                return Inputs()

            def post_process_instance_segmentation(self, outputs, **kwargs):
                self.arguments = kwargs
                return [{'masks': np.asarray([mask]), 'scores': scores}]

        processor = Processor()
        cv2 = ModuleType('cv2')
        cv2.RETR_EXTERNAL = 0
        cv2.CHAIN_APPROX_SIMPLE = 0

        def find_contours(segmentation, *_):
            ys, xs = np.nonzero(segmentation)
            contour = np.array([[[xs.min(), ys.min()]], [[xs.max(), ys.min()]],
                [[xs.max(), ys.max()]], [[xs.min(), ys.max()]]], dtype=np.int32)
            return [contour], None

        cv2.findContours = find_contours
        cv2.contourArea = lambda contour: float(len(contour))
        cv2.arcLength = lambda contour, closed: 100.0
        cv2.approxPolyDP = lambda contour, epsilon, closed: contour
        torch = SimpleNamespace(inference_mode=contextlib.nullcontext)
        with (patch.dict(sys.modules, {'cv2': cv2}),
                patch.object(road_segmentation, '_load_sam', return_value=(torch, lambda **inputs: object(), processor))):
            result = road_segmentation.propose_road_masks(np.zeros((100, 100, 3), dtype=np.uint8))

        self.assertEqual(processor.prompt, 'road surface')
        self.assertEqual(processor.arguments['threshold'], .25)
        self.assertEqual(processor.arguments['target_sizes'], [[100, 100]])
        self.assertEqual(len(result['candidates']), 1)
        candidate = result['candidates'][0]
        self.assertEqual(candidate['source'], 'sam3.1-road-surface-text-prompt')
        self.assertTrue(all(0 <= coordinate <= 1 for point in candidate['polygon'] for coordinate in point))
        self.assertTrue(result['review_required'])
        self.assertFalse(result['automatic_vehicle_detection_changed'])


if __name__ == '__main__':
    unittest.main()
