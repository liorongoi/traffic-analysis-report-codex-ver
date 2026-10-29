import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from PIL import Image
from unittest.mock import patch

from road_traffic.model_registry import (import_candidate, list_models, resolve_model,
    select_model, decision_history)
from road_traffic.store import Library


def _pack(path, content, classes=None):
    path.mkdir(parents=True)
    (path / 'model.onnx').write_bytes(content)
    manifest = {'name': 'Test vehicle candidate', 'files': {'model.onnx': {
        'sha256': hashlib.sha256(content).hexdigest(), 'bytes': len(content)}}}
    if classes is not None:
        manifest['classes'] = classes
    (path / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')


class ModelRegistryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.library = Library(root / 'workspace')
        self.mission = self.library.create('Registry test', 'Lebanon', 'Asia/Beirut')
        self.base = root / 'base'
        _pack(self.base, b'base model')
        self.candidate = root / 'trained' / 'roadloom-model-pack'
        _pack(self.candidate, b'trained model', ['car', 'truck'])

    def tearDown(self):
        self.temp.cleanup()

    def test_import_select_and_list_mission_local_model(self):
        imported = import_candidate(self.library, self.candidate)
        self.assertFalse(imported['existing'])
        listing = list_models(self.library, self.base, self.mission)
        self.assertEqual(listing['selected_id'], 'base')
        self.assertEqual(listing['models'][1]['id'], imported['id'])
        selected = select_model(self.library, self.base, self.mission,
            imported['id'], 'Operator', 'Improved reviewed car examples')
        self.assertTrue(selected['changed'])
        self.assertEqual(resolve_model(self.library, self.base, imported['id']).name, imported['id'])
        self.assertEqual(list_models(self.library, self.base, self.mission)['selected_id'], imported['id'])
        self.assertEqual(decision_history(self.library, self.mission)[0]['reason'],
            'Improved reviewed car examples')

    def test_import_rejects_pack_without_trained_class_declaration(self):
        incomplete = Path(self.temp.name) / 'incomplete'
        _pack(incomplete, b'incomplete model')
        with self.assertRaisesRegex(ValueError, 'trained vehicle class list'):
            import_candidate(self.library, incomplete)

    def test_selection_requires_audited_operator_and_reason(self):
        imported = import_candidate(self.library, self.candidate)
        with self.assertRaisesRegex(ValueError, 'operator name'):
            select_model(self.library, self.base, self.mission, imported['id'], '', 'reason')

    def test_new_image_run_freezes_selected_candidate_id_and_hash(self):
        from road_traffic.inference import InferenceService
        from road_traffic.inference_store import runs
        imported = import_candidate(self.library, self.candidate)
        select_model(self.library, self.base, self.mission, imported['id'], 'Operator', 'Reviewed candidate')
        source = Path(self.temp.name) / 'source'
        source.mkdir()
        Image.new('RGB', (8, 8), (10, 20, 30)).save(source / '17052026-1604-test.jpg', 'JPEG')
        self.library.import_folder(self.mission, source, 'camera')
        service = InferenceService(self.library, self.base)
        profile = {'device': 'cpu', 'cpu_threads': 1, 'workers': 1,
            'requested': 'auto', 'selected': 'auto'}
        with (patch('road_traffic.hardware.resolve_resource_profile', return_value=profile),
                patch.object(service, '_spawn')):
            run = service.start(self.mission)['run']
        config = json.loads(next(item for item in runs(self.library, self.mission) if item['id'] == run)['config'])
        self.assertEqual(config['vehicle_model_id'], imported['id'])
        self.assertEqual(config['model_sha256'], imported['sha256'])


if __name__ == '__main__':
    unittest.main()
