import json
import tempfile
import unittest
from pathlib import Path
from PIL import Image

from road_traffic.object_search import execute, interpret
from road_traffic.store import Library
from road_traffic.inference_store import create_run, search_scope_snapshot


class SearchLibrary:
    def __init__(self, rows):
        self.rows = rows

    def images(self, mission, run=None):
        return self.rows


class ObjectSearchTests(unittest.TestCase):
    def test_interpretation_is_typed_and_marks_visual_attributes_unknown(self):
        spec = interpret('Find red pickup trucks', mission_id='mission', run_id='run-1')
        self.assertEqual(spec['categories'], ['truck'])
        self.assertEqual(spec['required_attributes'], [
            {'attribute': 'color', 'value': 'red'}, {'attribute': 'subtype', 'value': 'pickup'}])
        self.assertTrue(spec['operator_confirmation_required'])
        self.assertIn('color', spec['interpretation']['unsupported_or_unobservable'])
        self.assertEqual(spec['scope']['run'], 'run-1')

    def test_interpretation_binds_the_exact_saved_run_schedule_views_and_regions(self):
        with tempfile.TemporaryDirectory() as workspace, tempfile.TemporaryDirectory() as source:
            library = Library(workspace)
            mission = library.create('Search scope', 'Israel', 'Asia/Jerusalem')
            Image.new('RGB', (24, 24), (60, 90, 120)).save(Path(source) / '17052026-1604-scope.jpg')
            library.import_folder(mission, source, 'camera')
            image = library.images(mission)[0]['id']
            view = library.create_view(mission, 'North gate', 'Operator')
            library.assign_view(mission, image, view['id'], 'Operator')
            region = library.add_view_region(mission, view['id'], 'include',
                [[0.05, 0.05], [0.95, 0.05], [0.95, 0.95], [0.05, 0.95]], 'Operator')
            library.add_scan_window(mission, {'weekdays': [6], 'start_date': '2026-05-17',
                'end_date': '2026-05-17', 'start_time': '17:00', 'end_time': '20:00'})
            run = create_run(library, mission, {'threshold': 0.18, 'tiled': True})
            snapshot = search_scope_snapshot(library, mission, run)
            spec = interpret('Find cars', mission_id=mission, run_id=run, scope_snapshot=snapshot)
            self.assertEqual(spec['scope_snapshot']['schedule']['timezone'], 'Asia/Jerusalem')
            self.assertEqual(spec['scope_snapshot']['schedule']['windows'][0]['weekdays'], [6])
            self.assertEqual(spec['scope_snapshot']['view_assignments'][0]['view_name'], 'North gate')
            self.assertEqual(spec['scope_snapshot']['view_assignments'][0]['image_count'], 1)
            self.assertEqual(spec['scope_snapshot']['regions_by_view'][view['id']][0]['id'], region['region_key'])
            self.assertIn('spec_hash', spec)

    def test_changed_run_scope_requires_reinterpretation(self):
        library = SearchLibrary([])
        scope = {'run_id': 'run-1', 'schedule': {'timezone': 'Asia/Jerusalem'},
            'view_assignments': [], 'regions_by_view': {}}
        spec = interpret('Find cars', mission_id='mission', run_id='run-1', scope_snapshot=scope)
        changed = {**scope, 'schedule': {'timezone': 'Europe/Riga'}}
        with self.assertRaisesRegex(ValueError, 'changed'):
            execute(library, 'mission', spec, confirmed=True, scope_snapshot=changed)

    def test_interpretation_does_not_claim_a_run_scope_when_no_run_exists(self):
        spec = interpret('Find cars', mission_id='mission')
        self.assertEqual(spec['scope']['run'], 'no saved inference run exists')
        self.assertIsNone(spec['scope_snapshot'])

    def test_unsupported_terms_and_missing_categories_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Unsupported search terms'):
            interpret('Find a red helicopter', mission_id='mission')
        with self.assertRaisesRegex(ValueError, 'Name a supported vehicle category'):
            interpret('Find red objects', mission_id='mission')

    def test_hebrew_query_uses_same_typed_vehicle_and_color_vocabulary(self):
        spec = interpret('מצא טנדרים אדומים', mission_id='mission')
        self.assertEqual(spec['categories'], ['truck'])
        self.assertEqual(spec['required_attributes'], [
            {'attribute': 'color', 'value': 'red'}, {'attribute': 'subtype', 'value': 'pickup'}])

    def test_execution_requires_confirmation_and_unchanged_spec(self):
        spec = interpret('Find trucks', mission_id='mission')
        library = SearchLibrary([])
        with self.assertRaisesRegex(ValueError, 'confirm'):
            execute(library, 'mission', spec, confirmed=False)
        tampered = dict(spec, categories=['car'])
        with self.assertRaisesRegex(ValueError, 'changed'):
            execute(library, 'mission', tampered, confirmed=True)
        with self.assertRaisesRegex(ValueError, 'changed'):
            execute(library, 'another-mission', spec, confirmed=True)

    def test_search_matches_saved_category_without_guessing_color_or_subtype(self):
        row = {'id': 'image-1', 'name': '01.jpg', 'capture_utc': '2026-01-01T00:00:00Z',
               'view_id': None, 'view_name': None, 'revision': 0, 'review_prediction_run': None,
               'prediction_run': 'run-1', 'predictions': [
                   {'kind': 'candidate', 'label': 'truck', 'box': [.1, .1, .2, .2]},
                   {'kind': 'candidate', 'label': 'car', 'box': [.3, .1, .2, .2]},
                   {'kind': 'light', 'label': 'streetlight', 'box': [.5, .1, .2, .2]}], 'annotations': '[]'}
        result = execute(SearchLibrary([row]), 'mission', interpret('Find red pickup trucks', mission_id='mission'), confirmed=True)
        self.assertEqual(result['matched_images'], 1)
        self.assertEqual(result['matched_objects'], 1)
        self.assertEqual(result['uncertain_objects'], 1)
        self.assertEqual(result['results'][0]['objects'][0]['uncertain_attributes'], ['color', 'subtype'])
        self.assertIn('not proof', result['evidence_note'])

    def test_exact_run_review_overrides_unrelated_predictions(self):
        row = {'id': 'image-1', 'name': '01.jpg', 'capture_utc': None, 'view_id': None, 'view_name': None,
               'revision': 1, 'review_prediction_run': 'run-1', 'prediction_run': 'run-1',
               'predictions': [{'kind': 'candidate', 'label': 'car'}],
               'annotations': json.dumps([{'kind': 'vehicle', 'label': 'truck', 'box': [.1, .1, .2, .2]}])}
        result = execute(SearchLibrary([row]), 'mission', interpret('Find trucks', mission_id='mission'), confirmed=True, run='run-1')
        self.assertEqual(result['results'][0]['objects'][0]['source'], 'human-reviewed')
        self.assertEqual(result['matched_objects'], 1)


if __name__ == '__main__':
    unittest.main()
