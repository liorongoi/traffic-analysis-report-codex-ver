import json
import unittest

from road_traffic.object_search import execute, interpret


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
