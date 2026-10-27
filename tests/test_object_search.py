import json
import tempfile
import unittest
from pathlib import Path
from PIL import Image

from datetime import datetime, timezone
from road_traffic.object_search import execute, execute_video, interpret, video_scope_snapshot
from road_traffic.store import Library
from road_traffic.inference_store import create_run, search_scope_snapshot
from road_traffic import video_store


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

    def test_specification_example_semi_trailer_phrase_is_parsed_without_inventing_subtype_evidence(self):
        spec = interpret('Find every occurrence of a semi-trailer truck', mission_id='mission')
        self.assertEqual(spec['categories'], ['truck'])
        self.assertEqual(spec['required_attributes'], [{'attribute': 'subtype', 'value': 'semi_trailer'}])
        self.assertIn('fine vehicle subtype', spec['interpretation']['unsupported_or_unobservable'])
        self.assertTrue(spec['operator_confirmation_required'])
        other_subtype=interpret('Find other trucks',mission_id='mission')
        self.assertEqual(other_subtype['required_attributes'],[{'attribute':'subtype','value':'other'}])

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

    def test_video_search_is_run_bound_uses_utc_local_time_and_keeps_uncertainty(self):
        with tempfile.TemporaryDirectory() as workspace, tempfile.TemporaryDirectory() as source_root:
            source_path = Path(source_root) / 'traffic.mkv'
            source_path.write_bytes(b'video-search-fixture')
            library = Library(workspace)
            mission = library.create('Video search', 'Israel', 'Asia/Jerusalem', {
                'input_type': 'videos', 'analysis_mode': 'video_traffic',
                'source_root': source_root, 'output_root': '',
            })
            source = video_store.register_video(library, mission, 'traffic.mkv',
                recording_start_utc=datetime(2026, 5, 17, 14, 0, tzinfo=timezone.utc),
                origin_provenance='operator confirmed from recorder log')
            run = video_store.create_run(library, mission, source['id'], {
                'detector': {'version': 'fixture', 'sha256': 'fixture'},
                'tracker_policy': {'max_gap_seconds': [1, 2], 'minimum_iou': .2,
                    'max_center_distance': .15, 'association_margin': .05,
                    'compatible_label_groups': []},
            })
            with library.connect(mission) as conn:
                conn.execute("UPDATE video_runs SET status='completed',stage='completed' WHERE id=?", (run['id'],))
                for event_id, track, label, utc_text, offset, uncertain in (
                    ('merge-source', 'track-1', ['truck'], '2026-05-17T14:30:00+00:00', 1800, 1),
                    ('merge-target', 'track-4', ['bus'], '2026-05-17T14:35:00+00:00', 2100, 0),
                    ('rejected-truck', 'track-5', ['truck'], '2026-05-17T14:40:00+00:00', 2400, 0),
                    ('out-window', 'track-2', ['truck'], '2026-05-17T18:30:00+00:00', 16200, 0),
                    ('wrong-class', 'track-3', ['car'], '2026-05-17T14:45:00+00:00', 2700, 0),
                ):
                    conn.execute('''INSERT INTO video_events(id,mission_id,run_id,source_id,track_id,
                        first_pts,first_time_base_num,first_time_base_den,last_pts,last_time_base_num,last_time_base_den,
                        first_offset_num,first_offset_den,last_offset_num,last_offset_den,first_utc,last_utc,labels,
                        observation_count,association_uncertain,representative_box,uncertainty_reasons,created)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                        (event_id, mission, run['id'], source['id'], track, offset, 1, 1, offset, 1, 1,
                         offset, 1, offset, 1, utc_text, utc_text, json.dumps(label), 2, uncertain,
                         '[0.1,0.1,0.2,0.2]', json.dumps(['ambiguous association'] if uncertain else []), '2026-05-17T14:00:00+00:00'))
            video_store.review_event(library, mission, 'merge-source', 0, 'Operator',
                'merge_tracks', {'target_event_id': 'merge-target'})
            video_store.review_event(library, mission, 'rejected-truck', 0, 'Operator', 'reject')
            scope = video_scope_snapshot(library, mission, run['id'])
            spec = interpret('Find trucks on Sunday between 17:00 and 20:00',
                mission_id=mission, run_id=run['id'], scope_snapshot=scope, media_type='videos')
            self.assertEqual(spec['media_type'], 'videos')
            result = execute_video(library, mission, spec, confirmed=True,
                run=run['id'], scope_snapshot=scope)
            self.assertEqual([item['event_id'] for item in result['results']], ['merge-target'])
            self.assertEqual(result['uncertain_events'], 1)
            self.assertEqual(result['results'][0]['source_relative_path'], 'traffic.mkv')
            with self.assertRaisesRegex(ValueError, 'changed'):
                execute_video(library, mission, spec, confirmed=True, run='another-run', scope_snapshot=scope)

    def test_video_local_time_search_requires_verified_utc_origin(self):
        with self.assertRaisesRegex(ValueError, 'verified UTC origin'):
            interpret('Find cars after 17:00', mission_id='mission', run_id='run-1',
                scope_snapshot={'schedule': {'timezone': 'Asia/Jerusalem'}}, media_type='videos')

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

    def test_search_applies_named_view_weekday_and_local_time_filters(self):
        def row(image_id, captured, view_id='north'):
            return {'id': image_id, 'name': image_id + '.jpg', 'capture_utc': captured,
                'view_id': view_id, 'view_name': 'North gate' if view_id == 'north' else 'South gate',
                'revision': 0, 'review_prediction_run': None, 'prediction_run': 'run-1',
                'predictions': [{'kind': 'candidate', 'label': 'truck', 'box': [.1, .1, .2, .2]}],
                'annotations': '[]'}
        scope = {'run_id': 'run-1', 'schedule': {'timezone': 'Asia/Jerusalem'},
            'view_assignments': [{'view_id': 'north', 'view_name': 'North gate'},
                                 {'view_id': 'south', 'view_name': 'South gate'}],
            'regions_by_view': {}}
        query = 'Find trucks on Sunday between 17:00 and 20:00 at North gate'
        spec = interpret(query, mission_id='mission', run_id='run-1', scope_snapshot=scope)
        self.assertEqual(spec['view_id'], 'north')
        self.assertEqual(spec['schedule_filter'], {'start_time': '17:00', 'end_time': '20:00', 'weekdays': [6]})
        result = execute(SearchLibrary([
            row('inside', '2026-05-17T16:04:00Z'),
            row('late', '2026-05-17T18:30:00Z'),
            row('other-view', '2026-05-17T16:04:00Z', 'south'),
        ]), 'mission', spec, confirmed=True, scope_snapshot=scope)
        self.assertEqual([item['image_id'] for item in result['results']], ['inside'])
        self.assertEqual(result['scope_filtered_images'], 2)

    def test_search_applies_named_region_geometry_and_saved_exclusion_precedence(self):
        target=[[.1,.1],[.8,.1],[.8,.8],[.1,.8]]
        excluded=[[.3,.3],[.4,.3],[.4,.4],[.3,.4]]
        scope={'run_id':'run-1','schedule':{'timezone':'Asia/Jerusalem'},
            'view_assignments':[{'view_id':'north','view_name':'North gate'}],
            'regions_by_view':{'north':[
                {'id':'roi-1','revision':2,'name':'Northbound lane','kind':'include','polygon':target},
                {'id':'roi-2','revision':1,'name':'Loading bay','kind':'exclude','polygon':excluded}]}}
        spec=interpret('Find trucks inside Northbound lane',mission_id='mission',run_id='run-1',scope_snapshot=scope)
        self.assertEqual(spec['view_id'],'north')
        self.assertEqual(spec['region_filter']['region_id'],'roi-1')
        row={'id':'inside','name':'inside.jpg','capture_utc':None,'view_id':'north','view_name':'North gate',
            'revision':0,'review_prediction_run':None,'prediction_run':'run-1',
            'predictions':[{'kind':'candidate','label':'truck','box':[.18,.18,.04,.04]},
                {'kind':'candidate','label':'truck','box':[.33,.33,.04,.04]},
                {'kind':'candidate','label':'truck','box':[.9,.9,.04,.04]}], 'annotations':'[]'}
        result=execute(SearchLibrary([row]),'mission',spec,confirmed=True,scope_snapshot=scope)
        self.assertEqual(result['matched_objects'],1)
        self.assertEqual(result['results'][0]['objects'][0]['box'],[.18,.18,.04,.04])

    def test_named_region_search_rejects_ambiguous_regions(self):
        polygon=[[.1,.1],[.9,.1],[.9,.9],[.1,.9]]
        scope={'view_assignments':[], 'regions_by_view':{
            'north':[{'id':'north-roi','name':'Driveway','kind':'include','polygon':polygon}],
            'south':[{'id':'south-roi','name':'Driveway','kind':'include','polygon':polygon}]}}
        with self.assertRaisesRegex(ValueError,'analysis region'):
            interpret('Find cars in Driveway',scope_snapshot=scope)

    def test_hebrew_named_region_query_resolves_saved_polygon(self):
        scope={'view_assignments':[], 'regions_by_view':{'north':[
            {'id':'north-roi','name':'נתיב צפון','kind':'include',
             'polygon':[[.1,.1],[.9,.1],[.9,.9],[.1,.9]]}]}}
        spec=interpret('מצא משאיות בנתיב צפון',scope_snapshot=scope)
        self.assertEqual(spec['region_filter']['region_id'],'north-roi')
        self.assertEqual(spec['view_id'],'north')

    def test_plain_language_roi_references_resolve_only_exact_saved_region_names(self):
        scope={'view_assignments':[], 'regions_by_view':{'north':[
            {'id':'north-roi','name':'Northbound lane','kind':'include',
             'polygon':[[.1,.1],[.9,.1],[.9,.9],[.1,.9]]}]}}
        for query in ('Find trucks in ROI Northbound lane',
                      'Find trucks in the saved region named Northbound lane',
                      'Find trucks on Northbound lane'):
            with self.subTest(query=query):
                spec=interpret(query,scope_snapshot=scope)
                self.assertEqual(spec['region_filter']['region_id'],'north-roi')
        with self.assertRaisesRegex(ValueError,'Unsupported search terms'):
            interpret('Find trucks near Northbound lane',scope_snapshot=scope)

    def test_overnight_weekday_filter_uses_start_day_and_hebrew_query(self):
        scope = {'run_id': 'run-1', 'schedule': {'timezone': 'Asia/Jerusalem'},
            'view_assignments': [], 'regions_by_view': {}}
        query = 'מצא משאיות ביום ראשון בין 17:00 ל 05:00'
        spec = interpret(query, mission_id='mission', run_id='run-1', scope_snapshot=scope)
        self.assertEqual(spec['schedule_filter']['weekdays'], [6])
        self.assertEqual(spec['schedule_filter']['start_time'], '17:00')
        self.assertEqual(spec['schedule_filter']['end_time'], '05:00')
        row = {'id': 'after-midnight', 'name': 'monday.jpg', 'capture_utc': '2026-05-17T23:00:00Z',
            'view_id': None, 'view_name': None, 'revision': 0, 'review_prediction_run': None,
            'prediction_run': 'run-1', 'predictions': [{'kind': 'candidate', 'label': 'truck'}],
            'annotations': '[]'}
        result = execute(SearchLibrary([row]), 'mission', spec, confirmed=True, scope_snapshot=scope)
        self.assertEqual(result['matched_objects'], 1)

    def test_explicit_weekday_overnight_span_does_not_repeat_on_end_weekday(self):
        scope = {'run_id': 'run-1', 'schedule': {'timezone': 'Asia/Jerusalem'},
            'view_assignments': [], 'regions_by_view': {}}
        spec = interpret('Find trucks from Sunday 17:00 through Monday 05:00',
            mission_id='mission', run_id='run-1', scope_snapshot=scope)
        self.assertEqual(spec['schedule_filter'], {
            'start_time': '17:00', 'end_time': '05:00', 'weekdays': [6]})
        hebrew = interpret('מצא משאיות מיום ראשון 17:00 עד יום שני 05:00',
            mission_id='mission', run_id='run-1', scope_snapshot=scope)
        self.assertEqual(hebrew['schedule_filter'], spec['schedule_filter'])

        def row(image_id, captured):
            return {'id': image_id, 'name': image_id + '.jpg', 'capture_utc': captured,
                'view_id': None, 'view_name': None, 'revision': 0, 'review_prediction_run': None,
                'prediction_run': 'run-1', 'predictions': [{'kind': 'candidate', 'label': 'truck',
                    'box': [.1, .1, .2, .2]}], 'annotations': '[]'}
        result = execute(SearchLibrary([
            row('sunday-evening', '2026-05-17T15:00:00Z'),
            row('monday-before-end', '2026-05-17T23:00:00Z'),
            row('monday-evening', '2026-05-18T15:00:00Z'),
        ]), 'mission', spec, confirmed=True, scope_snapshot=scope)
        self.assertEqual([item['image_id'] for item in result['results']],
            ['sunday-evening', 'monday-before-end'])

    def test_weekday_overnight_span_rejects_nonconsecutive_or_nonovernight_windows(self):
        for query in ('Find trucks from Sunday 17:00 through Tuesday 05:00',
                      'Find trucks from Sunday 05:00 through Monday 17:00'):
            with self.subTest(query=query), self.assertRaisesRegex(ValueError, 'weekday overnight'):
                interpret(query, mission_id='mission')

    def test_search_applies_multiple_explicit_local_time_windows(self):
        scope = {'run_id': 'run-1', 'schedule': {'timezone': 'Asia/Jerusalem'},
            'view_assignments': [], 'regions_by_view': {}}
        spec = interpret('Find trucks on Sunday between 17:00 and 18:00 or between 20:00 and 21:00',
            mission_id='mission', run_id='run-1', scope_snapshot=scope)
        self.assertEqual(spec['schedule_filter'], {'weekdays': [6], 'time_windows': [
            {'start_time': '17:00', 'end_time': '18:00'},
            {'start_time': '20:00', 'end_time': '21:00'}]})

        def row(image_id, captured):
            return {'id': image_id, 'name': image_id + '.jpg', 'capture_utc': captured,
                'view_id': None, 'view_name': None, 'revision': 0, 'review_prediction_run': None,
                'prediction_run': 'run-1', 'predictions': [{'kind': 'candidate', 'label': 'truck',
                    'box': [.1, .1, .2, .2]}], 'annotations': '[]'}
        result = execute(SearchLibrary([
            row('first-window', '2026-05-17T14:30:00Z'),
            row('between-windows', '2026-05-17T16:30:00Z'),
            row('second-window', '2026-05-17T17:30:00Z'),
            row('wrong-weekday', '2026-05-18T14:30:00Z'),
        ]), 'mission', spec, confirmed=True, scope_snapshot=scope)
        self.assertEqual([item['image_id'] for item in result['results']],
            ['first-window', 'second-window'])

    def test_search_rejects_more_than_four_explicit_time_windows(self):
        query = ('Find cars between 01:00 and 02:00 or between 03:00 and 04:00 '
                 'or between 05:00 and 06:00 or between 07:00 and 08:00 '
                 'or between 09:00 and 10:00')
        with self.assertRaisesRegex(ValueError, 'at most four'):
            interpret(query, mission_id='mission')

    def test_weekday_and_weekend_groups_are_supported_in_english_and_hebrew(self):
        scope = {'run_id': 'run-1', 'schedule': {'timezone': 'Asia/Jerusalem'},
            'view_assignments': [], 'regions_by_view': {}}
        weekends = interpret('Find trucks on weekends', mission_id='mission',
            run_id='run-1', scope_snapshot=scope)
        hebrew_weekends = interpret('מצא משאיות בסופי שבוע', mission_id='mission',
            run_id='run-1', scope_snapshot=scope)
        self.assertEqual(weekends['schedule_filter']['weekdays'], [5, 6])
        self.assertEqual(hebrew_weekends['schedule_filter']['weekdays'], [5, 6])
        weekdays = interpret('Find trucks on weekdays', mission_id='mission',
            run_id='run-1', scope_snapshot=scope)
        hebrew_weekdays = interpret('מצא משאיות בימי חול', mission_id='mission',
            run_id='run-1', scope_snapshot=scope)
        self.assertEqual(weekdays['schedule_filter']['weekdays'], [0, 1, 2, 3, 4])
        self.assertEqual(hebrew_weekdays['schedule_filter']['weekdays'], [0, 1, 2, 3, 4])

        def row(image_id, captured):
            return {'id': image_id, 'name': image_id + '.jpg', 'capture_utc': captured,
                'view_id': None, 'view_name': None, 'revision': 0, 'review_prediction_run': None,
                'prediction_run': 'run-1', 'predictions': [{'kind': 'candidate', 'label': 'truck',
                    'box': [.1, .1, .2, .2]}], 'annotations': '[]'}
        result = execute(SearchLibrary([
            row('friday', '2026-05-15T09:00:00Z'),
            row('saturday', '2026-05-16T09:00:00Z'),
            row('sunday', '2026-05-17T09:00:00Z'),
        ]), 'mission', weekends, confirmed=True, scope_snapshot=scope)
        self.assertEqual([item['image_id'] for item in result['results']], ['saturday', 'sunday'])

    def test_search_interprets_an_explicit_inclusive_iso_date_range(self):
        scope = {'run_id': 'run-1', 'schedule': {'timezone': 'Europe/Riga'},
            'view_assignments': [], 'regions_by_view': {}}
        spec = interpret('Find cars between 2026-05-17 and 2026-05-19',
            mission_id='mission', run_id='run-1', scope_snapshot=scope)
        self.assertEqual(spec['schedule_filter']['start_date'], '2026-05-17')
        self.assertEqual(spec['schedule_filter']['end_date'], '2026-05-19')

    def test_named_dayparts_are_local_time_filters_in_english_and_hebrew(self):
        scope = {'run_id': 'run-1', 'schedule': {'timezone': 'Asia/Jerusalem'},
            'view_assignments': [], 'regions_by_view': {}}
        morning = interpret('Find trucks in the morning on Monday', run_id='run-1',
            scope_snapshot=scope)
        evening = interpret('מצא משאיות בערב', run_id='run-1', scope_snapshot=scope)
        night = interpret('Find cars at night', run_id='run-1', scope_snapshot=scope)
        self.assertEqual(morning['schedule_filter'], {
            'start_time': '05:00', 'end_time': '12:00', 'weekdays': [0]})
        self.assertEqual(evening['schedule_filter'], {
            'start_time': '17:00', 'end_time': '21:00'})
        self.assertEqual(night['schedule_filter'], {
            'start_time': '21:00', 'end_time': '05:00'})
        with self.assertRaisesRegex(ValueError, 'either named'):
            interpret('Find cars in the morning between 06:00 and 09:00',
                run_id='run-1', scope_snapshot=scope)

    def test_english_after_and_hebrew_before_use_saved_local_timezone(self):
        scope={'run_id':'run-1','schedule':{'timezone':'Europe/Riga'},
            'view_assignments':[], 'regions_by_view':{}}
        rows=[]
        for image_id,captured in [('evening','2026-05-17T15:00:00Z'),
            ('afternoon','2026-05-17T13:30:00Z'),('early','2026-05-17T02:30:00Z'),
            ('morning','2026-05-17T03:30:00Z')]:
            rows.append({'id':image_id,'name':image_id+'.jpg','capture_utc':captured,'view_id':None,
                'view_name':None,'revision':0,'review_prediction_run':None,'prediction_run':'run-1',
                'predictions':[{'kind':'candidate','label':'truck','box':[.1,.1,.1,.1]}],'annotations':'[]'})
        after=interpret('Find trucks after 17:00',mission_id='mission',run_id='run-1',scope_snapshot=scope)
        after_result=execute(SearchLibrary(rows),'mission',after,confirmed=True,scope_snapshot=scope)
        self.assertEqual([row['image_id'] for row in after_result['results']],['evening'])
        before=interpret('מצא משאיות לפני 06:00',mission_id='mission',run_id='run-1',scope_snapshot=scope)
        before_result=execute(SearchLibrary(rows),'mission',before,confirmed=True,scope_snapshot=scope)
        self.assertEqual([row['image_id'] for row in before_result['results']],['early'])

    def test_before_midnight_is_an_empty_local_time_filter(self):
        scope={'schedule':{'timezone':'Asia/Jerusalem'},'view_assignments':[],'regions_by_view':{}}
        spec=interpret('Find cars before 00:00',scope_snapshot=scope)
        self.assertTrue(spec['schedule_filter']['empty_time_range'])

    def test_temporal_search_requires_saved_iana_timezone(self):
        with self.assertRaisesRegex(ValueError, 'recorded IANA time zone'):
            interpret('Find cars on Monday', mission_id='mission')

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

    def test_search_filters_known_operator_attributes_and_keeps_unknowns_uncertain(self):
        annotations=[
            {'kind':'vehicle','label':'truck','box':[.1,.1,.2,.2],'vehicle_attributes':{'color':'red','subtype':'pickup'}},
            {'kind':'vehicle','label':'truck','box':[.35,.1,.2,.2],'vehicle_attributes':{'color':'blue','subtype':'pickup'}},
            {'kind':'candidate','label':'truck','box':[.6,.1,.2,.2]},
        ]
        row={'id':'image-1','name':'01.jpg','capture_utc':'2026-01-01T00:00:00Z','view_id':None,'view_name':None,
            'revision':1,'review_prediction_run':'run-1','prediction_run':'run-1','predictions':[],
            'annotations':json.dumps(annotations)}
        spec=interpret('Find red pickup trucks',mission_id='mission',run_id='run-1')
        result=execute(SearchLibrary([row]),'mission',spec,confirmed=True,run='run-1')
        objects=result['results'][0]['objects']
        self.assertEqual(len(objects),2)
        self.assertEqual(objects[0]['uncertain_attributes'],[])
        self.assertEqual(objects[1]['uncertain_attributes'],['color','subtype'])
        self.assertEqual(result['uncertain_objects'],1)
        exclude_blue=interpret('Find trucks except blue',mission_id='mission',run_id='run-1')
        filtered=execute(SearchLibrary([row]),'mission',exclude_blue,confirmed=True,run='run-1')
        filtered_objects=filtered['results'][0]['objects']
        self.assertEqual(len(filtered_objects),2)
        self.assertEqual([item['uncertain_attributes'] for item in filtered_objects],[[],['color']])

    def test_manual_review_without_inference_run_is_searchable(self):
        row={'id':'image-1','name':'manual.jpg','capture_utc':None,'view_id':None,'view_name':None,
            'revision':1,'review_prediction_run':None,'prediction_run':None,'predictions':[],
            'annotations':json.dumps([{'kind':'vehicle','label':'truck','box':[.1,.1,.2,.2],
                'vehicle_attributes':{'color':'red','subtype':'pickup'}}])}
        spec=interpret('Find red pickup trucks',mission_id='mission')
        result=execute(SearchLibrary([row]),'mission',spec,confirmed=True)
        self.assertEqual(result['matched_objects'],1)
        self.assertEqual(result['results'][0]['objects'][0]['source'],'human-reviewed')
        self.assertEqual(result['results'][0]['objects'][0]['uncertain_attributes'],[])

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
