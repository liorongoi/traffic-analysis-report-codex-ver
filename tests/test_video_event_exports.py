import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
from types import SimpleNamespace
from xml.etree import ElementTree
from unittest.mock import patch

from road_traffic.store import Library
from road_traffic import video_event_exports
from road_traffic.video_event_exports import export_file, export_run, exports, reconcile_exports
from road_traffic.video_store import create_run, register_video, review_event, save_source_road_region


class VideoEventExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.source_root = root / 'source'
        self.output_root = root / 'output'
        self.source_root.mkdir()
        self.output_root.mkdir()
        self.video_name = '=קamera-01.mp4'
        (self.source_root / self.video_name).write_bytes(b'video bytes fixture')
        self.library = Library(root / 'workspace')
        self.mission = self.library.create('Video export', 'Israel', 'Asia/Jerusalem', {
            'input_type': 'videos', 'analysis_mode': 'video_traffic',
            'source_root': str(self.source_root), 'output_root': str(self.output_root),
            'export_formats': 'both'})
        self.source = register_video(self.library, self.mission, self.video_name)
        save_source_road_region(self.library,self.mission,self.source['id'],
            [[0,0],[.8,0],[.8,.8],[0,.8]],'Operator')
        self.run = create_run(self.library, self.mission, self.source['id'], {
            'detector': {'version': 'fixture', 'sha256': 'fixture'},
            'tracker_policy': {'max_gap_seconds': [1, 2], 'minimum_iou': .2,
                'max_center_distance': .2, 'association_margin': .05,
                'compatible_label_groups': []}})
        self.event_id = '00000000001234'
        with self.library.connect(self.mission) as conn:
            conn.execute('''INSERT INTO video_events(id,mission_id,run_id,source_id,track_id,
                first_pts,first_time_base_num,first_time_base_den,last_pts,last_time_base_num,last_time_base_den,
                first_offset_num,first_offset_den,last_offset_num,last_offset_den,first_utc,last_utc,labels,
                observation_count,association_uncertain,representative_box,uncertainty_reasons,created)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                (self.event_id, self.mission, self.run['id'], self.source['id'], 'track-0001',
                 10, 1, 10, 20, 1, 10, 1, 1, 2, 1, None, None,
                 json.dumps(['car']), 2, 0, '[0.1,0.2,0.3,0.4]', '[]', '2026-10-25T00:00:00+00:00'))
        self.clip_id = 'clip-linked-to-event'
        with self.library.connect(self.mission) as conn:
            conn.execute('''INSERT INTO video_clip_exports(id,mission_id,run_id,event_id,source_id,operator,
                pre_seconds,post_seconds,requested_start_num,requested_start_den,requested_end_num,requested_end_den,
                output_path,source_sha256,sha256,frame_count,status,created)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                (self.clip_id, self.mission, self.run['id'], self.event_id, self.source['id'], 'Operator',
                 .5, .5, 0, 1, 3, 1, str(self.output_root / 'clip.mp4'), self.source['sha256'],
                 'f' * 64, 12, 'completed', '2026-10-25T00:00:01+00:00'))
        review_event(self.library, self.mission, self.event_id, 0, 'בּוֹדֵק', 'correct_class',
            {'label': 'טנדר'})

    def tearDown(self):
        self.temp.cleanup()

    def test_csv_and_xlsx_snapshot_keep_event_lineage_reviews_unicode_and_clip_links(self):
        with self.library.connect(self.mission) as conn:
            count=video_event_exports._row_count(conn,self.mission,self.run['id'])
            csv_estimate=video_event_exports._estimate_csv_size(conn,self.mission,self.run['id'])
            xlsx_estimate=video_event_exports._estimate_xlsx_size(conn,self.mission,self.run['id'],count)
        result = export_run(self.library, self.mission, self.run['id'], 'בּוֹדֵק')
        self.assertEqual(result['status'], 'completed')
        self.assertEqual({item['format'] for item in result['files']}, {'csv', 'xlsx'})
        self.assertTrue(all(item['row_count'] == 1 and item['status'] == 'completed'
            for item in result['files']))
        self.assertLessEqual(Path(next(item['output_path'] for item in result['files']
            if item['format']=='csv')).stat().st_size,csv_estimate)
        self.assertLessEqual(Path(next(item['output_path'] for item in result['files']
            if item['format']=='xlsx')).stat().st_size,xlsx_estimate)
        csv_export = next(item for item in result['files'] if item['format'] == 'csv')
        with export_file(self.library, self.mission, csv_export['id']).open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row['event_id'], self.event_id)
        self.assertEqual(row['source_id'], self.source['id'])
        self.assertEqual(row['source_sha256'], self.source['sha256'])
        self.assertEqual(row['run_id'], self.run['id'])
        self.assertEqual(row['video_name'], "'=קamera-01.mp4")
        self.assertEqual(json.loads(row['labels']), ['טנדר'])
        self.assertEqual(json.loads(row['original_labels']), ['car'])
        self.assertEqual(row['review_status'], 'corrected')
        self.assertEqual(json.loads(row['representative_road_relation'])['label'],'on_road')
        self.assertEqual(row['review_operator'], 'בּוֹדֵק')
        self.assertEqual(json.loads(row['review_payload']), {'label': 'טנדר'})
        clips = json.loads(row['clip_exports'])
        self.assertEqual(clips[0]['id'], self.clip_id)
        self.assertEqual(clips[0]['sha256'], 'f' * 64)

        xlsx_export = next(item for item in result['files'] if item['format'] == 'xlsx')
        xlsx_path = export_file(self.library, self.mission, xlsx_export['id'])
        with zipfile.ZipFile(xlsx_path) as archive:
            sheet = ElementTree.fromstring(archive.read('xl/worksheets/sheet1.xml'))
            namespace = {'x': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
            self.assertFalse(sheet.findall('.//x:f', namespace))
            values = [node.text or '' for node in sheet.findall('.//x:is/x:t', namespace)]
        self.assertEqual(values[:len(row)], list(row.keys()))
        self.assertIn(self.event_id, values)
        self.assertIn('{"label": "טנדר"}', values)
        self.assertIn(self.clip_id, row['clip_exports'])
        self.assertEqual(hashlib.sha256(xlsx_path.read_bytes()).hexdigest(), xlsx_export['snapshot_sha256'])
        self.assertEqual(len(exports(self.library, self.mission, self.run['id'])), 2)

    def test_event_export_rejects_low_space_before_ledger_or_file_creation(self):
        with patch('road_traffic.store.shutil.disk_usage',return_value=SimpleNamespace(free=0)):
            result=export_run(self.library,self.mission,self.run['id'],'Operator')
        self.assertEqual(result['status'],'failed')
        self.assertEqual(len(result['errors']),2)
        self.assertTrue(all('Export needs approximately' in error for error in result['errors']))
        self.assertEqual(exports(self.library,self.mission,self.run['id']),[])
        self.assertEqual(list(self.output_root.iterdir()),[])

    def test_export_is_a_fixed_run_snapshot_and_download_detects_tampering(self):
        result = export_run(self.library, self.mission, self.run['id'], 'Operator')
        export = next(item for item in result['files'] if item['format'] == 'csv')
        with self.library.connect(self.mission) as conn:
            conn.execute('''INSERT INTO video_events(id,mission_id,run_id,source_id,track_id,
                first_pts,first_time_base_num,first_time_base_den,last_pts,last_time_base_num,last_time_base_den,
                first_offset_num,first_offset_den,last_offset_num,last_offset_den,first_utc,last_utc,labels,
                observation_count,association_uncertain,uncertainty_reasons,created)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                ('later-event', self.mission, self.run['id'], self.source['id'], 'track-0002',
                 30, 1, 10, 40, 1, 10, 3, 1, 4, 1, None, None, '["bus"]', 2, 0, '[]', '2026-10-25T00:00:02+00:00'))
        with export_file(self.library, self.mission, export['id']).open(encoding='utf-8-sig', newline='') as stream:
            saved = list(csv.DictReader(stream))
        self.assertEqual(len(saved), 1)
        self.assertEqual(saved[0]['event_id'], self.event_id)
        new_output = self.output_root.parent / 'new-event-exports'
        new_output.mkdir()
        with self.library.connect(self.mission) as conn:
            conn.execute('UPDATE mission_settings SET output_root=? WHERE mission_id=?',
                (str(new_output), self.mission))
        self.assertEqual(export_file(self.library, self.mission, export['id']), Path(export['output_path']))
        Path(export['output_path']).write_bytes(b'tampered')
        with self.assertRaisesRegex(ValueError, 'SHA-256 check'):
            export_file(self.library, self.mission, export['id'])

    def test_timing_correction_exports_effective_and_original_intervals(self):
        track_id = 'track-0001'
        with self.library.connect(self.mission) as conn:
            for frame_index in (10,15,20):
                conn.execute('''INSERT INTO video_frame_detections(mission_id,run_id,source_id,frame_index,
                    pts,time_base_num,time_base_den,offset_seconds,detections,created)
                    VALUES (?,?,?,?,?,?,?,?,?,?)''',
                    (self.mission,self.run['id'],self.source['id'],frame_index,frame_index,1,10,
                     frame_index/10,json.dumps([{'track_id':track_id,'label':'car','box':[.1,.2,.3,.4]}]),
                     '2026-10-25T00:00:02+00:00'))
        review_event(self.library,self.mission,self.event_id,1,'Operator','correct_timing',
            {'first_offset_seconds':'1.4','last_offset_seconds':'1.9'})
        result = export_run(self.library,self.mission,self.run['id'],'Operator')
        csv_export = next(item for item in result['files'] if item['format']=='csv')
        with export_file(self.library,self.mission,csv_export['id']).open(
                encoding='utf-8-sig',newline='') as stream:
            row = next(csv.DictReader(stream))
        self.assertEqual(row['first_offset_seconds'],'1.5')
        self.assertEqual(row['last_offset_seconds'],'2.0')
        self.assertEqual(row['original_first_offset_seconds'],'1.0')
        self.assertEqual(row['original_last_offset_seconds'],'2.0')
        self.assertEqual(row['review_action'],'correct_timing')
        self.assertEqual(json.loads(row['review_payload'])['first_frame_index'],15)

    def test_event_export_uses_corrected_merge_identity_and_preserves_original_event_ids(self):
        source_id=self.source['id'];run_id=self.run['id'];other_id='merge-source-export'
        with self.library.connect(self.mission) as conn:
            conn.execute('''INSERT INTO video_events(id,mission_id,run_id,source_id,track_id,
                first_pts,first_time_base_num,first_time_base_den,last_pts,last_time_base_num,last_time_base_den,
                first_offset_num,first_offset_den,last_offset_num,last_offset_den,first_utc,last_utc,labels,
                observation_count,association_uncertain,representative_box,uncertainty_reasons,created)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                (other_id,self.mission,run_id,source_id,'track-0002',0,1,10,10,1,10,0,1,1,1,
                 None,None,'["car"]',2,0,'[0.2,0.2,0.2,0.2]','[]','2026-10-25T00:00:00Z'))
        review_event(self.library,self.mission,other_id,0,'Operator','merge_tracks',{'target_event_id':self.event_id})
        result=export_run(self.library,self.mission,run_id,'Operator')
        csv_export=next(item for item in result['files'] if item['format']=='csv')
        with export_file(self.library,self.mission,csv_export['id']).open(encoding='utf-8-sig',newline='') as stream:
            rows=list(csv.DictReader(stream))
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]['event_id'],self.event_id)
        self.assertEqual(json.loads(rows[0]['original_event_ids']),sorted([self.event_id,other_id]))
        self.assertEqual(rows[0]['first_offset_seconds'],'0.0')
        self.assertEqual(rows[0]['last_offset_seconds'],'2.0')
        self.assertEqual(json.loads(rows[0]['identity_correction'])['action'],'merge_tracks')

    def test_event_export_materializes_split_children_at_observed_frame(self):
        with self.library.connect(self.mission) as conn:
            for frame_index in (10,15,20):
                conn.execute('''INSERT INTO video_frame_detections(mission_id,run_id,source_id,frame_index,
                    pts,time_base_num,time_base_den,offset_seconds,detections,created)
                    VALUES (?,?,?,?,?,?,?,?,?,?)''',
                    (self.mission,self.run['id'],self.source['id'],frame_index,frame_index,1,10,
                     frame_index/10,json.dumps([{'track_id':'track-0001','label':'car','box':[.1,.2,.3,.4]}]),
                     '2026-10-25T00:00:02+00:00'))
        review_event(self.library,self.mission,self.event_id,1,'Operator','split_track',{'frame_index':15})
        result=export_run(self.library,self.mission,self.run['id'],'Operator')
        csv_export=next(item for item in result['files'] if item['format']=='csv')
        with export_file(self.library,self.mission,csv_export['id']).open(encoding='utf-8-sig',newline='') as stream:
            rows=list(csv.DictReader(stream))
        self.assertEqual(len(rows),2)
        self.assertEqual([row['first_offset_seconds'] for row in rows],['1.0','1.5'])
        self.assertEqual([row['last_offset_seconds'] for row in rows],['1.0','2.0'])
        self.assertEqual({json.loads(row['identity_correction'])['child'] for row in rows},{1,2})
        self.assertEqual({tuple(json.loads(row['original_event_ids'])) for row in rows},{(self.event_id,)})

    def test_interrupted_publish_recovers_atomically(self):
        result = export_run(self.library, self.mission, self.run['id'], 'Operator')
        export = next(item for item in result['files'] if item['format'] == 'csv')
        target = Path(export['output_path'])
        temp_path = target.with_name('.recovered-event-table.csv.partial')
        temp_path.write_bytes(target.read_bytes())
        target.unlink()
        with self.library.connect(self.mission) as conn:
            conn.execute('''UPDATE video_event_table_exports SET status='publishing',temp_path=? WHERE id=?''',
                (str(temp_path), export['id']))
        self.assertEqual(reconcile_exports(self.library), 1)
        recovered = next(item for item in exports(self.library, self.mission, self.run['id'])
            if item['id'] == export['id'])
        self.assertEqual(recovered['status'], 'completed')
        self.assertTrue(Path(recovered['output_path']).is_file())

    def test_one_format_failure_reports_partial_success_and_failed_ledger_row(self):
        with patch.object(video_event_exports, '_write_xlsx', side_effect=OSError('disk full')):
            result = export_run(self.library, self.mission, self.run['id'], 'Operator')
        self.assertEqual(result['status'], 'partial')
        self.assertIn('disk full', '; '.join(result['errors']))
        by_format = {item['format']: item['status'] for item in result['files']}
        self.assertEqual(by_format, {'csv': 'completed', 'xlsx': 'failed'})


if __name__ == '__main__':
    unittest.main()
