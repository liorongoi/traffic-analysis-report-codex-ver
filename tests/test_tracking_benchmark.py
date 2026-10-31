from contextlib import closing
import json
import tempfile
import unittest
from pathlib import Path

from road_traffic.tracking_benchmark import (
    evaluate_files, evaluate_tracking, export_video_run, _maximum_assignment,
)


def obj(track, x=0, category=1):
    return {'track_id': track, 'category_id': category, 'bbox': [x, 0, 10, 10]}


def sequence(frames, metadata=None):
    return {'sequences': [{'sequence_id': 'camera-A', 'metadata': metadata or {},
                           'frames': [{'frame_id': index, 'objects': values}
                                      for index, values in enumerate(frames)]}]}


class TrackingBenchmarkTests(unittest.TestCase):
    def test_perfect_tracks_have_perfect_detection_and_identity_scores(self):
        data = sequence([[obj('car-1')], [obj('car-1')], [obj('car-2', 20)]],
                        {'country': 'Lebanon', 'condition': 'night'})
        result = evaluate_tracking(data, data)
        metrics = result['overall']
        self.assertEqual(metrics['idf1'], 1)
        self.assertEqual(metrics['mota'], 1)
        self.assertEqual(metrics['id_switches'], 0)
        self.assertEqual(result['strata']['country']['Lebanon']['gt_tracks'], 2)
        self.assertEqual(result['strata']['condition']['night']['frames'], 3)

    def test_identity_switch_and_split_track_are_reported(self):
        gt = sequence([[obj('gt-1')], [obj('gt-1')]])
        pred = sequence([[obj('p-1')], [obj('p-2')]])
        metrics = evaluate_tracking(gt, pred)['overall']
        self.assertEqual(metrics['id_switches'], 1)
        self.assertEqual(metrics['gt_tracks_split_across_predicted_ids'], 1)
        self.assertEqual(metrics['fraction_gt_tracks_split_across_predicted_ids'], 1)
        self.assertLess(metrics['idf1'], 1)
        self.assertLess(metrics['mota'], 1)

    def test_missed_observation_recovery_counts_fragmentation(self):
        gt = sequence([[obj('gt-1')], [obj('gt-1')], [obj('gt-1')]])
        pred = sequence([[obj('p-1')], [], [obj('p-1')]])
        metrics = evaluate_tracking(gt, pred)['overall']
        self.assertEqual(metrics['fragmentations'], 1)
        self.assertEqual(metrics['fn'], 1)

    def test_global_matching_uses_maximum_total_weight_not_greedy(self):
        self.assertEqual(set(_maximum_assignment([[.9, .8], [.85, 0]])), {(0, 1), (1, 0)})

    def test_class_mismatch_is_false_positive_and_false_negative(self):
        gt = sequence([[obj('gt-1', category=1)]])
        pred = sequence([[obj('p-1', category=2)]])
        metrics = evaluate_tracking(gt, pred)['overall']
        self.assertEqual((metrics['tp'], metrics['fp'], metrics['fn']), (0, 1, 1))

    def test_unmatched_extra_track_is_counted(self):
        gt = sequence([[obj('gt-1')]])
        pred = sequence([[obj('p-1'), obj('p-2', x=30)]])
        metrics = evaluate_tracking(gt, pred)['overall']
        self.assertEqual(metrics['fp'], 1)
        self.assertEqual(metrics['predicted_tracks'], 2)
        self.assertEqual(metrics['track_count_error'], 1)

    def test_sequence_and_frame_sets_must_match(self):
        gt = sequence([[obj('gt-1')]])
        pred = {'sequences': []}
        with self.assertRaisesRegex(ValueError, 'sequence IDs'):
            evaluate_tracking(gt, pred)
        pred = sequence([[obj('p-1')], []])
        with self.assertRaisesRegex(ValueError, 'Frame mismatch'):
            evaluate_tracking(gt, pred)

    def test_invalid_boxes_and_thresholds_are_rejected(self):
        gt = sequence([[obj('gt-1')]])
        invalid = sequence([[{'track_id': 'p', 'category_id': 1, 'bbox': [0, 0, 0, 2]}]])
        with self.assertRaisesRegex(ValueError, 'positive dimensions'):
            evaluate_tracking(gt, invalid)
        with self.assertRaisesRegex(ValueError, 'iou_threshold'):
            evaluate_tracking(gt, gt, iou_threshold=0)

    def test_json_command_saves_input_hashes_and_report_atomically(self):
        data = sequence([[obj('one')]])
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            gt, pred, report = root / 'gt.json', root / 'pred.json', root / 'out.json'
            gt.write_text(json.dumps(data), encoding='utf-8')
            pred.write_text(json.dumps(data), encoding='utf-8')
            result = evaluate_files(gt, pred, report)
            saved = json.loads(report.read_text(encoding='utf-8'))
            self.assertEqual(saved['overall']['idf1'], 1)
            self.assertEqual(len(saved['provenance']['ground_truth_sha256']), 64)
            self.assertEqual(result['metric_version'], 'roadloom-mot-v1')
            self.assertFalse(report.with_name(report.name + '.tmp').exists())

    def test_completed_video_run_exports_pixel_boxes_and_model_lineage(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            database = root / 'mission.sqlite3'
            import sqlite3
            with closing(sqlite3.connect(database)) as conn, conn:
                conn.executescript('''
                    CREATE TABLE mission(id TEXT PRIMARY KEY,country TEXT);
                    CREATE TABLE video_sources(id TEXT,mission_id TEXT,original_name TEXT,sha256 TEXT,metadata TEXT);
                    CREATE TABLE video_runs(id TEXT,mission_id TEXT,source_id TEXT,status TEXT,config TEXT,runtime_details TEXT);
                    CREATE TABLE video_frame_detections(run_id TEXT,frame_index INTEGER,pts INTEGER,
                        time_base_num INTEGER,time_base_den INTEGER,detections TEXT);
                ''')
                conn.execute('INSERT INTO mission VALUES (?,?)', ('mission-1', 'Lebanon'))
                metadata = {'media_probe': {'metadata': {'width': 100, 'height': 50}}}
                conn.execute('INSERT INTO video_sources VALUES (?,?,?,?,?)',
                    ('source-1', 'mission-1', 'test.mp4', 'f' * 64, json.dumps(metadata)))
                config = {'detector': {'resource_profile': {'selected': 'conservative'},
                    'version': 'vehicle-v1', 'sha256': 'a' * 64}}
                conn.execute('INSERT INTO video_runs VALUES (?,?,?,?,?,?)',
                    ('run-1', 'mission-1', 'source-1', 'completed', json.dumps(config), '{}'))
                detections = [{'box': [.1, .2, .3, .4], 'label': 'car', 'track_id': 'track-1'},
                    {'kind':'light','box':[.5,.2,.1,.1],'label':'possible paired vehicle lights',
                     'track_id':None,'light_subtype':'uncertain_light','pairing':'hypothesis'}]
                conn.execute('INSERT INTO video_frame_detections VALUES (?,?,?,?,?,?)',
                    ('run-1', 0, 0, 1, 1000, json.dumps(detections)))
            result = export_video_run(database, 'run-1', root / 'predictions.json')
            prediction = result['sequences'][0]
            self.assertEqual(prediction['sequence_id'], 'test.mp4')
            self.assertEqual(prediction['metadata']['country'], 'Lebanon')
            self.assertEqual(prediction['metadata']['model_id'], 'vehicle-v1')
            self.assertEqual(prediction['metadata']['hardware_profile'], 'conservative')
            self.assertEqual(prediction['frames'][0]['objects'][0]['bbox'], [10, 10, 30, 20])
            self.assertEqual(len(prediction['frames'][0]['objects']),1)

    def test_incomplete_video_run_cannot_be_exported(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            database = root / 'mission.sqlite3'
            import sqlite3
            with closing(sqlite3.connect(database)) as conn, conn:
                conn.executescript('''
                    CREATE TABLE mission(id TEXT PRIMARY KEY,country TEXT);
                    CREATE TABLE video_sources(id TEXT,mission_id TEXT,original_name TEXT,sha256 TEXT,metadata TEXT);
                    CREATE TABLE video_runs(id TEXT,mission_id TEXT,source_id TEXT,status TEXT,config TEXT,runtime_details TEXT);
                    CREATE TABLE video_frame_detections(run_id TEXT,frame_index INTEGER,pts INTEGER,
                        time_base_num INTEGER,time_base_den INTEGER,detections TEXT);
                ''')
                conn.execute('INSERT INTO mission VALUES (?,?)', ('mission-1', 'Lebanon'))
                conn.execute('INSERT INTO video_sources VALUES (?,?,?,?,?)',
                    ('source-1', 'mission-1', 'test.mp4', 'f' * 64, '{}'))
                conn.execute('INSERT INTO video_runs VALUES (?,?,?,?,?,?)',
                    ('run-1', 'mission-1', 'source-1', 'running', '{}', '{}'))
            with self.assertRaisesRegex(ValueError, 'Only completed'):
                export_video_run(database, 'run-1', root / 'predictions.json')


if __name__ == '__main__':
    unittest.main()
