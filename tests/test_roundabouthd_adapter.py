import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from road_traffic.roundabouthd import convert_sct_labels, write_manifest
from road_traffic.tracking_benchmark import evaluate_tracking

ROOT = Path(__file__).resolve().parents[1]


class RoundaboutHDAdapterTests(unittest.TestCase):
    def test_sct_labels_become_complete_pixel_ground_truth_sequences(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels = root / 'imagesc001_SCT_GT.txt'
            source = '1 7 10 20 30 50\n3 7 12 20 32 50\n'
            labels.write_text(source, encoding='utf-8')

            result = convert_sct_labels([('camera-1', labels)], frame_count=3,
                width=100, height=80, frame_offset=1)
            sequence = result['sequences'][0]
            self.assertEqual([frame['frame_id'] for frame in sequence['frames']], [0, 1, 2])
            self.assertEqual(sequence['frames'][0]['objects'], [{
                'track_id': 'camera-1:7', 'category_id': 'vehicle', 'bbox': [10.0, 20.0, 20.0, 30.0],
            }])
            self.assertEqual(sequence['frames'][1]['objects'], [])
            self.assertEqual(sequence['frames'][2]['objects'][0]['bbox'], [12.0, 20.0, 20.0, 30.0])
            self.assertEqual(sequence['metadata']['source_annotation_sha256'],
                hashlib.sha256(labels.read_bytes()).hexdigest())

            predictions = {'sequences': [{
                'sequence_id': 'camera-1',
                'frames': [{'frame_id': frame['frame_id'], 'objects': [
                    {'track_id': 'pred-7', 'category_id': 'car', 'bbox': item['bbox']}
                    for item in frame['objects']]} for frame in sequence['frames']],
            }]}
            report = evaluate_tracking(result, predictions, class_agnostic=True)
            self.assertEqual(report['overall']['frames'], 3)
            self.assertEqual(report['overall']['ground_truth_detections'], 2)
            self.assertEqual(report['overall']['idf1'], 1)

            output = root / 'labels' / 'tracking-ground-truth.json'
            write_manifest(result, output)
            self.assertEqual(json.loads(output.read_text(encoding='utf-8')), result)
            self.assertFalse(output.with_name(output.name + '.tmp').exists())

    def test_each_camera_has_an_independent_sequence_and_track_namespace(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second = root / 'one.txt', root / 'two.txt'
            first.write_text('0 1 0 0 10 10\n', encoding='utf-8')
            second.write_text('0 1 0 0 10 10\n', encoding='utf-8')
            result = convert_sct_labels([('c001', first), ('c002', second)],
                frame_count=1, width=20, height=20)
            self.assertEqual([item['sequence_id'] for item in result['sequences']], ['c001', 'c002'])
            self.assertEqual([item['frames'][0]['objects'][0]['track_id']
                for item in result['sequences']], ['c001:1', 'c002:1'])

    def test_rejects_invalid_rows_boxes_and_frame_ranges(self):
        with tempfile.TemporaryDirectory() as directory:
            labels = Path(directory) / 'labels.txt'
            for content, message in (
                    ('0 1 0 0 10\n', 'expected 6 columns'),
                    ('3 1 0 0 10 10\n', 'outside the requested clip'),
                    ('0 1 0 0 25 10\n', 'outside 20x20'),
                    ('0 1 nan 0 10 10\n', 'must be finite'),
                    ('0 1 10 10 10 15\n', 'invalid or outside')):
                labels.write_text(content, encoding='utf-8')
                with self.subTest(content=content), self.assertRaisesRegex(ValueError, message):
                    convert_sct_labels([('camera-1', labels)], frame_count=3,
                        width=20, height=20)

    def test_rejects_duplicate_ids_and_objects(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second = root / 'one.txt', root / 'two.txt'
            first.write_text('0 1 0 0 10 10\n0 1 1 1 11 11\n', encoding='utf-8')
            second.write_text('', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'duplicate object'):
                convert_sct_labels([('camera-1', first)], frame_count=1, width=20, height=20)
            first.write_text('0 1 0 0 10 10\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'duplicate sequence ID'):
                convert_sct_labels([('camera-1', first), ('camera-1', second)],
                    frame_count=1, width=20, height=20)

    def test_converter_and_class_agnostic_evaluator_cli_work_together(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels, ground_truth, predictions, report = (root / name for name in
                ('labels.txt', 'ground-truth.json', 'predictions.json', 'report.json'))
            labels.write_text('0 7 10 20 30 50\n', encoding='utf-8')
            subprocess.run([sys.executable, str(ROOT / 'scripts' / 'convert-roundabouthd-tracking.py'),
                '--labels', f'c001={labels}', '--frame-count', '1', '--width', '100',
                '--height', '80', '--output', str(ground_truth)], check=True,
                capture_output=True, text=True)
            predictions.write_text(json.dumps({'sequences': [{
                'sequence_id': 'c001',
                'frames': [{'frame_id': 0, 'objects': [{
                    'track_id': 'pred-7', 'category_id': 'car', 'bbox': [10, 20, 20, 30],
                }]}],
            }]}), encoding='utf-8')
            subprocess.run([sys.executable, str(ROOT / 'scripts' / 'evaluate-tracking-benchmark.py'),
                '--ground-truth', str(ground_truth), '--predictions', str(predictions),
                '--class-agnostic', '--report', str(report)], check=True,
                capture_output=True, text=True)
            saved = json.loads(report.read_text(encoding='utf-8'))
            self.assertEqual(saved['category_matching'],
                'class-agnostic; all categories mapped to vehicle')
            self.assertEqual(saved['overall']['idf1'], 1)


if __name__ == '__main__':
    unittest.main()
