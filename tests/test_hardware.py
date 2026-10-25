import tempfile
import unittest
from pathlib import Path

from road_traffic.hardware import inventory


class HardwareInventoryTests(unittest.TestCase):
    def test_report_has_capacity_and_marks_unverified_acceleration(self):
        with tempfile.TemporaryDirectory() as folder:
            report = inventory(Path(folder))
        self.assertIn('operating_system', report)
        self.assertIsInstance(report['cpu']['logical_cores'], int)
        self.assertGreater(report['cpu']['logical_cores'], 0)
        self.assertIn('total_gib', report['memory'])
        self.assertIn('free_bytes', report['storage'])
        self.assertEqual(report['gpu_execution'], 'not verified by provider listing')
        self.assertEqual(report['storage_performance']['state'], 'measured')
        self.assertGreater(report['storage_performance']['write_mib_s'], 0)
        self.assertGreater(report['storage_performance']['read_mib_s'], 0)
        self.assertFalse(list(Path(folder).glob('.traffic-analysis-storage-probe-*')))
        self.assertIn(report['onnx_runtime']['state'],
            ('not_installed', 'advertised_not_exercised', 'import_failed'))
