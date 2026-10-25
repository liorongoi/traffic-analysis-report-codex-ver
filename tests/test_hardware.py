import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from road_traffic.hardware import inventory, resolve_resource_profile


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

    def test_16_and_48_gib_profiles_keep_os_headroom_and_single_worker(self):
        gib=1024**3
        with tempfile.TemporaryDirectory() as folder, patch('road_traffic.hardware._memory',return_value=(16*gib,13*gib)):
            small=inventory(Path(folder))
        self.assertEqual(small['recommended_profile'],'conservative')
        self.assertEqual(small['resource_profiles']['conservative']['analysis_memory_gib'],8.0)
        self.assertEqual(small['resource_profiles']['conservative']['workers'],1)
        self.assertEqual(small['resource_profiles']['conservative']['cpu_threads'],1)
        self.assertGreaterEqual(small['resource_profiles']['conservative']['reserved_headroom_gib'],4.0)
        with patch('road_traffic.hardware._memory',return_value=(16*gib,13*gib)):
            override=resolve_resource_profile('expanded')
            automatic=resolve_resource_profile('auto')
        self.assertEqual((override['selected'],override['cpu_threads']),('expanded',min(4,os.cpu_count() or 1)))
        self.assertEqual(automatic['selected'],'conservative')
        with tempfile.TemporaryDirectory() as folder, patch('road_traffic.hardware._memory',return_value=(48*gib,42*gib)):
            large=inventory(Path(folder))
        self.assertEqual(large['recommended_profile'],'expanded')
        self.assertEqual(large['resource_profiles']['expanded']['analysis_memory_gib'],34.8)
        self.assertEqual(large['resource_profiles']['expanded']['workers'],1)
        self.assertEqual(large['resource_profiles']['expanded']['cpu_threads'],min(4,os.cpu_count() or 1))
        with patch('road_traffic.hardware._memory',return_value=(48*gib,42*gib)):
            automatic=resolve_resource_profile('auto')
        self.assertEqual((automatic['selected'],automatic['cpu_threads']),('expanded',min(4,os.cpu_count() or 1)))
        self.assertIn('advisory',large['resource_profiles']['expanded']['guidance'])
