import json
from pathlib import Path
import tempfile
import unittest
from road_traffic.schedules import eligible, validate_window
from road_traffic.store import Library
from road_traffic.inference_store import create_run


class ScheduleWindowTests(unittest.TestCase):
    def test_overnight_window_uses_start_day_and_half_open_boundaries(self):
        window={'weekdays':[6],'start_date':'2026-05-17','end_date':'2026-05-17',
            'start_time':'17:00','end_time':'05:00'}
        self.assertTrue(eligible('2026-05-17T17:00:00+00:00','UTC',[window]))
        self.assertTrue(eligible('2026-05-18T04:59:00+00:00','UTC',[window]))
        self.assertFalse(eligible('2026-05-18T05:00:00+00:00','UTC',[window]))
        self.assertFalse(eligible('2026-05-17T16:59:00+00:00','UTC',[window]))

    def test_dst_gap_and_fold_follow_iana_local_time_conversion(self):
        spring={'weekdays':[6],'start_time':'03:15','end_time':'04:30'}
        self.assertFalse(eligible('2026-03-29T00:30:00+00:00','Europe/Riga',[spring]))
        self.assertTrue(eligible('2026-03-29T01:00:00+00:00','Europe/Riga',[spring]))
        autumn={'weekdays':[6],'start_time':'03:15','end_time':'03:45'}
        self.assertTrue(eligible('2026-10-25T00:30:00+00:00','Europe/Riga',[autumn]))
        self.assertTrue(eligible('2026-10-25T01:30:00+00:00','Europe/Riga',[autumn]))

    def test_validation_rejects_equal_times_and_empty_weekdays(self):
        base={'weekdays':[0],'start_time':'17:00','end_time':'17:00'}
        with self.assertRaisesRegex(ValueError,'differ'):
            validate_window(base)
        with self.assertRaisesRegex(ValueError,'weekdays'):
            validate_window({**base,'weekdays':[],'end_time':'18:00'})

    def test_inference_run_snapshots_schedule_and_excludes_out_of_window_images(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);library=Library(root/'workspace')
            source=root/'source';source.mkdir()
            (source/'17052026-1604-sunday.jpg').write_bytes(b'sunday fixture')
            (source/'18052026-1200-monday.jpg').write_bytes(b'monday fixture')
            mission=library.create('Scheduled','Israel','UTC')
            library.import_folder(mission,source,'camera')
            library.add_scan_window(mission,{'weekdays':[6],'start_time':'16:00','end_time':'22:00'})
            run=create_run(library,mission,{'adapter':'test-fixture'})
            with library.connect(mission) as conn:
                record=conn.execute('SELECT total,config FROM inference_runs WHERE id=?',(run,)).fetchone()
                inputs=conn.execute('SELECT image_id FROM inference_inputs WHERE inference_id=?',(run,)).fetchall()
            self.assertEqual(record['total'],1)
            self.assertEqual(len(inputs),1)
            snapshot=json.loads(record['config'])['scan_schedule_snapshot']
            self.assertEqual(snapshot['timezone'],'UTC')
            self.assertEqual(snapshot['excluded_image_count'],1)
            rows=library.images(mission,run)
            self.assertEqual([row['inference_status'] for row in rows].count('excluded_by_schedule'),1)


if __name__=='__main__':
    unittest.main()
