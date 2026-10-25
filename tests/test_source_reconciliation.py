from pathlib import Path
import tempfile
import unittest

from PIL import Image

from road_traffic.monitor import SourceMonitor
from road_traffic.source_reconciliation import candidates, history, record
from road_traffic.store import Library


class SourceReconciliationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        root=Path(self.temp.name)
        self.library=Library(root/'workspace')
        self.source=root/'source'
        self.source.mkdir()
        self.mission=self.library.create('Continuous','Israel','Asia/Jerusalem',{
            'source_root':str(self.source),'processing_mode':'continuous'})
        self.old=self.source/'17052026-1605-camera.jpg'
        Image.new('RGB',(8,8),(30,60,90)).save(self.old,'JPEG')
        self.monitor=SourceMonitor(self.library,stability_seconds=0)
        self.monitor.scan(self.mission)
        self.old.rename(self.source/'17052026-1605-renamed.jpg')
        self.monitor.scan(self.mission)

    def tearDown(self):
        self.temp.cleanup()

    def test_manual_link_is_reversible_and_preserves_both_capture_records(self):
        proposals=candidates(self.library,self.mission)
        self.assertEqual(len(proposals),1)
        proposal=proposals[0]
        self.assertTrue(proposal['image_a_missing'] or proposal['image_b_missing'])
        linked=record(self.library,self.mission,proposal['image_a_id'],proposal['image_b_id'],
            'link','Operator A')
        self.assertTrue(linked['records_preserved'])
        self.assertEqual(linked['revision'],1)
        self.assertEqual(candidates(self.library,self.mission),[])
        with self.library.connect(self.mission) as conn:
            self.assertEqual(conn.execute('SELECT COUNT(*) FROM images').fetchone()[0],2)
            self.assertEqual(conn.execute('SELECT COUNT(*) FROM reviews').fetchone()[0],0)
        undone=record(self.library,self.mission,proposal['image_a_id'],proposal['image_b_id'],
            'unlink','Operator B')
        self.assertEqual(undone['revision'],2)
        self.assertEqual([row['action'] for row in history(self.library,self.mission)],['link','unlink'])
        self.assertEqual(len(candidates(self.library,self.mission)),1)
        with self.library.connect(self.mission) as conn:
            self.assertEqual(conn.execute('SELECT COUNT(*) FROM images').fetchone()[0],2)

    def test_checksum_mismatch_and_duplicate_current_action_are_rejected(self):
        proposal=candidates(self.library,self.mission)[0]
        record(self.library,self.mission,proposal['image_a_id'],proposal['image_b_id'],'link','Operator')
        with self.assertRaisesRegex(ValueError,'already has this current link state'):
            record(self.library,self.mission,proposal['image_a_id'],proposal['image_b_id'],'link','Operator')
        different=self.source/'17052026-1606-different.jpg'
        Image.new('RGB',(8,8),(200,10,10)).save(different,'JPEG')
        self.library.import_folder(self.mission,self.source,'camera',['17052026-1606-different.jpg'])
        with self.library.connect(self.mission) as conn:
            other=conn.execute("SELECT id FROM images WHERE name='17052026-1606-different.jpg'").fetchone()['id']
        with self.assertRaisesRegex(ValueError,'identical SHA-256'):
            record(self.library,self.mission,proposal['image_a_id'],other,'link','Operator')


if __name__=='__main__':
    unittest.main()
