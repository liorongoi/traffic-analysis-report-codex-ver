import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from road_traffic.domain import capture_time, named_zone, budget
from road_traffic.store import Library, safe_file


class TimeTests(unittest.TestCase):
    def test_confirmed_filename_precision(self):
        stamp, precision=capture_time('17052026-1604-dcincincifbvhubuhbv.jpg')
        self.assertEqual(stamp.isoformat(),'2026-05-17T16:04:00+00:00')
        self.assertEqual(precision,'minute')

    def test_invalid_dates_and_paths(self):
        for name in ('31022026-1604-x.jpg','17052026-2500-x.jpg','../17052026-1604-x.jpg','random.jpg'):
            with self.subTest(name=name), self.assertRaises(ValueError):capture_time(name)

    def test_explicit_fintraffic(self):
        name='C0450701_20261020T194513Z.jpg'
        with self.assertRaises(ValueError):capture_time(name)
        stamp, precision=capture_time(name,'fintraffic')
        self.assertEqual(precision,'second')
        self.assertEqual(stamp.astimezone(named_zone('Europe/Helsinki')).hour,22)

    def test_historical_dst(self):
        winter,_=capture_time('17012026-1604-x')
        summer,_=capture_time('17072026-1604-x')
        self.assertEqual(winter.astimezone(named_zone('Europe/Helsinki')).utcoffset().total_seconds(),7200)
        self.assertEqual(summer.astimezone(named_zone('Europe/Helsinki')).utcoffset().total_seconds(),10800)

    def test_memory(self):
        self.assertLessEqual(budget(16,12)['analysis_gib'],8)
        self.assertGreater(budget(48,40)['analysis_gib'],budget(16,12)['analysis_gib'])
        self.assertEqual(budget(48,2)['analysis_gib'],0)
        self.assertEqual(budget(48,40)['device'],'cpu')


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        self.lib=Library(self.root/'workspace')
        self.source=self.root/'source'
        self.source.mkdir()
        self.file=self.source/'17052026-1604-vehicle.jpg'
        self.file.write_bytes(b'explicit-test-fixture-not-real-image')
        self.mission=self.lib.create('Test','Israel','Asia/Jerusalem')
        self.lib.import_folder(self.mission,self.source,'camera')
        self.image=self.lib.images(self.mission)[0]
        self.annotations=[{'kind':'candidate','label':'single light','box':[.1,.2,.1,.2]}]
    def tearDown(self):self.tmp.cleanup()

    def save(self,revision=0,role='none'):
        return self.lib.review(self.mission,self.image['id'],revision,self.annotations,'Test operator',role,'partial')

    def test_copy_survives_source_changes(self):
        self.file.write_bytes(b'changed')
        self.assertEqual(self.lib.image_path(self.mission,self.image['id']).read_bytes(),b'explicit-test-fixture-not-real-image')

    def test_hash_corruption_detected(self):
        path=self.lib.image_path(self.mission,self.image['id'])
        path.write_bytes(b'tampered')
        with self.assertRaises(ValueError):self.lib.image_path(self.mission,self.image['id'])

    def test_revisions_and_conflict(self):
        self.assertEqual(self.save()['revision'],1)
        with self.assertRaises(ValueError):self.save()
        self.assertEqual(self.save(1)['revision'],2)
        self.assertEqual(self.lib.images(self.mission)[0]['revision'],2)
        with self.lib.connect(self.mission) as conn:
            self.assertEqual(conn.execute('SELECT COUNT(*) FROM reviews').fetchone()[0],2)
            with self.assertRaises(sqlite3.IntegrityError):conn.execute("UPDATE reviews SET operator='changed'")

    def test_holdout_is_protected(self):
        self.save(role='test')
        with self.assertRaises(ValueError):self.save(1,'training')

    def test_mission_isolation(self):
        other=self.lib.create('Other','Latvia','Europe/Riga')
        self.assertEqual(self.lib.images(other),[])
        with self.assertRaises(ValueError):self.lib.review(other,self.image['id'],0,self.annotations,'Operator')

    def test_geometry_invalid(self):
        for box in ([0,0,2,1],[0,0,0,1],[0,0,float('nan'),1]):
            with self.subTest(box=box),self.assertRaises(ValueError):
                self.lib.review(self.mission,self.image['id'],0,[{'kind':'vehicle','label':'car','box':box}],'Operator')

    def test_source_immutable(self):
        with self.lib.connect(self.mission) as conn:
            with self.assertRaises(sqlite3.IntegrityError):conn.execute('DELETE FROM images')

    def test_escape_and_symlink(self):
        with self.assertRaises(ValueError):safe_file(self.source,'../outside')
        link=self.source/'link.jpg'
        try:link.symlink_to(self.file)
        except OSError:self.skipTest('No symlink permission')
        with self.assertRaises(ValueError):safe_file(self.source,'link.jpg')

    def test_checksum_failure_is_atomic(self):
        fintraffic=self.source/'C0450701_20261020T194513Z.jpg'
        fintraffic.write_bytes(b'explicit-test-fixture')
        (self.source/'manifest.json').write_text(json.dumps({'downloads':[{'file':fintraffic.name,'sha256':'wrong'}]}))
        with self.assertRaises(ValueError):self.lib.import_folder(self.mission,self.source,'fintraffic')
        self.assertEqual(len(self.lib.images(self.mission)),1)
        with self.lib.connect(self.mission) as conn:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM runs WHERE status='failed'").fetchone()[0],1)

    def test_own_workspace_rejected(self):
        with self.assertRaises(ValueError):self.lib.import_folder(self.mission,self.lib.root,'camera')

if __name__=='__main__':unittest.main()
