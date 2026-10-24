import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from road_traffic.domain import capture_time, named_zone, budget
from road_traffic.store import Library, safe_file
from PIL import Image


def write_test_image(path,color=(40,80,120)):
    Image.new('RGB',(8,8),color).save(path,'JPEG')


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
        write_test_image(self.file)
        self.original_bytes=self.file.read_bytes()
        self.mission=self.lib.create('Test','Israel','Asia/Jerusalem')
        self.lib.import_folder(self.mission,self.source,'camera')
        self.image=self.lib.images(self.mission)[0]
        self.annotations=[{'kind':'candidate','label':'single light','box':[.1,.2,.1,.2]}]
    def tearDown(self):self.tmp.cleanup()

    def save(self,revision=0,role='none'):
        return self.lib.review(self.mission,self.image['id'],revision,self.annotations,'Test operator',role,'partial')

    def test_copy_survives_source_changes(self):
        self.file.write_bytes(b'changed')
        self.assertEqual(self.lib.image_path(self.mission,self.image['id']).read_bytes(),self.original_bytes)

    def test_export_image_iterator_is_bounded_and_excludes_later_imports(self):
        iterator=self.lib.iter_images(self.mission,batch_size=1)
        first=next(iterator)
        later=self.source/'17052026-1605-later.jpg';write_test_image(later,(80,20,10))
        self.lib.import_folder(self.mission,self.source,'camera')
        remaining=list(iterator)
        self.assertEqual([first['id'],*[row['id'] for row in remaining]],[self.image['id']])
        with self.assertRaisesRegex(ValueError,'batch size'):
            next(self.lib.iter_images(self.mission,batch_size=0))

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
        write_test_image(fintraffic)
        (self.source/'manifest.json').write_text(json.dumps({'downloads':[{'file':fintraffic.name,'sha256':'wrong'}]}))
        with self.assertRaises(ValueError):self.lib.import_folder(self.mission,self.source,'fintraffic')
        self.assertEqual(len(self.lib.images(self.mission)),1)
        with self.lib.connect(self.mission) as conn:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM runs WHERE status='failed'").fetchone()[0],1)

    def test_corrupt_image_import_reports_filename_and_rolls_back_batch(self):
        corrupt=self.source/'17052026-1606-corrupt.jpg';corrupt.write_bytes(b'not a decodable image')
        with self.assertRaisesRegex(ValueError,'Corrupt or unsupported image 17052026-1606-corrupt.jpg'):
            self.lib.import_folder(self.mission,self.source,'camera')
        self.assertEqual(len(self.lib.images(self.mission)),1)
        with self.lib.connect(self.mission) as conn:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM runs WHERE status='failed'").fetchone()[0],1)

    def test_own_workspace_rejected(self):
        with self.assertRaises(ValueError):self.lib.import_folder(self.mission,self.lib.root,'camera')

    def test_mission_configuration_round_trip(self):
        output=self.root/'configured-output';output.mkdir()
        mission=self.lib.create('Configured','Latvia','Europe/Riga',{
            'input_type':'images','analysis_mode':'object_search','location':'Riga bridge',
            'source_root':str(self.source),'include_subdirectories':False,
            'processing_mode':'continuous','output_root':str(output),
            'export_formats':'both'})
        settings=self.lib.settings(mission)
        self.assertEqual(settings['input_type'],'images')
        self.assertEqual(settings['analysis_mode'],'object_search')
        self.assertEqual(settings['location'],'Riga bridge')
        self.assertFalse(settings['include_subdirectories'])
        self.assertEqual(settings['processing_mode'],'continuous')
        self.assertEqual(settings['export_formats'],['csv','xlsx'])
        self.assertEqual(self.lib.missions()[0]['source_root'],str(self.source.resolve()))

    def test_nonrecursive_import_honors_mission_setting(self):
        nested=self.source/'subfolder';nested.mkdir()
        write_test_image(nested/'17052026-1605-nested.jpg')
        mission=self.lib.create('Flat import','Israel','Asia/Jerusalem',{
            'source_root':str(self.source),'include_subdirectories':False})
        imported=self.lib.import_folder(mission,self.source,'camera')
        self.assertEqual(imported['images'],1)
        self.assertEqual(self.lib.images(mission)[0]['name'],'17052026-1604-vehicle.jpg')

    def test_video_mission_rejects_still_image_import(self):
        mission=self.lib.create('Video','Israel','Asia/Jerusalem',{
            'input_type':'videos','analysis_mode':'video_traffic'})
        with self.assertRaisesRegex(ValueError,'video input'):
            self.lib.import_folder(mission,self.source,'camera')

    def test_mission_configuration_rejects_invalid_modes_and_paths(self):
        with self.assertRaisesRegex(ValueError,'not valid'):
            self.lib.create('Bad mode','Israel','Asia/Jerusalem',{'input_type':'videos','analysis_mode':'general_traffic'})
        with self.assertRaisesRegex(ValueError,'absolute path'):
            self.lib.create('Relative','Israel','Asia/Jerusalem',{'source_root':'relative/source'})
        with self.assertRaisesRegex(ValueError,'unavailable'):
            self.lib.create('Missing','Israel','Asia/Jerusalem',{'output_root':str(self.root/'missing')})

    def test_legacy_mission_gets_additive_settings_migration(self):
        mission='00000000-0000-4000-8000-000000000001'
        folder=self.lib.location(mission);folder.mkdir()
        conn=sqlite3.connect(folder/'mission.sqlite3')
        conn.execute('CREATE TABLE mission(id TEXT PRIMARY KEY,name TEXT,country TEXT,zone TEXT,created TEXT)')
        conn.execute('INSERT INTO mission VALUES (?,?,?,?,?)',(mission,'Legacy','Israel','Asia/Jerusalem','2026-01-01T00:00:00+00:00'))
        conn.commit();conn.close()
        self.assertEqual(self.lib.settings(mission)['input_type'],'images')
        self.assertEqual(self.lib.settings(mission)['export_formats'],['csv'])

    def test_interrupted_exports_are_reconciled_after_restart(self):
        mission=self.lib.create('Export recovery','Israel','Asia/Jerusalem')
        partial=self.root/'partial.csv';partial.write_text('partial data')
        missing=self.root/'missing.csv'
        with self.lib.connect(mission) as conn:
            conn.execute('INSERT INTO exports(id,mission_id,format,output_path,status,created) VALUES (?,?,?,?,?,?)',
                ('partial-id',mission,'csv',str(partial),'writing','2026-01-01T00:00:00+00:00'))
            conn.execute('INSERT INTO exports(id,mission_id,format,output_path,status,created) VALUES (?,?,?,?,?,?)',
                ('missing-id',mission,'csv',str(missing),'writing','2026-01-01T00:00:00+00:00'))
        self.assertEqual(self.lib.reconcile_exports(),2)
        history={row['id']:row for row in self.lib.exports(mission)}
        self.assertEqual(history['partial-id']['status'],'partial')
        self.assertEqual(history['missing-id']['status'],'failed')

    def test_continuous_monitor_imports_stable_versions_once_and_keeps_replacements(self):
        from road_traffic.monitor import SourceMonitor
        source=self.root/'monitor-source';source.mkdir()
        mission=self.lib.create('Continuous images','Israel','Asia/Jerusalem',{
            'source_root':str(source),'include_subdirectories':True,'processing_mode':'continuous'})
        image=source/'17052026-1605-camera-a.jpg'
        write_test_image(image,(10,20,30))
        monitor=SourceMonitor(self.lib,stability_seconds=0)
        first=monitor.scan(mission)
        self.assertEqual(len(self.lib.images(mission)),1)
        self.assertEqual(first['files']['imported'],1)
        # A new monitor instance models a service restart against the same mission DB.
        monitor=SourceMonitor(self.lib,stability_seconds=0)
        monitor.scan(mission)
        self.assertEqual(len(self.lib.images(mission)),1)
        write_test_image(image,(30,20,10))
        monitor.scan(mission)
        rows=self.lib.images(mission)
        self.assertEqual(len(rows),2)
        self.assertEqual(len({row['sha256'] for row in rows}),2)
        self.assertEqual(monitor.status(mission)['files']['imported'],1)

    def test_monitor_keeps_corrupt_image_visible_as_per_file_error(self):
        from road_traffic.monitor import SourceMonitor
        source=self.root/'corrupt-monitor';source.mkdir()
        mission=self.lib.create('Corrupt monitor','Israel','Asia/Jerusalem',{
            'source_root':str(source),'processing_mode':'continuous'})
        (source/'17052026-1605-corrupt.jpg').write_bytes(b'not an image')
        state=SourceMonitor(self.lib,stability_seconds=0).scan(mission)
        self.assertEqual(state['files']['error'],1)
        self.assertIn('Corrupt or unsupported image',state['recent_errors'][0]['error'])
        self.assertEqual(self.lib.images(mission),[])

if __name__=='__main__':unittest.main()
