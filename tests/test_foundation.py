import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import threading
import unittest
import uuid
from types import SimpleNamespace
from unittest.mock import patch
from road_traffic.domain import capture_time, named_zone, budget
from road_traffic.store import (Library, safe_file, digest, _estimate_csv_size, _estimate_xlsx_size,
    _image_export_values)
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
    def test_mission_connection_closes_after_context(self):
        with self.lib.connect(self.mission) as connection:
            self.assertEqual(connection.execute('SELECT 1').fetchone()[0],1)
        with self.assertRaises(sqlite3.ProgrammingError):
            connection.execute('SELECT 1')

    def test_mission_request_id_makes_retries_idempotent(self):
        request_id=str(uuid.uuid4())
        created=self.lib.create('Retry mission','Lebanon','Asia/Beirut',request_id=request_id)
        repeated=self.lib.create('Retry mission','Lebanon','Asia/Beirut',request_id=request_id)
        self.assertEqual(created,request_id)
        self.assertEqual(repeated,created)
        with self.assertRaisesRegex(ValueError,'already in use'):
            self.lib.create('Different mission','Lebanon','Asia/Beirut',request_id=request_id)
        self.assertEqual([row['id'] for row in self.lib.missions()].count(request_id),1)

    def test_monitor_mission_listing_cannot_see_half_created_database(self):
        real_connect=sqlite3.connect
        opened=threading.Event();finish=threading.Event();result={}
        def delayed_connect(database,*args,**kwargs):
            connection=real_connect(database,*args,**kwargs)
            if Path(database).name=='mission.sqlite3' and Path(database).parent.parent.name=='.mission-initializing':
                opened.set()
                if not finish.wait(5):raise TimeoutError('test did not release mission creation')
            return connection
        def create_mission():
            try:result['mission']=self.lib.create('Atomic mission','Iran','Asia/Tehran')
            except BaseException as exc:result['error']=exc
        with patch('road_traffic.store.sqlite3.connect',side_effect=delayed_connect):
            thread=threading.Thread(target=create_mission)
            thread.start()
            try:
                self.assertTrue(opened.wait(30),'mission initialization did not pause')
                self.assertEqual([row['id'] for row in self.lib.missions()],[self.mission])
            finally:
                finish.set()
                thread.join(30)
        self.assertFalse(thread.is_alive())
        self.assertNotIn('error',result)
        self.assertIn(result['mission'],[row['id'] for row in self.lib.missions()])

    def test_failed_mission_publication_cleans_staging_without_masking_error(self):
        replace=Path.replace
        def fail_publication(path,target):
            if path.parent.name=='.mission-initializing':
                raise OSError('injected publication failure')
            return replace(path,target)
        with patch.object(Path,'replace',fail_publication):
            with self.assertRaisesRegex(OSError,'injected publication failure'):
                self.lib.create('Unpublished mission','Israel','Asia/Jerusalem')
        staging=self.lib.root/'.mission-initializing'
        self.assertFalse(any(staging.iterdir()))
        self.assertEqual([row['id'] for row in self.lib.missions()],[self.mission])


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

    def test_operator_vehicle_attributes_are_validated_and_preserved_in_review(self):
        annotation={'kind':'vehicle','label':'truck','box':[.1,.2,.3,.4],
            'vehicle_attributes':{'color':'red','subtype':'pickup'}}
        self.lib.review(self.mission,self.image['id'],0,[annotation],'Test operator')
        saved=json.loads(self.lib.images(self.mission)[0]['annotations'])[0]
        self.assertEqual(saved['vehicle_attributes'],{'color':'red','subtype':'pickup'})
        bad=[dict(annotation,vehicle_attributes={'color':'infrared'}),
             dict(annotation,vehicle_attributes={'make':'Ford'}),
             dict(annotation,kind='light')]
        for item in bad:
            with self.subTest(item=item),self.assertRaisesRegex(ValueError,'[Vv]ehicle attributes'):
                self.lib.review(self.mission,self.image['id'],1,[item],'Test operator')

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
        with self.assertRaisesRegex(ValueError,'outside application-managed storage'):
            self.lib.create('Workspace source','Israel','Asia/Jerusalem',{'source_root':str(self.lib.root)})
        with self.assertRaisesRegex(ValueError,'outside application-managed storage'):
            self.lib.create('Workspace output','Israel','Asia/Jerusalem',{'output_root':str(self.lib.root)})

    def test_source_and_output_overlap_is_rejected_in_both_directions(self):
        output=self.source/'generated-exports';output.mkdir()
        with self.assertRaisesRegex(ValueError,'must not overlap'):
            self.lib.create('Nested output','Israel','Asia/Jerusalem',{
                'source_root':str(self.source),'output_root':str(output)})
        with self.assertRaisesRegex(ValueError,'must not overlap'):
            self.lib.create('Parent output','Israel','Asia/Jerusalem',{
                'source_root':str(output),'output_root':str(self.source)})

    def test_mission_configuration_round_trip(self):
        output=self.root/'configured-output';output.mkdir()
        mission=self.lib.create('Configured','Latvia','Europe/Riga',{
            'input_type':'images','analysis_mode':'object_search','location':'Riga bridge',
            'source_root':str(self.source),'include_subdirectories':False,
            'processing_mode':'continuous','output_root':str(output),
            'export_formats':'both','resource_profile':'conservative'})
        settings=self.lib.settings(mission)
        self.assertEqual(settings['input_type'],'images')
        self.assertEqual(settings['analysis_mode'],'object_search')
        self.assertEqual(settings['location'],'Riga bridge')
        self.assertFalse(settings['include_subdirectories'])
        self.assertEqual(settings['processing_mode'],'continuous')
        self.assertEqual(settings['resource_profile'],'conservative')
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

    def test_image_import_checks_required_free_space_before_creating_a_run(self):
        with patch('road_traffic.store.shutil.disk_usage',return_value=SimpleNamespace(free=0)):
            with self.assertRaisesRegex(ValueError,'Image import needs .* bytes are free'):
                self.lib.import_folder(self.mission,self.source,'camera')
        with self.lib.connect(self.mission) as connection:
            self.assertEqual(connection.execute('SELECT COUNT(*) FROM runs').fetchone()[0],1)
            self.assertEqual(connection.execute('SELECT COUNT(*) FROM images').fetchone()[0],1)

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

    def test_output_directory_is_write_probed_and_probe_file_is_removed(self):
        output=self.root/'write-probe-output';output.mkdir()
        mission=self.lib.create('Writable output','Israel','Asia/Jerusalem',{'output_root':str(output)})
        self.assertTrue(self.lib.settings(mission)['output_root'])
        self.assertEqual(list(output.iterdir()),[])
        with patch('road_traffic.store.tempfile.NamedTemporaryFile',side_effect=PermissionError('ACL denies create')):
            with self.assertRaisesRegex(ValueError,'Output directory failed a write check'):
                self.lib.create('Read-only output','Israel','Asia/Jerusalem',{'output_root':str(output)})

    def test_csv_and_xlsx_exports_check_estimated_space_before_ledger_or_file_creation(self):
        for format_name in ('csv','xlsx'):
            with self.subTest(format=format_name):
                output=self.root/f'output-{format_name}';output.mkdir()
                mission=self.lib.create(f'Export {format_name}','Israel','Asia/Jerusalem',{
                    'output_root':str(output),'export_formats':[format_name]})
                self.lib.import_folder(mission,self.source,'camera')
                method=self.lib.export_csv if format_name=='csv' else self.lib.export_xlsx
                with patch('road_traffic.store.shutil.disk_usage',return_value=SimpleNamespace(free=0)):
                    with self.assertRaisesRegex(ValueError,'Export needs approximately .* bytes including a safety margin'):
                        method(mission)
                with patch('road_traffic.store.shutil.disk_usage',side_effect=OSError('volume unavailable')):
                    with self.assertRaisesRegex(ValueError,'Could not check free space in output volume'):
                        method(mission)
                with self.lib.connect(mission) as conn:
                    self.assertEqual(conn.execute('SELECT COUNT(*) FROM exports').fetchone()[0],0)
                self.assertEqual(list(output.iterdir()),[])

    def test_export_size_estimates_cover_serialized_unicode_review_outputs(self):
        output=self.root/'estimated-export-output';output.mkdir()
        mission=self.lib.create('Estimated export','Israel','Asia/Jerusalem',{
            'output_root':str(output),'export_formats':['csv','xlsx']})
        self.lib.import_folder(mission,self.source,'camera')
        image=self.lib.images(mission)[0]
        self.lib.review(mission,image['id'],0,[{'kind':'vehicle','label':'car','box':[.1,.2,.3,.4]}],
            '=מפעיל','training','reviewed')
        with self.lib.connect(mission) as conn:
            snapshot_rowid=conn.execute('SELECT MAX(rowid) FROM images').fetchone()[0]
            count=conn.execute('SELECT COUNT(*) FROM images WHERE rowid<=?',(snapshot_rowid,)).fetchone()[0]
        csv_estimate=_estimate_csv_size(self.lib,mission,None,snapshot_rowid,None)
        xlsx_estimate=_estimate_xlsx_size(self.lib,mission,None,snapshot_rowid,None,count)
        with patch('road_traffic.store.shutil.disk_usage',return_value=SimpleNamespace(free=10**9)):
            result=self.lib.export_configured(mission)
        actual={Path(item['path']).suffix[1:]:Path(item['path']).stat().st_size for item in result['files']}
        self.assertLessEqual(actual['csv'],csv_estimate)
        self.assertLessEqual(actual['xlsx'],xlsx_estimate)

    def test_full_disk_export_failure_removes_staging_file_and_keeps_failure_record(self):
        original_values=_image_export_values
        for format_name in ('csv','xlsx'):
            with self.subTest(format=format_name):
                output=self.root/f'atomic-output-{format_name}';output.mkdir()
                mission=self.lib.create(f'Atomic export {format_name}','Israel','Asia/Jerusalem',{
                    'output_root':str(output),'export_formats':[format_name]})
                self.lib.import_folder(mission,self.source,'camera')
                calls=0
                def fail_during_write(*args,**kwargs):
                    nonlocal calls
                    calls+=1
                    if calls==2:
                        raise OSError('disk full')
                    return original_values(*args,**kwargs)
                method=self.lib.export_csv if format_name=='csv' else self.lib.export_xlsx
                with patch('road_traffic.store._image_export_values',side_effect=fail_during_write):
                    with self.assertRaisesRegex(ValueError,'failed: disk full'):
                        method(mission)
                with self.lib.connect(mission) as conn:
                    row=conn.execute('SELECT status,error FROM exports').fetchone()
                self.assertEqual(row['status'],'failed')
                self.assertIn('disk full',row['error'])
                self.assertEqual(list(output.iterdir()),[])

    def test_publishing_export_reconciles_atomically_and_removes_stale_temp(self):
        mission=self.lib.create('Publishing recovery','Israel','Asia/Jerusalem')
        target=self.root/'published.csv';target.write_bytes(b'complete export')
        temp=target.with_name(f'.{target.name}.partial');temp.write_bytes(b'stale partial')
        with self.lib.connect(mission) as conn:
            conn.execute('''INSERT INTO exports(id,mission_id,format,output_path,snapshot_sha256,row_count,
                status,created) VALUES (?,?,?,?,?,?,?,?)''',
                ('publishing-id',mission,'csv',str(target),digest(target),4,'publishing','2026-01-01'))
        self.assertEqual(self.lib.reconcile_exports(),1)
        record={item['id']:item for item in self.lib.exports(mission)}['publishing-id']
        self.assertEqual(record['status'],'completed')
        self.assertEqual(record['snapshot_sha256'],digest(target))
        self.assertFalse(temp.exists())

    def test_publishing_export_recovers_verified_staging_file_before_rename(self):
        mission=self.lib.create('Staged publish recovery','Israel','Asia/Jerusalem')
        target=self.root/'staged-recovery.xlsx'
        temp=target.with_name(f'.{target.name}.partial');temp.write_bytes(b'complete staged export')
        with self.lib.connect(mission) as conn:
            conn.execute('''INSERT INTO exports(id,mission_id,format,output_path,snapshot_sha256,row_count,
                status,created) VALUES (?,?,?,?,?,?,?,?)''',
                ('staged-id',mission,'xlsx',str(target),digest(temp),2,'publishing','2026-01-01'))
        self.assertEqual(self.lib.reconcile_exports(),1)
        record={item['id']:item for item in self.lib.exports(mission)}['staged-id']
        self.assertEqual(record['status'],'completed')
        self.assertEqual(target.read_bytes(),b'complete staged export')
        self.assertEqual(record['snapshot_sha256'],digest(target))
        self.assertFalse(temp.exists())

    def test_legacy_mission_gets_additive_settings_migration(self):
        mission='00000000-0000-4000-8000-000000000001'
        folder=self.lib.location(mission);folder.mkdir()
        conn=sqlite3.connect(folder/'mission.sqlite3')
        conn.execute('CREATE TABLE mission(id TEXT PRIMARY KEY,name TEXT,country TEXT,zone TEXT,created TEXT)')
        conn.execute('INSERT INTO mission VALUES (?,?,?,?,?)',(mission,'Legacy','Israel','Asia/Jerusalem','2026-01-01T00:00:00+00:00'))
        conn.commit();conn.close()
        self.assertEqual(self.lib.settings(mission)['input_type'],'images')
        self.assertEqual(self.lib.settings(mission)['export_formats'],['csv'])
        self.assertEqual(self.lib.settings(mission)['resource_profile'],'auto')

    def test_mission_rejects_unknown_resource_profile(self):
        with self.assertRaisesRegex(ValueError,'Resource profile'):
            self.lib.create('Bad profile','Israel','Asia/Jerusalem',{'resource_profile':'turbo'})

    def test_interrupted_exports_are_reconciled_after_restart(self):
        mission=self.lib.create('Export recovery','Israel','Asia/Jerusalem')
        partial=self.root/'partial.csv';partial.write_text('partial data')
        missing=self.root/'missing.csv'
        stale_temp=missing.with_name(f'.{missing.name}.partial');stale_temp.write_text('stale partial')
        with self.lib.connect(mission) as conn:
            conn.execute('INSERT INTO exports(id,mission_id,format,output_path,status,created) VALUES (?,?,?,?,?,?)',
                ('partial-id',mission,'csv',str(partial),'writing','2026-01-01T00:00:00+00:00'))
            conn.execute('INSERT INTO exports(id,mission_id,format,output_path,status,created) VALUES (?,?,?,?,?,?)',
                ('missing-id',mission,'csv',str(missing),'writing','2026-01-01T00:00:00+00:00'))
        self.assertEqual(self.lib.reconcile_exports(),2)
        history={row['id']:row for row in self.lib.exports(mission)}
        self.assertEqual(history['partial-id']['status'],'partial')
        self.assertEqual(history['missing-id']['status'],'failed')
        self.assertFalse(stale_temp.exists())

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

    def test_continuous_monitor_pause_persists_and_resume_imports(self):
        from road_traffic.monitor import SourceMonitor
        source=self.root/'paused-monitor';source.mkdir()
        mission=self.lib.create('Paused monitor','Israel','Asia/Jerusalem',{
            'source_root':str(source),'processing_mode':'continuous'})
        image=source/'17052026-1605-paused.jpg';write_test_image(image)
        self.lib.set_monitoring(mission,False)
        monitor=SourceMonitor(self.lib,stability_seconds=0)
        state=monitor.scan(mission)
        self.assertEqual(state['status'],'paused')
        self.assertFalse(state['enabled'])
        self.assertEqual(self.lib.images(mission),[])
        reopened=Library(self.lib.root)
        self.assertFalse(reopened.monitoring_enabled(mission))
        reopened.set_monitoring(mission,True)
        state=SourceMonitor(reopened,stability_seconds=0).scan(mission)
        self.assertTrue(state['enabled'])
        self.assertEqual(len(reopened.images(mission)),1)

    def test_continuous_monitor_tracks_missing_paths_without_deleting_snapshots(self):
        from road_traffic.monitor import SourceMonitor
        source=self.root/'missing-monitor';source.mkdir()
        mission=self.lib.create('Missing path monitor','Israel','Asia/Jerusalem',{
            'source_root':str(source),'processing_mode':'continuous'})
        image=source/'17052026-1605-camera.jpg';write_test_image(image)
        monitor=SourceMonitor(self.lib,stability_seconds=0)
        monitor.scan(mission)
        image.unlink()
        state=monitor.scan(mission)
        self.assertEqual(state['files']['missing'],1)
        self.assertEqual(state['missing_paths'][0]['relative_path'],image.name)
        self.assertEqual(len(self.lib.images(mission)),1)
        write_test_image(image)
        state=monitor.scan(mission)
        self.assertNotIn('missing',state['files'])
        self.assertEqual(len(self.lib.images(mission)),1)

if __name__=='__main__':unittest.main()
