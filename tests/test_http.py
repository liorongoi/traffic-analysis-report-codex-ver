import csv
import hashlib
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
import zipfile
import xml.etree.ElementTree as ET
from PIL import Image


def write_test_image(path,color=(45,85,125)):
    Image.new('RGB',(8,8),color).save(path,'JPEG')


class HttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory()
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',0))
            cls.port=sock.getsockname()[1]
        cls.base=f'http://127.0.0.1:{cls.port}'
        cls.proc=subprocess.Popen([sys.executable,'-m','road_traffic','serve','--port',str(cls.port),'--workspace',cls.tmp.name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        for attempt in range(100):
            try:
                with urllib.request.urlopen(cls.base+'/api/session',timeout=.2) as response:
                    cls.session=json.load(response)
                break
            except (OSError,ValueError):time.sleep(.03)
        else:
            cls.proc.terminate()
            raise RuntimeError('Development server did not become ready')
    @classmethod
    def tearDownClass(cls):
        cls.proc.terminate()
        cls.proc.wait(timeout=10)
        cls.tmp.cleanup()
    def request(self,path,data=None,token=True,headers=None):
        hdr=headers or {}
        if data is not None:
            hdr['Content-Type']='application/json'
            if token:hdr['X-Session-Token']=self.session['token']
        return urllib.request.urlopen(urllib.request.Request(self.base+path,data=json.dumps(data).encode() if data is not None else None,headers=hdr),timeout=5)
    def test_page_and_local_assets(self):
        for path in ('/','/app.js','/style.css'):
            with self.request(path) as response:
                self.assertEqual(response.status,200)
                self.assertIn("default-src 'self'",response.headers['Content-Security-Policy'])
        with self.request('/timezone-countries.json') as response:
            self.assertEqual(response.status,200)
            self.assertIn('application/json',response.headers['Content-Type'])
            self.assertEqual(len(json.load(response)['features']),5)
        with self.request('/') as response:
            page=response.read().decode('utf-8')
            self.assertIn('id="helpPanel"',page)
            self.assertIn('id="videoWorkflow"',page)
            self.assertIn('id="videoRunForm"',page)
            self.assertIn('id="videoEventList"',page)
            self.assertIn('id="videoPlayer"',page)
            self.assertIn('id="videoClipExports"',page)
            self.assertIn('id="hardwareReport"',page)
            self.assertIn('id="objectSearchQuery"',page)
            self.assertIn('id="confirmObjectSearch"',page)
            self.assertIn('id="proposeViews"',page)
            self.assertIn('id="viewProposals"',page)
            with self.request('/api/hardware') as response:
                hardware=json.load(response)
            self.assertIn('cpu',hardware)
            self.assertEqual(hardware['gpu_execution'],'not verified by provider listing')
            self.assertEqual(hardware['storage_performance']['state'],'measured')
            self.assertGreater(hardware['storage_performance']['write_mib_s'],0)
            for key in ('keyboardHelp','helpImport','helpSchedule','helpViews','helpReview','helpScope'):
                self.assertIn(f'data-t="{key}"',page)
    def test_write_requires_session_token(self):
        with self.assertRaises(urllib.error.HTTPError) as exc:self.request('/api/missions',{'name':'Test'},token=False)
        self.assertEqual(exc.exception.code,403)
    def test_cross_origin_write_rejected(self):
        with self.assertRaises(urllib.error.HTTPError) as exc:self.request('/api/missions',{'name':'Test'},headers={'Origin':'http://external.invalid'})
        self.assertEqual(exc.exception.code,403)
    def test_rebinding_host_rejected(self):
        with self.assertRaises(urllib.error.HTTPError) as exc:self.request('/api/session',headers={'Host':'external.invalid'})
        self.assertEqual(exc.exception.code,403)
    def test_create_empty_mission_and_export(self):
        with self.request('/api/missions',{'name':'בדיקת משימה','country':'Israel','zone':'Asia/Jerusalem'}) as response:
            mission=json.load(response)['id']
        with self.request('/api/images?mission='+mission) as response:self.assertEqual(json.load(response),[])
        with self.request('/api/export?mission='+mission) as response:
            rows=list(csv.reader(io.StringIO(response.read().decode('utf-8-sig'))))
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0][0],'image')

    def test_video_source_and_run_status_api(self):
        with tempfile.TemporaryDirectory() as external:
            source_root=Path(external)/'video-api-source'
            source_root.mkdir()
            (source_root/'intersection.mkv').write_bytes(b'video-fixture')
            with self.request('/api/missions',{'name':'Video API fixture','country':'Israel','zone':'Asia/Jerusalem',
                'input_type':'videos','analysis_mode':'video_traffic','source_root':str(source_root)}) as response:
                mission=json.load(response)['id']
            with self.request('/api/video/sources',{'mission':mission,'relative_path':'intersection.mkv',
                'recording_start_utc':'2026-05-17T16:04:00+00:00','origin_provenance':'test metadata'}) as response:
                source=json.load(response)
            self.assertEqual(source['recording_start_utc'],'2026-05-17T16:04:00+00:00')
            media_path=f"/api/video/media?mission={mission}&source={source['id']}"
            with self.request(media_path,headers={'Range':'bytes=1-4'}) as response:
                self.assertEqual(response.status,206)
                self.assertEqual(response.headers['Content-Range'],'bytes 1-4/13')
                self.assertEqual(response.headers['Accept-Ranges'],'bytes')
                self.assertEqual(response.read(),b'ideo')
            with self.assertRaises(urllib.error.HTTPError) as exc:
                self.request(media_path,headers={'Range':'bytes=99-'})
            self.assertEqual(exc.exception.code,416)
            with self.request('/api/video/sources?mission='+mission) as response:
                self.assertEqual(json.load(response)[0]['id'],source['id'])
            with self.request('/api/video/runs?mission='+mission) as response:
                self.assertEqual(json.load(response),[])
            with self.request('/api/video/clips?mission='+mission) as response:
                self.assertEqual(json.load(response),[])
            with self.assertRaises(urllib.error.HTTPError) as exc:
                self.request('/api/video/clips',{'mission':mission,'event_id':'missing',
                    'pre_seconds':1,'post_seconds':1,'operator':'Operator'})
            self.assertEqual(exc.exception.code,400)
    def test_invalid_body_returns_error(self):
        with self.assertRaises(urllib.error.HTTPError) as exc:self.request('/api/missions',{'name':'x','country':'Israel','zone':'bad/zone'})
        self.assertEqual(exc.exception.code,400)

    def test_offline_object_search_interpret_confirm_and_execute(self):
        with self.request('/api/missions',{'name':'Search fixture','country':'Israel','zone':'Asia/Jerusalem'}) as response:
            mission=json.load(response)['id']
        with self.request('/api/object-search/interpret',{'mission':mission,'query':'Find red pickup trucks'}) as response:
            spec=json.load(response)['spec']
        self.assertEqual(spec['categories'],['truck'])
        self.assertTrue(spec['operator_confirmation_required'])
        with self.assertRaises(urllib.error.HTTPError) as exc:
            self.request('/api/object-search/execute',{'mission':mission,'spec':spec,'confirmed':False})
        self.assertEqual(exc.exception.code,400)
        with self.request('/api/object-search/execute',{'mission':mission,'spec':spec,'confirmed':True}) as response:
            result=json.load(response)
        self.assertEqual(result['matched_images'],0)
        self.assertEqual(result['matched_objects'],0)

    def test_view_proposals_route_is_local_and_still_image_scoped(self):
        with self.request('/api/missions',{'name':'View proposal fixture','country':'Israel','zone':'Asia/Jerusalem'}) as response:
            mission=json.load(response)['id']
        with self.request('/api/views/proposals',{'mission':mission}) as response:
            result=json.load(response)
        self.assertEqual(result['available_images'],0)
        self.assertEqual(result['groups'],[])
        with self.request('/api/missions',{'name':'Video view fixture','country':'Israel','zone':'Asia/Jerusalem',
                'input_type':'videos','analysis_mode':'video_traffic'}) as response:
            video_mission=json.load(response)['id']
        with self.assertRaises(urllib.error.HTTPError) as exc:
            self.request('/api/views/proposals',{'mission':video_mission})
        self.assertEqual(exc.exception.code,400)

    def test_scan_window_api_round_trip_and_validation(self):
        with self.request('/api/missions',{'name':'Schedule fixture','country':'Latvia','zone':'Europe/Riga'}) as response:
            mission=json.load(response)['id']
        window={'weekdays':[6],'start_date':'2026-01-01','end_date':'2026-12-31',
            'start_time':'17:00','end_time':'05:00'}
        with self.request('/api/scan-windows',{'mission':mission,'window':window}) as response:
            created=json.load(response)
        with self.request('/api/scan-windows?mission='+mission) as response:
            rows=json.load(response)
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]['id'],created['id'])
        self.assertEqual(rows[0]['weekdays'],[6])
        with self.assertRaises(urllib.error.HTTPError) as exc:
            self.request('/api/scan-windows',{'mission':mission,'window':{**window,'end_time':'17:00'}})
        self.assertEqual(exc.exception.code,400)
        with self.request('/api/scan-windows',{'mission':mission,'action':'delete','id':created['id']}) as response:
            self.assertTrue(json.load(response)['deleted'])
        with self.request('/api/scan-windows?mission='+mission) as response:
            self.assertEqual(json.load(response),[])

    def test_camera_view_api_records_assignment_history(self):
        from road_traffic.store import Library
        with self.request('/api/missions',{'name':'View fixture','country':'Israel','zone':'Asia/Jerusalem'}) as response:
            mission=json.load(response)['id']
        lib=Library(self.tmp.name)
        with tempfile.TemporaryDirectory() as source:
            write_test_image(Path(source)/'17052026-1604-view.jpg')
            lib.import_folder(mission,source,'camera')
        image=lib.images(mission)[0]['id']
        with self.request('/api/views',{'mission':mission,'action':'create','name':'Gate A','operator':'Operator'}) as response:
            view=json.load(response)
        with self.request('/api/views',{'mission':mission,'action':'assign','image_id':image,'view_id':view['id'],'operator':'Operator'}) as response:
            self.assertEqual(json.load(response)['revision'],1)
        with self.request('/api/view-history?mission='+mission+'&image='+image) as response:
            history=json.load(response)
        self.assertEqual(history[0]['view_name'],'Gate A')
        with self.request('/api/views?mission='+mission) as response:
            self.assertEqual(json.load(response)[0]['image_count'],1)

    def test_batch_camera_view_api_assignment(self):
        from road_traffic.store import Library
        with self.request('/api/missions',{'name':'Batch view fixture','country':'Israel','zone':'Asia/Jerusalem'}) as response:
            mission=json.load(response)['id']
        lib=Library(self.tmp.name)
        with tempfile.TemporaryDirectory() as source:
            write_test_image(Path(source)/'17052026-1604-view-a.jpg')
            write_test_image(Path(source)/'17052026-1605-view-b.jpg',(55,85,125))
            lib.import_folder(mission,source,'camera')
        images=lib.images(mission)
        with self.request('/api/views',{'mission':mission,'action':'create','name':'Approach A','operator':'Operator'}) as response:
            view=json.load(response)
        with self.request('/api/views',{'mission':mission,'action':'assign_many','image_ids':[row['id'] for row in images],
            'view_id':view['id'],'operator':'Operator'}) as response:
            self.assertEqual(json.load(response)['changed'],2)
        with self.request('/api/views?mission='+mission) as response:
            self.assertEqual(json.load(response)[0]['image_count'],2)
        for row in images:
            with self.request('/api/view-history?mission='+mission+'&image='+row['id']) as response:
                self.assertEqual(json.load(response)[0]['view_name'],'Approach A')

    def test_view_region_api_round_trip_and_append_only_history(self):
        from road_traffic.store import Library
        with self.request('/api/missions',{'name':'Region fixture','country':'Israel','zone':'Asia/Jerusalem'}) as response:
            mission=json.load(response)['id']
        lib=Library(self.tmp.name)
        with tempfile.TemporaryDirectory() as source:
            write_test_image(Path(source)/'17052026-1604-region.jpg')
            lib.import_folder(mission,source,'camera')
        image=lib.images(mission)[0]['id']
        with self.request('/api/views',{'mission':mission,'action':'create','name':'Gate B','operator':'Operator'}) as response:
            view=json.load(response)
        with self.request('/api/views',{'mission':mission,'action':'assign','image_id':image,'view_id':view['id'],'operator':'Operator'}) as response:
            self.assertEqual(json.load(response)['revision'],1)
        polygon=[[.1,.1],[.9,.1],[.9,.9],[.1,.9]]
        with self.request('/api/view-regions',{'mission':mission,'action':'create','view_id':view['id'],
            'kind':'include','polygon':polygon,'operator':'Operator'}) as response:
            region=json.load(response)
        with self.request('/api/view-regions?mission='+mission+'&view_id='+view['id']) as response:
            self.assertEqual(json.load(response)[0]['region_key'],region['region_key'])
        with self.request('/api/view-regions',{'mission':mission,'action':'update','region_id':region['region_key'],
            'kind':'exclude','polygon':polygon,'operator':'Operator'}) as response:
            self.assertEqual(json.load(response)['revision'],2)
        with self.request('/api/view-region-history?mission='+mission+'&view_id='+view['id']) as response:
            history=json.load(response)
        self.assertEqual([entry['action'] for entry in history],['upsert','upsert'])
        with self.request('/api/view-regions',{'mission':mission,'action':'delete','region_id':region['region_key'],
            'operator':'Operator'}) as response:
            self.assertTrue(json.load(response)['deleted'])
        with self.request('/api/view-regions?mission='+mission+'&view_id='+view['id']) as response:
            self.assertEqual(json.load(response),[])

    def test_selected_report_and_rejected_export_counts(self):
        from road_traffic.store import Library
        from road_traffic.inference_store import create_run
        with self.request('/api/missions',{'name':'Export fixture','country':'Israel','zone':'Asia/Jerusalem'}) as response:
            mission=json.load(response)['id']
        lib=Library(self.tmp.name)
        with tempfile.TemporaryDirectory() as source:
            write_test_image(Path(source)/'17052026-1604-fixture.jpg')
            lib.import_folder(mission,source,'camera')
        image=lib.images(mission)[0]['id'];run=create_run(lib,mission,{'adapter':'explicit-test-fixture'})
        original=[{'kind':'candidate','label':'car','box':[.1,.1,.1,.1],'origin':'model'},
                  {'kind':'candidate','label':'truck','box':[.5,.5,.1,.1],'origin':'model'}]
        with lib.connect(mission) as conn:
            conn.execute('INSERT INTO predictions VALUES (?,?,?,?,?,?)',(run,image,json.dumps(original),'{}',.1,None))
            conn.execute("UPDATE inference_runs SET status='completed' WHERE id=?",(run,))
        review=[{**original[0],'kind':'vehicle'},{**original[1],'kind':'vehicle','rejected':True}]
        with self.request('/api/review',{'mission':mission,'image':image,'revision':0,'annotations':review,
            'operator':'=formula-test','role':'none','status':'partial','prediction_run':run}) as response:
            self.assertEqual(json.load(response)['revision'],1)
        with self.request('/api/export?mission='+mission+'&run='+run) as response:
            rows=list(csv.DictReader(io.StringIO(response.read().decode('utf-8-sig'))))
        self.assertEqual(rows[0]['confirmed_vehicles'],'1')
        self.assertEqual(rows[0]['prediction_run'],run)
        self.assertEqual(rows[0]['review_prediction_run'],run)
        self.assertEqual(rows[0]['operator'],"'=formula-test")
        self.assertEqual(json.loads(rows[0]['original_predictions']),original)
        with self.request('/api/inference/report?mission='+mission+'&run='+run) as response:
            report=json.load(response)
        self.assertEqual(report['run']['id'],run)
        self.assertEqual(report['images'][0]['predictions'],original)

    def test_inference_report_rejects_unknown_run(self):
        with self.request('/api/missions',{'name':'Report fixture','country':'Israel','zone':'Asia/Jerusalem'}) as response:
            mission=json.load(response)['id']
        with self.assertRaises(urllib.error.HTTPError) as exc:self.request('/api/inference/report?mission='+mission+'&run=missing')
        self.assertEqual(exc.exception.code,400)

    def test_roi_rerun_route_rejects_unknown_run(self):
        with self.request('/api/missions',{'name':'ROI route fixture','country':'Israel','zone':'Asia/Jerusalem'}) as response:
            mission=json.load(response)['id']
        with self.assertRaises(urllib.error.HTTPError) as exc:
            self.request('/api/inference/roi-rerun',{'mission':mission,'run':'missing','operator':'Operator'})
        self.assertEqual(exc.exception.code,400)
        self.assertIn('Inference run not found',json.loads(exc.exception.read())['error'])

    def test_mission_settings_are_saved_and_unsupported_actions_are_blocked(self):
        with tempfile.TemporaryDirectory() as external:
            source=Path(external)/'configured-source';source.mkdir()
            output=Path(external)/'configured-output';output.mkdir()
            data={'name':'Configured video mission','country':'Latvia','zone':'Europe/Riga',
                'input_type':'videos','analysis_mode':'video_traffic','location':'Riga',
                'source_root':str(source),'include_subdirectories':False,
                'processing_mode':'continuous','output_root':str(output),'export_formats':['csv','xlsx']}
            with self.request('/api/missions',data) as response:mission=json.load(response)['id']
            with self.request('/api/session') as response:
                saved=next(m for m in json.load(response)['missions'] if m['id']==mission)
            self.assertEqual(saved['input_type'],'videos')
            self.assertEqual(saved['analysis_mode'],'video_traffic')
            self.assertFalse(saved['include_subdirectories'])
            self.assertEqual(saved['export_formats'],['csv','xlsx'])
            with self.assertRaises(urllib.error.HTTPError) as exc:
                self.request('/api/inference/start',{'mission':mission})
            self.assertEqual(exc.exception.code,400)
            self.assertIn('still-image General Traffic',exc.exception.read().decode())
            with self.request('/api/export?mission='+mission) as response:
                self.assertIn('text/csv',response.headers['Content-Type'])

    def test_continuous_monitor_can_be_paused_and_resumed_over_http(self):
        with tempfile.TemporaryDirectory() as external:
            source=Path(external)/'source';source.mkdir()
            with self.request('/api/missions',{'name':'Monitor controls','country':'Israel','zone':'Asia/Jerusalem',
                'source_root':str(source),'processing_mode':'continuous'}) as response:
                mission=json.load(response)['id']
            with self.request('/api/monitoring?mission='+mission) as response:
                self.assertTrue(json.load(response)['enabled'])
            with self.request('/api/monitoring',{'mission':mission,'action':'pause'}) as response:
                self.assertFalse(json.load(response)['enabled'])
            with self.request('/api/monitoring?mission='+mission) as response:
                self.assertEqual(json.load(response)['status'],'paused')
            with self.request('/api/monitoring',{'mission':mission,'action':'resume'}) as response:
                self.assertTrue(json.load(response)['enabled'])
            from road_traffic.store import Library
            library=Library(Path(self.tmp.name))
            with library.connect(mission) as conn:
                conn.execute("INSERT INTO monitor_events(source_run_id,kind,details,created) VALUES ('fixture','new_data','Imported fixture batch','now')")
            with self.request('/api/monitoring?mission='+mission) as response:
                alerts=json.load(response)['alerts']
                self.assertEqual(alerts[0]['details'],'Imported fixture batch')
            with self.request('/api/monitoring',{'mission':mission,'action':'acknowledge_alerts','ids':[alerts[0]['id']]}) as response:
                self.assertEqual(json.load(response)['alerts'],[])

    def test_csv_export_writes_traceable_non_overwriting_snapshot(self):
        with tempfile.TemporaryDirectory() as external:
            source=Path(external)/'source';source.mkdir()
            output=Path(external)/'output';output.mkdir()
            image=source/'17052026-1604-0007.jpg';write_test_image(image)
            data={'name':'Saved CSV export','country':'Israel','zone':'Asia/Jerusalem',
                'input_type':'images','analysis_mode':'general_traffic','location':'Jerusalem',
                'source_root':str(source),'include_subdirectories':True,
                'processing_mode':'once','output_root':str(output),'export_formats':['csv']}
            with self.request('/api/missions',data) as response:mission=json.load(response)['id']
            with self.request('/api/import',{'mission':mission,'path':str(source),'preset':'camera'}) as response:
                self.assertEqual(json.load(response)['images'],1)
            with self.request('/api/export',{'mission':mission,'run':None}) as response:first=json.load(response)
            with self.request('/api/export',{'mission':mission,'run':None}) as response:second=json.load(response)
            self.assertNotEqual(first['path'],second['path'])
            self.assertEqual(first['status'],'completed')
            self.assertEqual(first['row_count'],1)
            self.assertTrue((output/Path(first['path']).name).is_file())
            with Path(first['path']).open(encoding='utf-8-sig',newline='') as stream:
                row=next(csv.DictReader(stream))
            self.assertEqual(row['mission_id'],mission)
            self.assertEqual(row['filename'],image.name)
            self.assertEqual(row['source_sha256'],hashlib.sha256(image.read_bytes()).hexdigest())
            with self.request('/api/exports?mission='+mission) as response:history=json.load(response)
            self.assertEqual(len(history),2)
            self.assertEqual(history[0]['status'],'completed')
            self.assertEqual(history[0]['snapshot_sha256'],first['snapshot_sha256'])

    def test_configured_xlsx_and_csv_exports_preserve_unicode_without_formula_cells(self):
        with tempfile.TemporaryDirectory() as external:
            source=Path(external)/'source';source.mkdir()
            output=Path(external)/'output';output.mkdir()
            image=source/'17052026-1604-0007.jpg';write_test_image(image)
            data={'name':'Saved XLSX export','country':'Israel','zone':'Asia/Jerusalem',
                'source_root':str(source),'include_subdirectories':True,'processing_mode':'once',
                'output_root':str(output),'export_formats':['xlsx','csv']}
            with self.request('/api/missions',data) as response:mission=json.load(response)['id']
            with self.request('/api/import',{'mission':mission,'path':str(source),'preset':'camera'}):pass
            with self.request('/api/images?mission='+mission) as response:image_id=json.load(response)[0]['id']
            with self.request('/api/review',{'mission':mission,'image':image_id,'revision':0,'annotations':[],
                'operator':'=מפעיל','role':'none','status':'partial'}):pass
            with self.request('/api/export',{'mission':mission,'run':None}) as response:result=json.load(response)
            files={entry['format']:entry for entry in result['files']}
            self.assertEqual(set(files),{'csv','xlsx'})
            self.assertEqual(files['xlsx']['row_count'],1)
            with zipfile.ZipFile(files['xlsx']['path']) as book:
                self.assertIn('xl/worksheets/sheet1.xml',book.namelist())
                sheet=ET.fromstring(book.read('xl/worksheets/sheet1.xml'))
            ns={'x':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
            cells={cell.attrib['r']:cell for cell in sheet.findall('.//x:sheetData/x:row[@r="2"]/x:c',ns)}
            self.assertEqual(cells['F2'].findtext('x:is/x:t',namespaces=ns),image.name)
            self.assertEqual(cells['T2'].findtext('x:is/x:t',namespaces=ns),'=מפעיל')
            self.assertIsNone(sheet.find('.//x:f',ns))
