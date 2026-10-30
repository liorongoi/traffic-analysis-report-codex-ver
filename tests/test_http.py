Warning: truncated output (original token count: 15352)
Total output lines: 953

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
import uuid
import urllib.error
import urllib.parse
import urllib.request
import zipfile
import xml.etree.ElementTree as ET
from PIL import Image
from datetime import datetime, timezone
from road_traffic.store import Library
from road_traffic import video_store


def write_test_image(path,color=(45,85,125)):
    Image.new('RGB',(8,8),color).save(path,'JPEG')


def local_plate_runtime_available():
    try:
        import cv2
        import numpy
        import onnxruntime
        return True
    except (ImportError,OSError):
        return False


def local_view_runtime_available():
    try:
        import cv2
        import numpy
        return True
    except (ImportError,OSError):
        return False


class HttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory()
        cls.server_log_path=Path(cls.tmp.name)/'server.stderr.log'
        cls.server_log=cls.server_log_path.open('wb')
        cls.server_diagnostics_requested=False
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',0))
            cls.port=sock.getsockname()[1]
        cls.base=f'http://127.0.0.1:{cls.port}'
        cls.proc=subprocess.Popen([sys.executable,'-m','road_traffic','serve','--port',str(cls.port),'--workspace',cls.tmp.name],stdout=subprocess.DEVNULL,stderr=cls.server_log)
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
        server_exit=cls.proc.poll()
        cls.proc.terminate()
        cls.proc.wait(timeout=10)
        cls.server_log.flush()
        cls.server_log.close()
        if server_exit is not None or cls.server_diagnostics_requested:
            diagnostics=cls.server_log_path.read_text(encoding='utf-8',errors='replace')
            print(f'HTTP test server exited before teardown with code {server_exit}.\n'
                f'Full HTTP test server stderr:\n{diagnostics}',file=sys.stderr)
        cls.tmp.cleanup()
    def request(self,path,data=None,token=True,headers=None,_retry=0):
        hdr=headers or {}
        if data is not None:
            if path=='/api/missions' and isinstance(data,dict):
                data.setdefault('request_id',str(uuid.uuid4()))
            hdr['Content-Type']='application/json'
            if token:hdr['X-Session-Token']=self.session['token']
        try:
            return urllib.request.urlopen(urllib.request.Request(self.base+path,data=json.dumps(data).encode() if data is not None else None,headers=hdr),timeout=30)
        except urllib.error.HTTPError as exc:
            body=exc.read()
            if exc.code >= 500:
                self.__class__.server_diagnostics_requested=True
            diagnostic=urllib.error.HTTPError(exc.url,exc.code,exc.msg,exc.hdrs,io.BytesIO(body))
            diagnostic.add_note('HTTP response body: '+body.decode('utf-8',errors='replace'))
            raise diagnostic from None
        except (OSError,urllib.error.URLError):
            self.__class__.server_diagnostics_requested=True
            if path=='/api/missions' and data is not None and _retry==0:
                return self.request(path,data,token,headers,_retry=1)
            raise
    def test_page_and_local_assets(self):
        for path in ('/','/app.js','/style.css'):
            with self.request(path) as response:
                self.assertEqual(response.status,200)
                self.assertIn("default-src 'self'",response.headers['Content-Security-Policy'])
        with self.request('/app.js') as response:
            script=response.read().decode('utf-8')
        self.assertIn('openVideoEvent',script)
        self.assertIn('video-event-preview-button',script)
        self.assertIn('source_relative_path',script)
        self.assertIn('function renderLearningReport()',script)
        self.assertIn('function renderLearningCapabilities()',script)
        self.assertIn('candidateValidation',script)
        self.assertIn('review_completion',script)
        self.assertIn('candidateTrainingEvidenceStatus',script)
        self.assertIn('plate-region detection saves image chips for operator approval or rejection and does not read plate text.',script)
        self.assertNotIn('OCR',script)
        self.assertIn("new Option(tr('promotedLearningModel'),'@promoted')",script)
        self.assertIn("night_head_id=choice==='@promoted'?learningInfo?.governance?.promoted_candidate_id",script)
        self.assertIn('function resolveTimezoneSearch()',script)
        self.assertIn("fetch('/api/timezones')",script)
        self.assertIn("fetch('/timezone-cities.json')",script)
        self.assertIn("fetch('/timezone-zones.json')",script)
        self.assertIn('function chooseTimezoneLocation(country,zone,lon,lat)',script)
        self.assertIn('timezone boundary',script.lower())
        self.assertIn('function timezoneCityMatches(query)',script)
        self.assertIn('function applyTimezoneCity(city)',script)
        self.assertNotIn('Continuous video monitoring is not implemented.',script)
        self.assertIn('Continuous video monitoring can be enabled after an explicit detector and tracker policy is saved.',script)
        self.assertIn('correct_timing',script)
        self.assertIn('videoMetadataUnavailable',script)
        self.assertIn("action:'merge'",script)
        self.assertIn('target_labels',script)
        self.assertIn('Notification.requestPermission()',script)
        self.assertIn('notifyDesktopAlerts(alerts)',script)
        self.assertIn("tr('resourceProfiles')",script)
        self.assertIn("resource_profile:$('imageResourceProfile').value",script)
        self.assertIn("resource_profile:$('videoResourceProfile').value",script)
        self.assertIn('resourceUsageText',script)
        self.assertIn('cpu_percent_host_capacity',script)

    def test_vehicle_model_pack_import_and_mission_selection_api(self):
        from road_traffic.store import Library
        import hashlib
        root=Path(self.tmp.name)
        library=Library(root)
        mission=library.create('Vehicle model API','Lebanon','Asia/Beirut')
        pack=root/'trained-model-pack';pack.mkdir()
        payload=b'fixture onnx payload'
        (pack/'model.onnx').write_bytes(payload)
        (pack/'manifest.json').write_text(json.dumps({'name':'Reviewed car model',
            'classes':['car'],'files':{'model.onnx':{'sha256':hashlib.sha256(payload).hexdigest(),
            'bytes':len(payload)}}}),encoding='utf-8')
        with self.request('/api/detector/models/import',{'path':str(pack)}) as response:
            imported=json.load(response)
        self.assertRegex(imported['id'],r'^candidate-[a-f0-9]{12}$')
        with self.request('/api/detector/models?mission='+mission) as response:
            listing=json.load(response)
        self.assertEqual(listing['selected_id'],'base')
        self.assertIn(imported['id'],{item['id'] for item in listing['models']})
        with self.request('/api/detector/models/select',{'mission':mission,
                'model_id':imported['id'],'operator':'Operator','reason':'Reviewed held-out examples'}) as response:
            selected=json.load(response)
        self.assertEqual(selected['selected_id'],imported['id'])
        with self.request('/api/detector/models?mission='+mission) as response:
            listing=json.load(response)
        self.assertEqual(listing['selected_id'],imported['id'])
        self.assertEqual(listing['history'][0]['reason'],'Reviewed held-out examples')
        with self.request('/timezone-countries.json') as response:
            self.assertEqual(response.status,200)
            self.assertIn('application/json',response.headers['Content-Type'])
            self.assertEqual(len(json.load(response)['features']),239)
        with self.request('/timezone-zones.json') as response:
            self.assertEqual(response.status,200)
            self.assertIn('application/json',response.headers['Content-Type'])
            self.assertEqual(len(json.load(response)['features']),419)
        with self.request('/timezone-cities.json') as response:
            self.assertEqual(response.status,200)
            self.assertIn('application/json',response.headers['Content-Type'])
            self.assertEqual(len(json.load(response)['cities']),34188)
        with self.request('/') as response:
            page=response.read().decode('utf-8')
            self.assertIn('RoadLoom',page)
            self.assertNotIn('lanLogout',page)
            self.assertNotIn('id="plateSearch"',page)
            self.assertIn('name="resource_profile"',page)
            self.assertIn('id="helpPanel"',page)
            self.assertIn('id="videoWorkflow"',page)
            self.assertIn('id="videoRunForm"',page)
            self.assertIn('id="videoEventList"',page)
            self.assertIn('id="exportVideoEventTable"',page)
            self.assertIn('id="videoEventTableExports"',page)
            self.assertIn('id="videoPlayer"',page)
            self.assertIn('id="videoMediaMetadata"',page)
            self.assertIn('id="timezoneSearch"',page)
            self.assertIn('id="applyTimezoneSearch"',page)
            self.assertIn('id="imageResourceProfile"',page)
            self.assertIn('id="videoResourceProfile"',page)
            self.assertIn('Video monitoring requires an explicit saved detector and tracker policy.',page)
            self.assertIn('id="videoTargetLabels"',page)
            self.assertIn('id="mergeViewButton"',page)
            self.assertIn('id="desktopAlerts"',page)
            self.assertIn('id="sourceReconciliation"',page)
            self.assertIn('id="sourceLinkCandidates"',page)
            self.assertIn('id="evaluateNightTest"',page)
            self.assertIn('id="videoClipExports"',page)
        with self.request('/roadloom.svg') as response:
            self.assertEqual(response.status,200)
            self.assertIn('<svg',response.read().decode('utf-8'))
            self.assertIn('id="hardwareReport"',page)
            self.assertIn('id="learningReport"',page)
            self.assertIn('id="plateWorkflow"',page)
            self.assertIn('id="plateCrops"',page)
            self.assertNotIn('plateOcrMetrics',page)
            self.assertNotIn('plateSearch',page)
            self.assertNotIn('OCR',page)
            self.assertIn('id="detectCurrentPlates"',page)
            self.assertIn('id="objectSearchQuery"',page)
            self.assertIn('id="confirmObjectSearch"',page)
            self.assertIn('data-t="searchRegionHelp"',page)
            self.assertIn('id="proposeViews"',page)
            self.assertIn('id="viewProposals"',page)
            self.assertIn('id="imageSearch"',page)
            self.assertIn('data-sort="capture_utc"',page)
            self.assertIn('data-sort="view_name"',page)
            self.assertIn('data-sort="inference_status"',page)
            self.assertIn('data-sort="review_status"',page)
            with self.request('/api/hardware') as response:
                hardware=json.load(response)
            self.assertIn('cpu',hardware)
            self.assertEqual(hardware['gpu_execution'],'not verified by provider listing')
            self.assertIn('conservative',hardware['resource_profiles'])
            self.assertIn('expanded',hardware['resource_profiles'])
            self.assertEqual(hardware['resource_profiles']['conservative']['workers'],1)
            self.assertEqual(hardware['storage_performance']['state'],'measured')
            self.assertGreater…9352 tokens truncated…  self.assertEqual(report['images'][0]['predictions'],original)

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
                'processing_mode':'continuous','output_root':str(output),'export_formats':['csv','xlsx'],
                'resource_profile':'expanded'}
            with self.request('/api/missions',data) as response:mission=json.load(response)['id']
            with self.request('/api/session') as response:
                saved=next(m for m in json.load(response)['missions'] if m['id']==mission)
            self.assertEqual(saved['input_type'],'videos')
            self.assertEqual(saved['analysis_mode'],'video_traffic')
            self.assertFalse(saved['include_subdirectories'])
            self.assertEqual(saved['export_formats'],['csv','xlsx'])
            self.assertEqual(saved['resource_profile'],'expanded')
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

    def test_bulk_signoff_confirms_fixed_scope_and_exports_without_individual_review(self):
        with tempfile.TemporaryDirectory() as external:
            source=Path(external)/'source';source.mkdir()
            output=Path(external)/'output';output.mkdir()
            image=source/'17052026-1604-first.jpg';write_test_image(image)
            data={'name':'Bulk scope','country':'Israel','zone':'Asia/Jerusalem',
                'input_type':'images','analysis_mode':'general_traffic','source_root':str(source),
                'output_root':str(output),'export_formats':['csv','xlsx']}
            with self.request('/api/missions',data) as response:mission=json.load(response)['id']
            with self.request('/api/import',{'mission':mission,'path':str(source),'preset':'camera'}):pass
            from road_traffic.store import Library
            from road_traffic.inference_store import create_run
            library=Library(Path(self.tmp.name));run=create_run(library,mission,{'adapter':'test-fixture'})
            with library.connect(mission) as conn:
                conn.execute("UPDATE inference_runs SET status='completed',stage='completed',done=total,finished=? WHERE id=?",
                    ('2026-10-25T00:00:00+00:00',run))
            with self.request(f'/api/bulk-approval/preview?mission={mission}&run={run}') as response:
                scope=json.load(response)
            self.assertEqual(scope['image_count'],1)
            self.assertFalse(scope['independent_ground_truth'])
            image2=source/'17052026-1605-later.jpg';write_test_image(image2)
            with self.request('/api/import',{'mission':mission,'path':str(source),'preset':'camera'}):pass
            request={'mission':mission,'preview_id':scope['preview_id'],'operator':'Operator A'}
            with self.request('/api/bulk-approval',request) as response:approved=json.load(response)
            self.assertEqual(approved['status'],'completed')
            self.assertEqual(approved['image_count'],1)
            self.assertFalse(approved['independent_ground_truth'])
            self.assertEqual(approved['individual_reviews_created'],0)
            self.assertEqual(len(approved['files']),2)
            self.assertEqual(approved['files'][0]['row_count'],1)
            self.assertEqual(approved['files'][1]['row_count'],1)
            with Path(approved['files'][0]['path']).open(encoding='utf-8-sig',newline='') as stream:
                rows=list(csv.DictReader(stream))
            self.assertEqual(len(rows),1)
            self.assertEqual(rows[0]['bulk_approval_id'],approved['id'])
            self.assertEqual(rows[0]['bulk_approval_type'],'bulk_unreviewed')
            self.assertEqual(rows[0]['review_status'],'unreviewed')
            with zipfile.ZipFile(approved['files'][1]['path']) as archive:
                sheet=archive.read('xl/worksheets/sheet1.xml').decode()
            self.assertIn('bulk_approval_id',sheet)
            self.assertIn(approved['id'],sheet)
            self.assertIn('bulk_unreviewed',sheet)
            with self.request(f'/api/bulk-approvals?mission={mission}') as response:history=json.load(response)
            self.assertEqual(history[0]['image_count'],1)
            self.assertEqual(history[0]['status'],'completed')
            self.assertTrue(all(row['review_status']=='unreviewed' for row in library.images(mission)))

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
