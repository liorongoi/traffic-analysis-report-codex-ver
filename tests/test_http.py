import csv
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
    def test_invalid_body_returns_error(self):
        with self.assertRaises(urllib.error.HTTPError) as exc:self.request('/api/missions',{'name':'x','country':'Israel','zone':'bad/zone'})
        self.assertEqual(exc.exception.code,400)

    def test_selected_report_and_rejected_export_counts(self):
        from road_traffic.store import Library
        from road_traffic.inference_store import create_run
        with self.request('/api/missions',{'name':'Export fixture','country':'Israel','zone':'Asia/Jerusalem'}) as response:
            mission=json.load(response)['id']
        lib=Library(self.tmp.name)
        with tempfile.TemporaryDirectory() as source:
            (Path(source)/'17052026-1604-fixture.jpg').write_bytes(b'explicit-http-ledger-fixture')
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
            with self.assertRaises(urllib.error.HTTPError) as exc:
                self.request('/api/export?mission='+mission)
            self.assertEqual(exc.exception.code,400)
            self.assertIn('XLSX output',exc.exception.read().decode())
