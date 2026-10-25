"""Loopback development host with an opt-in TLS-protected, single-user LAN mode."""
import csv
import io
import json
import mimetypes
import re
import secrets
import ssl
import threading
import time
import webbrowser
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs
from .store import Library
from . import video_store, video_clips, video_event_exports
from . import source_reconciliation


def serve(workspace, web, port=8765, open_browser=False, *, host='127.0.0.1',
        public_host=None, tls_cert=None, tls_key=None, access_code=None):
    loopback=host in ('127.0.0.1','localhost','::1')
    if not loopback:
        if not public_host or not re.fullmatch(r'[A-Za-z0-9.-]+',public_host):
            raise ValueError('LAN mode requires --public-host with the exact DNS name or IPv4 address used by browsers')
        if not tls_cert or not tls_key or not Path(tls_cert).is_file() or not Path(tls_key).is_file():
            raise ValueError('LAN mode requires an existing TLS certificate and private key')
        if not isinstance(access_code,str) or len(access_code)<16:
            raise ValueError('Set TRAFFIC_ACCESS_CODE to a secret of at least 16 characters before enabling LAN mode')
    elif any((public_host,tls_cert,tls_key,access_code)):
        raise ValueError('LAN host, TLS and access-code settings can only be used with a non-loopback bind address')
    library = Library(workspace)
    library.reconcile_exports()
    from .monitor import SourceMonitor
    from .inference import InferenceService
    inference=InferenceService(library,Path(web).parent/'model-packs'/'yolox-s')
    from .video_inference import VideoInferenceService
    video_inference=VideoInferenceService(library,Path(web).parent/'model-packs'/'yolox-s',inference,
        plate_pack=Path(web).parent/'model-packs'/'license-plate-yolox')
    monitor = SourceMonitor(library,on_batch=inference.start_continuous_batch,
        on_video=video_inference.start_continuous_source)
    monitor.start()
    video_clips.reconcile_exports(library)
    video_event_exports.reconcile_exports(library)
    token = secrets.token_urlsafe(32)
    session_state={'active':None,'failed':{},'lock':threading.Lock()}
    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, data, content='application/json; charset=utf-8'):
            if not isinstance(data, bytes):
                data = json.dumps(data, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', content)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', "default-src 'self'; img-src 'self'; connect-src 'self'; style-src 'self'; script-src 'self'; frame-ancestors 'none'; base-uri 'none'")
            self.end_headers()
            self.wfile.write(data)

        def valid_host(self):
            expected=(f'{public_host}:{self.server.server_port}' if not loopback else None)
            return self.headers.get('Host')==expected if expected else self.headers.get('Host') in (
                f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}')

        def valid_origin(self):
            scheme='https' if not loopback else 'http'
            allowed=(f'{scheme}://{public_host}:{self.server.server_port}' if not loopback else None)
            origins=(allowed,) if allowed else (f'http://127.0.0.1:{self.server.server_port}',
                f'http://localhost:{self.server.server_port}')
            return self.headers.get('Origin') in origins

        def session_id(self):
            if loopback:return None
            try:
                cookie=SimpleCookie();cookie.load(self.headers.get('Cookie',''))
                value=cookie.get('traffic_session')
                candidate=value.value if value else None
            except Exception:
                candidate=None
            with session_state['lock']:
                active=session_state['active']
                if not active or candidate!=active['id'] or time.time()-active['created_at']>12*60*60:
                    if active and time.time()-active['created_at']>12*60*60:session_state['active']=None
                    return None
            return candidate

        def login_page(self):
            preferences=[]
            for index,item in enumerate(self.headers.get('Accept-Language','').split(',')):
                language,*parameters=item.strip().lower().split(';')
                quality=1.0
                for parameter in parameters:
                    if parameter.strip().startswith('q='):
                        try:quality=float(parameter.strip()[2:])
                        except ValueError:quality=0.0
                preferences.append((-quality,index,language))
            hebrew=bool(preferences and sorted(preferences)[0][2].startswith('he'))
            if hebrew:
                body='''<!doctype html><html lang="he" dir="rtl"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ניתוח תנועה · כניסה</title><body><main><h1>ניתוח תנועה</h1><p>הזינו את קוד הגישה שקיבלתם ממפעיל העמדה.</p><form method="post" action="/api/login"><label for="code">קוד גישה</label><input id="code" name="code" type="password" minlength="16" autocomplete="current-password" required><button type="submit">כניסה</button></form><p id="error" role="alert"></p><script src="/login.js"></script></main></body></html>'''.encode('utf-8')
            else:
                body=b'''<!doctype html><html lang="en" dir="ltr"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Traffic Analysis Sign In</title><body><main><h1>Traffic Analysis</h1><p>Enter the access code provided by the workstation operator.</p><form method="post" action="/api/login"><label for="code">Access code</label><input id="code" name="code" type="password" minlength="16" autocomplete="current-password" required><button type="submit">Sign in</button></form><p id="error" role="alert"></p><script src="/login.js"></script></main></body></html>'''
            self.reply(200,body,'text/html; charset=utf-8')

        def login(self):
            if not self.valid_origin() or self.headers.get('Content-Type','').split(';',1)[0].strip().lower()!='application/x-www-form-urlencoded':
                return self.reply(403,{'error':'Sign-in requires the configured HTTPS origin'})
            remote=self.client_address[0]
            now=time.time()
            with session_state['lock']:
                attempts=session_state['failed'].get(remote,{'count':0,'until':0})
                if attempts['until']>now:
                    return self.reply(429,{'error':'Too many sign-in attempts; wait before retrying'})
            try:length=int(self.headers.get('Content-Length','0'))
            except ValueError:length=0
            if not 0<length<=4096:return self.reply(413,{'error':'Sign-in request is invalid'})
            fields=parse_qs(self.rfile.read(length).decode('utf-8','replace'),keep_blank_values=True)
            supplied=fields.get('code',[''])[0]
            if not secrets.compare_digest(supplied,access_code):
                with session_state['lock']:
                    attempt=session_state['failed'].setdefault(remote,{'count':0,'until':0})
                    attempt['count']+=1
                    if attempt['count']>=5:attempt.update(count=0,until=now+60)
                return self.reply(401,{'error':'Access code is incorrect'})
            session=secrets.token_urlsafe(32)
            with session_state['lock']:
                session_state['active']={'id':session,'created_at':now}
                session_state['failed'].pop(remote,None)
            self.send_response(303)
            self.send_header('Location','/')
            self.send_header('Set-Cookie',f'traffic_session={session}; Path=/; Max-Age=43200; HttpOnly; Secure; SameSite=Strict')
            self.send_header('Cache-Control','no-store')
            self.send_header('Content-Length','0')
            self.end_headers()

        def login_script(self):
            body=b"document.querySelector('form').addEventListener('submit',async e=>{e.preventDefault();const button=e.currentTarget.querySelector('button');button.disabled=true;try{const response=await fetch('/api/login',{method:'POST',headers:{'Content-Type':'application/x-www-form-urlencoded'},body:new URLSearchParams(new FormData(e.currentTarget))});if(response.redirected||response.ok)location.assign('/');else{const result=await response.json();document.querySelector('#error').textContent=result.error||'Sign in failed';}}catch(error){document.querySelector('#error').textContent='Sign in failed';}finally{button.disabled=false;}});"
            self.reply(200,body,'text/javascript; charset=utf-8')

        def stream_video(self, query):
            path = video_store.source_path(library, query['mission'][0], query['source'][0])
            source = video_store.source_detail(library, query['mission'][0], query['source'][0])
            size = path.stat().st_size
            etag = f"\"{source['sha256']}\""
            header = self.headers.get('Range')
            if_range = self.headers.get('If-Range')
            if if_range and if_range != etag:
                header = None
            start, end, status = 0, max(0, size - 1), 200
            if header:
                match = re.fullmatch(r'bytes=(\d*)-(\d*)', header.strip())
                if not match or size == 0:
                    match = None
                if match:
                    left, right = match.groups()
                    if not left:
                        suffix = int(right or 0)
                        start = max(0, size - suffix) if suffix else size
                        end = size - 1
                    else:
                        start = int(left)
                        end = min(int(right), size - 1) if right else size - 1
                    if start >= size or start > end:
                        match = None
                    else:
                        status = 206
                if not match:
                    self.send_response(416)
                    self.send_header('Content-Range', f'bytes */{size}')
                    self.send_header('Content-Length', '0')
                    self.send_header('Accept-Ranges', 'bytes')
                    self.end_headers()
                    return
            length = max(0, end - start + 1)
            self.send_response(status)
            self.send_header('Content-Type', mimetypes.guess_type(path.name)[0] or 'application/octet-stream')
            self.send_header('Content-Length', str(length))
            self.send_header('Accept-Ranges', 'bytes')
            self.send_header('ETag', etag)
            self.send_header('Cache-Control', 'private, no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            if status == 206:
                self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
            self.end_headers()
            if not length:
                return
            with path.open('rb') as stream:
                stream.seek(start)
                remaining = length
                while remaining:
                    block = stream.read(min(1024 * 1024, remaining))
                    if not block:
                        break
                    self.wfile.write(block)
                    remaining -= len(block)

        def download_clip(self, query):
            path = video_clips.export_file(library, query['mission'][0], query['export'][0])
            size = path.stat().st_size
            self.send_response(200)
            self.send_header('Content-Type', 'video/mp4')
            self.send_header('Content-Length', str(size))
            self.send_header('Content-Disposition', f'attachment; filename="{path.name}"')
            self.send_header('Cache-Control', 'private, no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    self.wfile.write(block)

        def download_event_table(self, query):
            path = video_event_exports.export_file(library, query['mission'][0], query['export'][0])
            size = path.stat().st_size
            content = ('text/csv; charset=utf-8' if path.suffix.lower() == '.csv' else
                'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
            self.send_response(200)
            self.send_header('Content-Type', content)
            self.send_header('Content-Length', str(size))
            self.send_header('Content-Disposition', f'attachment; filename="{path.name}"')
            self.send_header('Cache-Control', 'private, no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    self.wfile.write(block)

        def do_GET(self):
            if not self.valid_host():
                return self.reply(403, {'error': 'Configured host required'})
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            if not loopback:
                if parsed.path in ('/', '/login') and not self.session_id():
                    return self.login_page()
                if parsed.path == '/login.js':
                    return self.login_script()
                if not self.session_id():
                    return self.reply(401, {'error':'Sign in to access this workstation'})
            try:
                if parsed.path == '/api/session':
                    return self.reply(200, {'token': token, 'missions': library.missions(),
                        'mode': 'loopback-development' if loopback else 'authenticated-lan'})
                if parsed.path == '/api/hardware':
                    from .hardware import inventory
                    return self.reply(200, inventory(library.root))
                if parsed.path == '/api/learning/capabilities':
                    from .learning_capabilities import inventory
                    return self.reply(200, inventory())
                if parsed.path == '/api/timezones':
                    from zoneinfo import available_timezones
                    return self.reply(200, sorted(available_timezones()))
                if parsed.path == '/api/images':
                    return self.reply(200, library.images(query['mission'][0],query.get('run',[None])[0]))
                if parsed.path == '/api/plates':
                    from .plate_review import list_crops
                    return self.reply(200, list_crops(library,query['mission'][0],query.get('image',[None])[0]))
                if parsed.path == '/api/plate-crop':
                    from .plate_review import crop_file
                    path=crop_file(library,query['mission'][0],query['crop'][0])
                    return self.reply(200,path.read_bytes(),'image/jpeg')
                if parsed.path == '/api/plate-context':
                    from .plate_review import context_file
                    path=context_file(library,query['mission'][0],query['crop'][0])
                    return self.reply(200,path.read_bytes(),'image/jpeg')
                if parsed.path == '/api/night-learning':
                    from .learning import inventory,candidates,evaluation_status,governance_status
                    from .learning_capabilities import inventory as capability_inventory
                    m=query['mission'][0]
                    return self.reply(200,{'inventory':inventory(library,m),'candidates':candidates(library,m),
                        'evaluation':evaluation_status(library,m),'governance':governance_status(library,m),
                        'capabilities':capability_inventory()})
                if parsed.path == '/api/inference':
                    return self.reply(200,inference.status(query['mission'][0]))
                if parsed.path == '/api/video/media':
                    return self.stream_video(query)
                if parsed.path == '/api/video/frame-overlay':
                    return self.reply(200,video_store.frame_overlay(library,
                        query['mission'][0],query['run'][0],float(query['seconds'][0])))
                if parsed.path == '/api/video/clips':
                    return self.reply(200,video_clips.exports(library,query['mission'][0]))
                if parsed.path == '/api/video/event-exports':
                    return self.reply(200,video_event_exports.exports(library,query['mission'][0],
                        query.get('run',[None])[0]))
                if parsed.path == '/api/video/event-exports/download':
                    return self.download_event_table(query)
                if parsed.path == '/api/video/clips/download':
                    return self.download_clip(query)
                if parsed.path == '/api/video/sources':
                    return self.reply(200,video_store.sources(library,query['mission'][0]))
                if parsed.path == '/api/video/runs':
                    return self.reply(200,video_inference.status(query['mission'][0]))
                if parsed.path == '/api/video/events':
                    return self.reply(200,video_store.events(library,query['mission'][0],query['run'][0],
                        limit=int(query.get('limit',['100'])[0]),offset=int(query.get('offset',['0'])[0])))
                if parsed.path == '/api/video/event-thumbnail':
                    return self.reply(200,video_store.event_thumbnail(library,query['mission'][0],
                        query['event_id'][0]),'image/jpeg')
                if parsed.path == '/api/exports':
                    return self.reply(200,library.exports(query['mission'][0]))
                if parsed.path == '/api/bulk-approval/preview':
                    from .bulk_approval import preview
                    return self.reply(200,preview(library,query['mission'][0],query.get('run',[None])[0]))
                if parsed.path == '/api/bulk-approvals':
                    from .bulk_approval import history
                    return self.reply(200,history(library,query['mission'][0]))
                if parsed.path == '/api/monitoring':
                    return self.reply(200,monitor.status(query['mission'][0]))
                if parsed.path == '/api/source-reconciliation':
                    mission=query['mission'][0]
                    return self.reply(200,{'candidates':source_reconciliation.candidates(library,mission),
                        'history':source_reconciliation.history(library,mission)})
                if parsed.path == '/api/scan-windows':
                    return self.reply(200,library.scan_windows(query['mission'][0]))
                if parsed.path == '/api/views':
                    return self.reply(200,library.views(query['mission'][0]))
                if parsed.path == '/api/view-history':
                    return self.reply(200,library.view_history(query['mission'][0],query.get('image',[None])[0]))
                if parsed.path == '/api/view-regions':
                    return self.reply(200,library.view_regions(query['mission'][0],query['view_id'][0]))
                if parsed.path == '/api/view-region-history':
                    return self.reply(200,library.view_region_history(query['mission'][0],query['view_id'][0]))
                if parsed.path == '/api/inference/report':
                    mission=query['mission'][0]
                    jobs=inference.status(mission)
                    run=query.get('run',[None])[0] or (jobs[0]['id'] if jobs else None)
                    job=next((j for j in jobs if j['id']==run),None)
                    if not job:raise ValueError('Inference run not found')
                    return self.reply(200,{'run':job,'accuracy':'Not measured; model scores and light proposals are not independent ground truth.',
                        'images':library.images(mission,run)})
                if parsed.path == '/api/image':
                    path = library.image_path(query['mission'][0], query['image'][0])
                    return self.reply(200, path.read_bytes(), mimetypes.guess_type(path.name)[0] or 'application/octet-stream')
                if parsed.path == '/api/export':
                    mission = query['mission'][0]
                    formats = library.settings(mission)['export_formats']
                    if 'csv' not in formats:
                        raise ValueError('This route does not stream XLSX output; use the configured export action to create XLSX files')
                    rows = library.images(mission,query.get('run',[None])[0])
                    stream = io.StringIO(newline='')
                    writer = csv.writer(stream)
                    writer.writerow(['image','capture_utc','review_status','revision','confirmed_vehicles','candidates','lights','annotations','prediction_run','inference_status','original_predictions','review_prediction_run','evidence_role','operator'])
                    for row in rows:
                        all_labels = json.loads(row['annotations'])
                        labels = [a for a in all_labels if not a.get('rejected')]
                        name = row['name']
                        if name.lstrip().startswith(('=', '+', '-', '@')):
                            name = "'" + name
                        payload = json.dumps(all_labels, ensure_ascii=False)
                        writer.writerow([name, row['capture_utc'], row['review_status'], row['revision'],
                            sum(x['kind']=='vehicle' for x in labels),sum(x['kind']=='candidate' for x in labels),
                            sum(x['kind']=='light' for x in labels),payload,row.get('prediction_run'),row.get('inference_status'),json.dumps(row.get('predictions',[]),ensure_ascii=False),row.get('review_prediction_run'),row.get('evidence_role'),
                            ("'"+row.get('operator','')) if row.get('operator','').lstrip().startswith(('=','+','-','@')) else row.get('operator','')])
                    return self.reply(200, ('\ufeff'+stream.getvalue()).encode('utf-8'), 'text/csv; charset=utf-8')
                files = {'/':'index.html','/app.js':'app.js','/style.css':'style.css',
                    '/timezone-countries.json':'data/timezone-countries.json'}
                if parsed.path in files:
                    path = Path(web) / files[parsed.path]
                    return self.reply(200, path.read_bytes(), mimetypes.guess_type(path.name)[0]+'; charset=utf-8')
                self.reply(404, {'error':'Not found'})
            except (ValueError, KeyError, FileNotFoundError) as exc:
                self.reply(400, {'error':str(exc)})

        def do_POST(self):
            if not self.valid_host():
                return self.reply(403, {'error':'Configured host required'})
            if not loopback and self.path == '/api/login':
                return self.login()
            if not loopback and self.path == '/api/logout':
                if not self.session_id() or not self.valid_origin():
                    return self.reply(403, {'error':'Authenticated same-origin session required'})
                with session_state['lock']:
                    session_state['active']=None
                self.send_response(303)
                self.send_header('Location','/')
                self.send_header('Set-Cookie','traffic_session=; Path=/; Max-Age=0; HttpOnly; Secure; SameSite=Strict')
                self.send_header('Content-Length','0')
                self.end_headers()
                return
            if (not loopback and not self.session_id()) or self.headers.get('X-Session-Token') != token:
                return self.reply(403, {'error':'Session token required'})
            if (self.headers.get('Origin') is not None and not self.valid_origin()) or (not loopback and not self.valid_origin()):
                return self.reply(403, {'error':'Invalid origin'})
            try:
                length = int(self.headers.get('Content-Length','0'))
                if not 0 < length <= 1024*1024:
                    return self.reply(413, {'error':'Request exceeds limit'})
                data = json.loads(self.rfile.read(length))
                if self.path == '/api/missions':
                    settings = {key: data[key] for key in (
                        'input_type', 'analysis_mode', 'location', 'source_root',
                        'include_subdirectories', 'processing_mode', 'output_root', 'export_formats',
                        'resource_profile'
                    ) if key in data}
                    result = {'id': library.create(data['name'],data['country'],data['zone'],settings)}
                elif self.path == '/api/export':
                    mission = data['mission']
                    if any(job['status'] in ('queued', 'running') for job in inference.status(mission)):
                        raise ValueError('Wait for processing to finish before exporting a stable review snapshot')
                    result = library.export_configured(mission, data.get('run'))
                elif self.path == '/api/bulk-approval':
                    from .bulk_approval import approve_and_export
                    mission=data['mission']
                    if any(job['status'] in ('queued','running') for job in inference.status(mission)):
                        raise ValueError('Wait for image processing to finish before bulk sign-off')
                    result=approve_and_export(library,mission,preview_id=data['preview_id'],
                        operator=data['operator'])
                elif self.path == '/api/import':
                    result = library.import_folder(data['mission'],data['path'],data['preset'])
                elif self.path == '/api/scan-windows':
                    if data.get('action') == 'delete':
                        result = library.delete_scan_window(data['mission'],data['id'])
                    else:
                        result = library.add_scan_window(data['mission'],data['window'])
                elif self.path == '/api/monitoring':
                    action=data.get('action')
                    if action in ('pause','resume'):
                        result=monitor.set_enabled(data['mission'],action=='resume')
                    elif action=='acknowledge_alerts':
                        result=monitor.acknowledge(data['mission'],data.get('ids'))
                    else:raise ValueError('Choose pause, resume, or acknowledge_alerts')
                elif self.path == '/api/source-reconciliation':
                    result=source_reconciliation.record(library,data['mission'],data.get('image_a_id'),
                        data.get('image_b_id'),data.get('action'),data.get('operator'))
                elif self.path == '/api/views':
                    action=data.get('action')
                    if action == 'create':
                        result=library.create_view(data['mission'],data['name'],data['operator'])
                    elif action == 'rename':
                        result=library.rename_view(data['mission'],data['id'],data['name'],data['operator'])
                    elif action == 'merge':
                        result=library.merge_views(data['mission'],data['source_id'],data['target_id'],data['operator'])
                    elif action == 'assign':
                        result=library.assign_view(data['mission'],data['image_id'],data.get('view_id'),data['operator'])
                    elif action == 'assign_many':
                        result=library.assign_views(data['mission'],data.get('image_ids'),data.get('view_id'),data['operator'])
                    else:
                        raise ValueError('Choose create, rename, merge, assign, or assign_many action')
                elif self.path == '/api/views/proposals':
                    from .view_grouping import propose
                    if library.settings(data['mission'])['input_type'] != 'images':
                        raise ValueError('Camera-view proposals require a still-image mission')
                    result = propose(library, data['mission'])
                elif self.path == '/api/view-regions':
                    action=data.get('action')
                    if action == 'create':
                        result=library.add_view_region(data['mission'],data['view_id'],data['kind'],data['polygon'],data['operator'])
                    elif action == 'update':
                        result=library.update_view_region(data['mission'],data['region_id'],data['kind'],data['polygon'],data['operator'])
                    elif action == 'delete':
                        result=library.delete_view_region(data['mission'],data['region_id'],data['operator'])
                    else:
                        raise ValueError('Choose create, update, or delete action')
                elif self.path == '/api/plates/detect':
                    from .plate_review import propose
                    result=propose(library,data['mission'],data['image_id'],Path(web).parent/'model-packs'/'license-plate-yolox',data.get('threshold',.3))
                elif self.path == '/api/plates/detect-video':
                    result=video_inference.start_plate_review(data['mission'],data['source_id'],threshold=data.get('threshold',.3))
                elif self.path == '/api/plates/review':
                    from .plate_review import review
                    result=review(library,data['mission'],data['crop_id'],data['expected_revision'],data['decision'],data['operator'])
                elif self.path == '/api/review':
                    result = library.review(data['mission'],data['image'],data['revision'],data['annotations'],data['operator'],data['role'],data['status'],data.get('prediction_run'))
                elif self.path == '/api/object-search/interpret':
                    from .object_search import interpret
                    from .inference_store import latest_run_id, search_scope_snapshot
                    library.settings(data['mission'])
                    run = data.get('run') or latest_run_id(library,data['mission'])
                    snapshot = search_scope_snapshot(library,data['mission'],run)
                    result = {'spec': interpret(data['query'], mission_id=data['mission'],
                        view_id=data.get('view_id') or None, run_id=run, scope_snapshot=snapshot)}
                elif self.path == '/api/object-search/execute':
                    from .object_search import execute
                    from .inference_store import search_scope_snapshot
                    spec=data.get('spec')
                    run=(spec.get('run_id') if isinstance(spec,dict) else None) or data.get('run') or None
                    snapshot=search_scope_snapshot(library,data['mission'],run)
                    result = execute(library, data['mission'], spec,
                        confirmed=data.get('confirmed'), run=data.get('run') or None,
                        scope_snapshot=snapshot)
                elif self.path == '/api/inference/start':
                    result=inference.start(data['mission'],data.get('threshold',.18),data.get('tiled',True),data.get('night_lights',True),data.get('night_head_id'),resource_profile=data.get('resource_profile','auto'))
                elif self.path == '/api/inference/roi-rerun':
                    result=inference.rerun_roi(data['mission'],data['run'],data['operator'])
                elif self.path == '/api/inference/resume':
                    result=inference.resume(data['mission'],data['run'])
                elif self.path == '/api/night-learning/train':
                    from .learning import train
                    with inference.lock:
                        if inference.training_active or any(p.poll() is None for p in inference.processes.values()):raise ValueError('Finish active processing before fitting the light head')
                        inference.training_active=True
                    try:
                        result=train(library,data['mission'])
                    finally:
                        with inference.lock:inference.training_active=False
                elif self.path == '/api/night-learning/evaluate-test':
                    from .learning import evaluate_test
                    with inference.lock:
                        if inference.training_active or any(p.poll() is None for p in inference.processes.values()):
                            raise ValueError('Finish active processing before evaluating the held-out test set')
                        inference.training_active=True
                    try:
                        result=evaluate_test(library,data['mission'],data.get('candidate_id'),data.get('operator'))
                    finally:
                        with inference.lock:inference.training_active=False
                elif self.path == '/api/night-learning/decision':
                    from .learning import record_model_decision
                    result=record_model_decision(library,data['mission'],data.get('action'),
                        data.get('candidate_id'),data.get('operator'),data.get('reason'))
                elif self.path == '/api/inference/stop':
                    result=inference.stop(data['mission'],data['run'])
                elif self.path == '/api/video/sources':
                    result=video_store.register_video(library,data['mission'],data['relative_path'],
                        metadata=data.get('metadata'),recording_start_utc=data.get('recording_start_utc'),
                        origin_provenance=data.get('origin_provenance'))
                elif self.path == '/api/video/runs':
                    result=video_inference.start(data['mission'],data['source_id'],data['tracker_policy'],
                        threshold=data['threshold'],tiled=data['tiled'],
                        target_labels=data.get('target_labels'),resource_profile=data.get('resource_profile','auto'))
                elif self.path == '/api/video/monitor-policy':
                    result=video_inference.save_continuous_policy(data['mission'],
                        tracker_policy=data['tracker_policy'],threshold=data['threshold'],
                        tiled=data['tiled'],target_labels=data.get('target_labels'),
                        resource_profile=data.get('resource_profile','auto'))
                    monitor.wake_event.set()
                elif self.path == '/api/video/runs/resume':
                    result=video_inference.resume(data['mission'],data['run'])
                elif self.path == '/api/video/runs/cancel':
                    result=video_inference.cancel(data['mission'],data['run'])
                elif self.path == '/api/video/events/review':
                    result=video_store.review_event(library,data['mission'],data['event_id'],
                        data['expected_revision'],data['operator'],data['action'],data.get('payload'))
                elif self.path == '/api/video/clips':
                    with inference.lock:
                        if inference.training_active or inference._has_active_worker():
                            raise ValueError('Finish local processing before exporting a video clip')
                        inference.training_active=True
                    try:
                        result=video_clips.export_event_clip(library,data['mission'],data['event_id'],
                            data['pre_seconds'],data['post_seconds'],data['operator'])
                    finally:
                        with inference.lock:
                            inference.training_active=False
                elif self.path == '/api/video/event-exports':
                    result=video_event_exports.export_run(library,data['mission'],data['run'],data['operator'])
                else:
                    return self.reply(404, {'error':'Not found'})
                self.reply(200,result)
            except (ValueError, KeyError, TypeError, OSError) as exc:
                self.reply(400,{'error':str(exc)})
            except Exception:
                self.log_error('Internal request error')
                self.reply(500,{'error':'Operation failed; inspect local diagnostics'})
    server = ThreadingHTTPServer((host,port), Handler)
    if not loopback:
        context=ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.minimum_version=ssl.TLSVersion.TLSv1_2
        context.load_cert_chain(tls_cert,tls_key)
        server.socket=context.wrap_socket(server.socket,server_side=True)
    scheme='http' if loopback else 'https'
    display_host='127.0.0.1' if loopback else public_host
    print(f'Traffic Analysis {"development" if loopback else "authenticated LAN"} host: {scheme}://{display_host}:{server.server_port}', flush=True)
    print('Local YOLOX-S inference available when the verified model pack and runtime are installed. No external assets.', flush=True)
    if open_browser:
        threading.Thread(target=webbrowser.open, args=(f'{scheme}://{display_host}:{server.server_port}',), daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        monitor.stop()
        server.server_close()
