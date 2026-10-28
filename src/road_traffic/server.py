"""Local road-traffic review host with an opt-in TLS-protected LAN mode."""
import csv
import io
import json
import mimetypes
import re
import secrets
import ssl
import threading
import traceback
import webbrowser
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs
from .store import Library
from . import video_store, video_clips, video_event_exports
from . import source_reconciliation


def serve(workspace, web, port=8765, open_browser=False, *, host='127.0.0.1',
        public_host=None, tls_cert=None, tls_key=None):
    loopback=host in ('127.0.0.1','localhost','::1')
    if not loopback:
        if not public_host or not re.fullmatch(r'[A-Za-z0-9.-]+',public_host):
            raise ValueError('LAN mode requires --public-host with the exact DNS name or IPv4 address used by browsers')
        if not tls_cert or not tls_key or not Path(tls_cert).is_file() or not Path(tls_key).is_file():
            raise ValueError('LAN mode requires an existing TLS certificate and private key')
    elif any((public_host,tls_cert,tls_key)):
        raise ValueError('LAN host and TLS settings can only be used with a non-loopback bind address')
    library = Library(workspace)
    library.reconcile_exports()
    from .learning import recover_interrupted_tasks
    recover_interrupted_tasks(library)
    from .monitor import SourceMonitor
    from .inference import InferenceService
    inference=InferenceService(library,Path(web).parent/'model-packs'/'yolox-s')
    from .video_inference import VideoInferenceService
    video_inference=VideoInferenceService(library,Path(web).parent/'model-packs'/'yolox-s',inference,
        plate_pack=Path(web).parent/'model-packs'/'license-plate-yolox')
    monitor = SourceMonitor(library,on_batch=inference.start_continuous_batch,
        on_video=video_inference.start_continuous_source)
    video_clips.reconcile_exports(library)
    video_event_exports.reconcile_exports(library)
    token = secrets.token_urlsafe(32)
    learning_dispatch_lock=threading.Lock()
    def start_claimed_learning_task(task):
        from . import learning
        def worker():
            try:
                learning.run_task(library,task['mission_id'],task['id'],
                    yield_checkpoint=inference.training_checkpoint)
            except Exception:
                traceback.print_exc()
            finally:
                with inference.lock:
                    inference.training_active=False
                dispatch_next_learning_task()
        thread=threading.Thread(target=worker,name=f"roadloom-learning-{task['kind']}",daemon=True)
        try:
            thread.start()
        except Exception as exc:
            with library.connect(task['mission_id']) as c:
                c.execute("UPDATE learning_tasks SET status='failed',error=?,finished=? WHERE id=?",
                    (str(exc)[:1000],datetime.now(timezone.utc).isoformat(),task['id']))
            with inference.lock:
                inference.training_active=False
            raise
    def dispatch_next_learning_task():
        from .learning import task_status
        retry_after_worker=False
        task_to_start=None
        with learning_dispatch_lock:
            with inference.lock:
                if inference.training_active:return
                if inference._has_active_worker():
                    retry_after_worker=True
                else:
                    for mission in library.missions():
                        task_to_start=next((item for item in task_status(library,mission['id'])
                            if item['status']=='queued'),None)
                        if task_to_start:
                            inference.training_active=True
                            break
            if task_to_start:
                start_claimed_learning_task(task_to_start)
                return
        if retry_after_worker:
            retry=threading.Timer(1.0,dispatch_next_learning_task)
            retry.daemon=True
            retry.start()
    def schedule_learning_task(mission,kind,payload=None):
        from . import learning
        with learning_dispatch_lock:
            with inference.lock:
                if inference.training_active or inference._has_active_worker():
                    raise ValueError('Finish active processing before starting a learning task')
                task=learning.queue_task(library,mission,kind,payload)
                inference.training_active=True
            start_claimed_learning_task(task)
        return task
    dispatch_next_learning_task()
    monitor.start()
    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, data, content='application/json; charset=utf-8'):
            if not isinstance(data, bytes):
                data = json.dumps(data, ensure_ascii=False).encode()
            self.close_connection = True
            try:
                self.send_response(status)
                self.send_header('Content-Type', content)
                self.send_header('Content-Length', str(len(data)))
                self.send_header('Connection', 'close')
                self.send_header('Cache-Control', 'no-store')
                self.send_header('X-Content-Type-Options', 'nosniff')
                self.send_header('Content-Security-Policy', "default-src 'self'; img-src 'self'; connect-src 'self'; style-src 'self'; script-src 'self'; frame-ancestors 'none'; base-uri 'none'")
                self.end_headers()
                self.wfile.write(data)
            except OSError:
                # Clients can navigate away or hit their own request timeout
                # while a bounded local operation is still completing. The
                # socket may already be invalid by the time a worker writes
                # its response (including WinError 10038 on Windows). Treat
                # every socket write failure as a disconnect; letting it
                # reach do_POST's operation-error handler would try to write
                # a second response to the same dead socket.
                self.close_connection = True

        def handle(self):
            try:
                super().handle()
            except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
                # A caller may close its HTTP/1.1 socket as soon as it has
                # received the one response this local API supports.
                self.close_connection = True
            except OSError as exc:
                if getattr(exc, 'winerror', None) == 10038:
                    self.close_connection = True
                else:
                    raise

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

        def download_learning_dataset(self, query):
            temporary = None
            try:
                from .dataset_export import build
                temporary, path = build(library, query['mission'][0])
                size = path.stat().st_size
                self.send_response(200)
                self.send_header('Content-Type', 'application/zip')
                self.send_header('Content-Length', str(size))
                self.send_header('Content-Disposition', 'attachment; filename="RoadLoom-Learning-Dataset.zip"')
                self.send_header('Cache-Control', 'private, no-store')
                self.send_header('X-Content-Type-Options', 'nosniff')
                self.end_headers()
                with path.open('rb') as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b''):
                        self.wfile.write(block)
            finally:
                if temporary is not None:
                    temporary.cleanup()

        def do_GET(self):
            if not self.valid_host():
                return self.reply(403, {'error': 'Configured host required'})
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            try:
                if parsed.path == '/api/session':
                    return self.reply(200, {'token': token, 'missions': library.missions(),
                        'mode': 'loopback-development' if loopback else 'lan'})
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
                    return self.reply(200, list_crops(library,query['mission'][0],
                        query.get('image',[None])[0],query.get('q',[''])[0]))
                if parsed.path == '/api/plates/page':
                    from .plate_review import list_crops_page
                    return self.reply(200,list_crops_page(library,query['mission'][0],
                        offset=int(query.get('offset',['0'])[0]),
                        limit=int(query.get('limit',['40'])[0]),
                        image_id=query.get('image',[None])[0],search=query.get('q',[''])[0]))
                if parsed.path == '/api/plates/metrics':
                    from .plate_review import ocr_quality_report
                    return self.reply(200,ocr_quality_report(library,query['mission'][0]))
                if parsed.path == '/api/plate-crop':
                    from .plate_review import crop_file
                    path=crop_file(library,query['mission'][0],query['crop'][0])
                    return self.reply(200,path.read_bytes(),'image/jpeg')
                if parsed.path == '/api/plate-context':
                    from .plate_review import context_file
                    path=context_file(library,query['mission'][0],query['crop'][0])
                    return self.reply(200,path.read_bytes(),'image/jpeg')
                if parsed.path == '/api/night-learning':
                    from .learning import (inventory,reviewed_evidence_inventory,candidates,
                        evaluation_status,governance_status,task_status)
                    from .learning_capabilities import inventory as capability_inventory
                    m=query['mission'][0]
                    tasks=task_status(library,m)
                    for task in tasks:
                        task['yielded_to_inference']=bool(task['status']=='running' and inference.training_yielded)
                    return self.reply(200,{'inventory':inventory(library,m),'candidates':candidates(library,m),
                        'evidence_inventory':reviewed_evidence_inventory(library,m),
                        'evaluation':evaluation_status(library,m),'governance':governance_status(library,m),
                        'tasks':tasks,
                        'capabilities':capability_inventory()})
                if parsed.path == '/api/learning/dataset':
                    return self.download_learning_dataset(query)
                if parsed.path == '/api/inference':
                    return self.reply(200,inference.status(query['mission'][0]))
                if parsed.path == '/api/video/media':
                    return self.stream_video(query)
                if parsed.path == '/api/video/frame-overlay':
                    return self.reply(200,video_store.frame_overlay(library,
                        query['mission'][0],query['run'][0],float(query['seconds'][0])))
                if parsed.path == '/api/video/clips':
                    return self.reply(200,video_clips.exports(library,query['mission'][0]))
                if parsed.path == '/api/video/clips/estimate':
                    return self.reply(200,video_clips.estimate_event_clip(library,
                        query['mission'][0],query['event_id'][0],query['pre_seconds'][0],
                        query['post_seconds'][0]))
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
                    '/timezone-countries.json':'data/timezone-countries.json',
                    '/timezone-zones.json':'data/timezone-zones.json',
                    '/timezone-cities.json':'data/timezone-cities.json','/roadloom.svg':'roadloom.svg','/roadloom.ico':'roadloom.ico'}
                if parsed.path in files:
                    path = Path(web) / files[parsed.path]
                    return self.reply(200, path.read_bytes(), mimetypes.guess_type(path.name)[0]+'; charset=utf-8')
                self.reply(404, {'error':'Not found'})
            except (ValueError, KeyError, FileNotFoundError) as exc:
                self.reply(400, {'error':str(exc)})

        def do_POST(self):
            if not self.valid_host():
                return self.reply(403, {'error':'Configured host required'})
            if self.headers.get('X-Session-Token') != token:
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
                    result = {'id': library.create(data['name'],data['country'],data['zone'],settings,
                        request_id=data.get('request_id'))}
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
                        result=library.add_view_region(data['mission'],data['view_id'],data['kind'],data['polygon'],data['operator'],data.get('name'))
                    elif action == 'update':
                        result=library.update_view_region(data['mission'],data['region_id'],data['kind'],data['polygon'],data['operator'],data.get('name'))
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
                elif self.path == '/api/plates/text-review':
                    from .plate_review import review_text
                    result=review_text(library,data['mission'],data['crop_id'],data.get('expected_revision',0),data['decision'],data.get('verified_text',''),data['operator'])
                elif self.path == '/api/plates/export':
                    from .plate_review import export_crops_csv
                    result=export_crops_csv(library,data['mission'])
                elif self.path == '/api/review':
                    result = library.review(data['mission'],data['image'],data['revision'],data['annotations'],data['operator'],data['role'],data['status'],data.get('prediction_run'))
                elif self.path == '/api/object-search/interpret':
                    from .object_search import interpret,video_scope_snapshot
                    from .inference_store import latest_run_id, search_scope_snapshot
                    settings=library.settings(data['mission'])
                    if settings['input_type']=='videos':
                        completed=[item for item in video_store.runs(library,data['mission'])
                            if item['status']=='completed']
                        run=data.get('run') or (completed[0]['id'] if completed else None)
                        if not run:
                            raise ValueError('Complete a video analysis run before searching events')
                        snapshot=video_scope_snapshot(library,data['mission'],run)
                        result={'spec':interpret(data['query'],mission_id=data['mission'],
                            run_id=run,scope_snapshot=snapshot,media_type='videos')}
                    else:
                        run = data.get('run') or latest_run_id(library,data['mission'])
                        snapshot = search_scope_snapshot(library,data['mission'],run)
                        result = {'spec': interpret(data['query'], mission_id=data['mission'],
                            view_id=data.get('view_id') or None, run_id=run, scope_snapshot=snapshot)}
                elif self.path == '/api/object-search/execute':
                    from .object_search import execute,execute_video,video_scope_snapshot
                    spec=data.get('spec')
                    run=(spec.get('run_id') if isinstance(spec,dict) else None) or data.get('run') or None
                    if isinstance(spec,dict) and spec.get('media_type')=='videos':
                        snapshot=video_scope_snapshot(library,data['mission'],run)
                        result=execute_video(library,data['mission'],spec,
                            confirmed=data.get('confirmed'),run=data.get('run') or None,
                            scope_snapshot=snapshot)
                    else:
                        from .inference_store import search_scope_snapshot
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
                    result=schedule_learning_task(data['mission'],'train')
                elif self.path == '/api/night-learning/evaluate-test':
                    result=schedule_learning_task(data['mission'],'evaluate_test',{
                        'candidate_id':data.get('candidate_id'),'operator':data.get('operator')})
                elif self.path == '/api/night-learning/cancel':
                    from .learning import cancel_task
                    result=cancel_task(library,data['mission'],data['task_id'])
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
                    video_store.logical_event_by_id(library,data['mission'],data['event_id'])
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
            except Exception as exc:
                self.log_error('Internal request error (%s): %s', type(exc).__name__, exc)
                traceback.print_exc()
                self.reply(500,{'error':'Operation failed; inspect local diagnostics'})
    server = ThreadingHTTPServer((host,port), Handler)
    if not loopback:
        context=ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.minimum_version=ssl.TLSVersion.TLSv1_2
        context.load_cert_chain(tls_cert,tls_key)
        server.socket=context.wrap_socket(server.socket,server_side=True)
    scheme='http' if loopback else 'https'
    display_host='127.0.0.1' if loopback else public_host
    print(f'RoadLoom {"local" if loopback else "LAN"} host: {scheme}://{display_host}:{server.server_port}', flush=True)
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
