"""Loopback development host. LAN deployment requires the planned authentication/TLS host."""
import csv
import io
import json
import mimetypes
import re
import secrets
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs
from .store import Library
from . import video_store, video_clips


def serve(workspace, web, port=8765, open_browser=False):
    library = Library(workspace)
    library.reconcile_exports()
    from .monitor import SourceMonitor
    from .inference import InferenceService
    inference=InferenceService(library,Path(web).parent/'model-packs'/'yolox-s')
    from .video_inference import VideoInferenceService
    video_inference=VideoInferenceService(library,Path(web).parent/'model-packs'/'yolox-s',inference)
    monitor = SourceMonitor(library,on_batch=inference.start_continuous_batch)
    monitor.start()
    video_clips.reconcile_exports(library)
    token = secrets.token_urlsafe(32)
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
            return self.headers.get('Host') in (f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}')

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

        def do_GET(self):
            if not self.valid_host():
                return self.reply(403, {'error': 'Loopback host required'})
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            try:
                if parsed.path == '/api/session':
                    return self.reply(200, {'token': token, 'missions': library.missions(), 'mode': 'loopback-development'})
                if parsed.path == '/api/hardware':
                    from .hardware import inventory
                    return self.reply(200, inventory(library.root))
                if parsed.path == '/api/images':
                    return self.reply(200, library.images(query['mission'][0],query.get('run',[None])[0]))
                if parsed.path == '/api/night-learning':
                    from .learning import inventory,candidates
                    m=query['mission'][0]
                    return self.reply(200,{'inventory':inventory(library,m),'candidates':candidates(library,m)})
                if parsed.path == '/api/inference':
                    return self.reply(200,inference.status(query['mission'][0]))
                if parsed.path == '/api/video/media':
                    return self.stream_video(query)
                if parsed.path == '/api/video/clips':
                    return self.reply(200,video_clips.exports(library,query['mission'][0]))
                if parsed.path == '/api/video/clips/download':
                    return self.download_clip(query)
                if parsed.path == '/api/video/sources':
                    return self.reply(200,video_store.sources(library,query['mission'][0]))
                if parsed.path == '/api/video/runs':
                    return self.reply(200,video_inference.status(query['mission'][0]))
                if parsed.path == '/api/video/events':
                    return self.reply(200,video_store.events(library,query['mission'][0],query['run'][0],
                        limit=int(query.get('limit',['100'])[0]),offset=int(query.get('offset',['0'])[0])))
                if parsed.path == '/api/exports':
                    return self.reply(200,library.exports(query['mission'][0]))
                if parsed.path == '/api/monitoring':
                    return self.reply(200,monitor.status(query['mission'][0]))
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
            if not self.valid_host() or self.headers.get('X-Session-Token') != token:
                return self.reply(403, {'error':'Session token required'})
            if self.headers.get('Origin') not in (None, f'http://127.0.0.1:{self.server.server_port}', f'http://localhost:{self.server.server_port}'):
                return self.reply(403, {'error':'Invalid origin'})
            try:
                length = int(self.headers.get('Content-Length','0'))
                if not 0 < length <= 1024*1024:
                    return self.reply(413, {'error':'Request exceeds limit'})
                data = json.loads(self.rfile.read(length))
                if self.path == '/api/missions':
                    settings = {key: data[key] for key in (
                        'input_type', 'analysis_mode', 'location', 'source_root',
                        'include_subdirectories', 'processing_mode', 'output_root', 'export_formats'
                    ) if key in data}
                    result = {'id': library.create(data['name'],data['country'],data['zone'],settings)}
                elif self.path == '/api/export':
                    mission = data['mission']
                    if any(job['status'] in ('queued', 'running') for job in inference.status(mission)):
                        raise ValueError('Wait for processing to finish before exporting a stable review snapshot')
                    result = library.export_configured(mission, data.get('run'))
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
                elif self.path == '/api/views':
                    action=data.get('action')
                    if action == 'create':
                        result=library.create_view(data['mission'],data['name'],data['operator'])
                    elif action == 'rename':
                        result=library.rename_view(data['mission'],data['id'],data['name'],data['operator'])
                    elif action == 'assign':
                        result=library.assign_view(data['mission'],data['image_id'],data.get('view_id'),data['operator'])
                    elif action == 'assign_many':
                        result=library.assign_views(data['mission'],data.get('image_ids'),data.get('view_id'),data['operator'])
                    else:
                        raise ValueError('Choose create, rename, assign, or assign_many action')
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
                elif self.path == '/api/review':
                    result = library.review(data['mission'],data['image'],data['revision'],data['annotations'],data['operator'],data['role'],data['status'],data.get('prediction_run'))
                elif self.path == '/api/object-search/interpret':
                    from .object_search import interpret
                    library.settings(data['mission'])
                    run = data.get('run') or None
                    rows = library.images(data['mission'], run)
                    run = run or next((row.get('prediction_run') for row in rows if row.get('prediction_run')), None)
                    result = {'spec': interpret(data['query'], mission_id=data['mission'],
                        view_id=data.get('view_id') or None, run_id=run)}
                elif self.path == '/api/object-search/execute':
                    from .object_search import execute
                    result = execute(library, data['mission'], data.get('spec'),
                        confirmed=data.get('confirmed'), run=data.get('run') or None)
                elif self.path == '/api/inference/start':
                    result=inference.start(data['mission'],data.get('threshold',.18),data.get('tiled',True),data.get('night_lights',True),data.get('night_head_id'))
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
                elif self.path == '/api/inference/stop':
                    result=inference.stop(data['mission'],data['run'])
                elif self.path == '/api/video/sources':
                    result=video_store.register_video(library,data['mission'],data['relative_path'],
                        metadata=data.get('metadata'),recording_start_utc=data.get('recording_start_utc'),
                        origin_provenance=data.get('origin_provenance'))
                elif self.path == '/api/video/runs':
                    result=video_inference.start(data['mission'],data['source_id'],data['tracker_policy'],
                        threshold=data['threshold'],tiled=data['tiled'])
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
                else:
                    return self.reply(404, {'error':'Not found'})
                self.reply(200,result)
            except (ValueError, KeyError, TypeError, OSError) as exc:
                self.reply(400,{'error':str(exc)})
            except Exception:
                self.log_error('Internal request error')
                self.reply(500,{'error':'Operation failed; inspect local diagnostics'})
    server = ThreadingHTTPServer(('127.0.0.1',port), Handler)
    print(f'Traffic Analysis development host: http://127.0.0.1:{server.server_port}', flush=True)
    print('Local YOLOX-S inference available when the verified model pack and runtime are installed. No external assets.', flush=True)
    if open_browser:
        threading.Thread(target=webbrowser.open, args=(f'http://127.0.0.1:{server.server_port}',), daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        monitor.stop()
        server.server_close()
