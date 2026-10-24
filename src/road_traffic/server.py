"""Loopback development host. LAN deployment requires the planned authentication/TLS host."""
import csv
import io
import json
import mimetypes
import secrets
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs
from .store import Library


def serve(workspace, web, port=8765, open_browser=False):
    library = Library(workspace)
    library.reconcile_exports()
    from .monitor import SourceMonitor
    monitor = SourceMonitor(library)
    monitor.start()
    from .inference import InferenceService
    inference=InferenceService(library,Path(web).parent/'model-packs'/'yolox-s')
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

        def do_GET(self):
            if not self.valid_host():
                return self.reply(403, {'error': 'Loopback host required'})
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            try:
                if parsed.path == '/api/session':
                    return self.reply(200, {'token': token, 'missions': library.missions(), 'mode': 'loopback-development'})
                if parsed.path == '/api/images':
                    return self.reply(200, library.images(query['mission'][0],query.get('run',[None])[0]))
                if parsed.path == '/api/night-learning':
                    from .learning import inventory,candidates
                    m=query['mission'][0]
                    return self.reply(200,{'inventory':inventory(library,m),'candidates':candidates(library,m)})
                if parsed.path == '/api/inference':
                    return self.reply(200,inference.status(query['mission'][0]))
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
                    if 'xlsx' in formats:
                        raise ValueError('This mission requests XLSX output, which is not implemented yet; no export was generated')
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
                files = {'/':'index.html','/app.js':'app.js','/style.css':'style.css'}
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
                    result = library.export_csv(mission, data.get('run'))
                elif self.path == '/api/import':
                    result = library.import_folder(data['mission'],data['path'],data['preset'])
                elif self.path == '/api/scan-windows':
                    if data.get('action') == 'delete':
                        result = library.delete_scan_window(data['mission'],data['id'])
                    else:
                        result = library.add_scan_window(data['mission'],data['window'])
                elif self.path == '/api/views':
                    action=data.get('action')
                    if action == 'create':
                        result=library.create_view(data['mission'],data['name'],data['operator'])
                    elif action == 'rename':
                        result=library.rename_view(data['mission'],data['id'],data['name'],data['operator'])
                    elif action == 'assign':
                        result=library.assign_view(data['mission'],data['image_id'],data.get('view_id'),data['operator'])
                    else:
                        raise ValueError('Choose create, rename, or assign action')
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
                elif self.path == '/api/inference/start':
                    result=inference.start(data['mission'],data.get('threshold',.18),data.get('tiled',True),data.get('night_lights',True),data.get('night_head_id'))
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
