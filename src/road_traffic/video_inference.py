"""Supervise durable video jobs in the app's single local worker slot."""
from pathlib import Path
import subprocess
import sys

from . import video_store
from .store import now


class VideoInferenceService:
    def __init__(self, library, pack, inference_service, *, worker_script=None):
        self.library = library
        self.pack = Path(pack).resolve()
        self.inference = inference_service
        self.worker_script = Path(worker_script or
            Path(__file__).resolve().parents[2] / 'scripts' / 'run-video-analysis.py')
        self.recover_after_restart()

    def recover_after_restart(self):
        for mission_row in self.library.missions():
            mission = mission_row['id']
            for job in video_store.runs(self.library, mission):
                if job['status'] not in ('queued', 'running'):
                    continue
                if self.inference._pid_alive(job['worker_pid']):
                    continue
                saved = job['frames_done']
                reason = (f'Video worker ended after a service restart; {saved} frames are saved. '
                    'Resume this run to continue from its verified checkpoint.')
                video_store.fail_worker(self.library, mission, job['id'], reason)

    def _spawn(self, mission, run_id):
        folder = self.library.location(mission) / 'video' / run_id
        folder.mkdir(parents=True, exist_ok=True)
        process = None
        try:
            with (folder / 'worker.log').open('ab') as log:
                process = subprocess.Popen([sys.executable, str(self.worker_script),
                    '--workspace', str(self.library.root), '--mission', mission,
                    '--run', run_id, '--pack', str(self.pack)],
                    stdout=log, stderr=subprocess.STDOUT)
            self.inference.processes[(mission, run_id)] = process
            video_store.set_worker_pid(self.library, mission, run_id, process.pid)
            return process
        except Exception as exc:
            if process is not None:
                self.inference.processes.pop((mission, run_id), None)
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
            video_store.fail_worker(self.library, mission, run_id,
                f'Unable to start video worker: {type(exc).__name__}: {exc}')
            raise

    def start(self, mission, source_id, tracker_policy, *, threshold, tiled):
        from .detector import VERSION, model_pack
        settings = self.library.settings(mission)
        if settings['input_type'] != 'videos' or settings['analysis_mode'] != 'video_traffic':
            raise ValueError('Automatic video analysis supports Video Traffic missions only')
        if type(threshold) not in (int, float) or not .01 <= threshold <= .9:
            raise ValueError('Detector threshold must be between 0.01 and 0.90')
        if type(tiled) is not bool:
            raise ValueError('tiled must be a boolean')
        _, sha = model_pack(self.pack)
        config = {'detector': {'version': VERSION, 'sha256': sha,
            'threshold': float(threshold), 'tiled': tiled,
            'device': 'CPUExecutionProvider', 'human_confirmation': True}}
        with self.inference.lock:
            if self.inference.training_active:
                raise ValueError('Finish correction-head training before starting analysis')
            if self.inference._has_active_worker():
                raise ValueError('Another local processing worker is active')
            run = video_store.create_run(self.library, mission, source_id,
                {**config, 'tracker_policy': tracker_policy})
            self._spawn(mission, run['id'])
        return {'run': run['id'], 'status': 'queued', 'config': run['config']}

    def resume(self, mission, run_id):
        with self.inference.lock:
            if self.inference.training_active:
                raise ValueError('Finish correction-head training before resuming analysis')
            if self.inference._has_active_worker():
                raise ValueError('Another local processing worker is active')
            job = video_store.run_detail(self.library, mission, run_id)
            if job['status'] not in ('failed', 'cancelled'):
                raise ValueError('Only failed or cancelled video runs can be resumed')
            if self.inference._pid_alive(job['worker_pid']):
                raise ValueError('The previous video worker is still stopping; retry in a moment')
            # Revalidate the stored source and frozen detector identity before resuming.
            from .detector import model_pack, VERSION
            _, sha = model_pack(self.pack)
            detector = job['config'].get('detector', {})
            if detector.get('version') != VERSION or detector.get('sha256') != sha:
                raise ValueError('Detector changed since this run; create a new video run')
            video_store.transition(self.library, mission, run_id, 'queued')
            self._spawn(mission, run_id)
        return {'run': run_id, 'status': 'queued', 'frames_done': job['frames_done']}

    def cancel(self, mission, run_id):
        return video_store.request_cancel(self.library, mission, run_id)

    def status(self, mission):
        with self.inference.lock:
            for job in video_store.runs(self.library, mission):
                if job['status'] not in ('queued', 'running'):
                    continue
                process = self.inference.processes.get((mission, job['id']))
                if process is not None and process.poll() is None:
                    continue
                if self.inference._pid_alive(job['worker_pid']):
                    continue
                return_code = process.returncode if process is not None else None
                log_path = self.library.location(mission) / 'video' / job['id'] / 'worker.log'
                video_store.fail_worker(self.library, mission, job['id'],
                    f'Video worker exited unexpectedly ({return_code}); see {log_path}')
        return video_store.runs(self.library, mission)


class VehicleProposalAdapter:
    """Keep model class labels while exposing only detection geometry to tracking."""
    def __init__(self, detector):
        self.detector = detector

    def infer(self, rgb_frame):
        return [{'label': item.get('label') or item.get('suggested_kind') or item.get('kind'),
            'box': item['box']} for item in self.detector.infer(rgb_frame)]


def execute_worker(library, mission, run_id, pack):
    run = video_store.run_detail(library, mission, run_id)
    detector_config = run['config'].get('detector', {})
    from .detector import VehicleDetector, VERSION
    if detector_config.get('version') != VERSION:
        raise ValueError('Video detector version changed; create a new run')
    detector = VehicleDetector(pack, threshold=detector_config['threshold'],
        tiled=detector_config['tiled'])
    if detector.sha != detector_config.get('sha256'):
        raise ValueError('Video detector model hash changed; create a new run')

    def should_cancel():
        video_store.prepare(library, mission)
        with library.connect(mission) as conn:
            row = conn.execute('SELECT cancel_requested FROM video_runs WHERE mission_id=? AND id=?',
                (mission, run_id)).fetchone()
        if row is None:
            raise ValueError('Video run disappeared while its worker was active')
        return bool(row['cancel_requested'])

    try:
        return video_store.execute_video_run(library, mission, run_id,
            VehicleProposalAdapter(detector), should_cancel=should_cancel)
    except Exception as exc:
        with library.connect(mission) as conn:
            row = conn.execute('SELECT status FROM video_runs WHERE mission_id=? AND id=?',
                (mission, run_id)).fetchone()
        if row and row['status'] in ('queued', 'running'):
            video_store.fail_worker(library, mission, run_id,
                f'{type(exc).__name__}: {exc}')
        raise

