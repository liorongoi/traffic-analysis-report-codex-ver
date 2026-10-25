"""Supervise durable video jobs in the app's single local worker slot."""
from pathlib import Path
import subprocess
import sys

from . import video_store
from .store import now


class VideoInferenceBusy(ValueError):
    retryable = True


class VideoInferenceService:
    def __init__(self, library, pack, inference_service, *, worker_script=None, plate_pack=None):
        self.library = library
        self.pack = Path(pack).resolve()
        self.inference = inference_service
        self.plate_pack=Path(plate_pack or self.pack.parent/'license-plate-yolox').resolve()
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

    def start(self, mission, source_id, tracker_policy, *, threshold, tiled, target_labels=None, resource_profile='auto'):
        from .detector import VERSION, model_pack
        from .detector import VEHICLES
        from .hardware import resolve_resource_profile
        settings = self.library.settings(mission)
        if settings['input_type'] != 'videos' or settings['analysis_mode'] != 'video_traffic':
            raise ValueError('Automatic video analysis supports Video Traffic missions only')
        if type(threshold) not in (int, float) or not .01 <= threshold <= .9:
            raise ValueError('Detector threshold must be between 0.01 and 0.90')
        if type(tiled) is not bool:
            raise ValueError('tiled must be a boolean')
        supported_labels=tuple(VEHICLES.values())
        if target_labels is None:target_labels=list(supported_labels)
        if (not isinstance(target_labels,list) or not target_labels
                or any(not isinstance(label,str) for label in target_labels)
                or len(set(target_labels))!=len(target_labels)
                or not set(target_labels).issubset(supported_labels)):
            raise ValueError('Choose one or more supported video vehicle classes')
        target_labels=[label for label in supported_labels if label in target_labels]
        resources=resolve_resource_profile(resource_profile)
        _, sha = model_pack(self.pack)
        config = {'detector': {'version': VERSION, 'sha256': sha,
            'threshold': float(threshold), 'tiled': tiled,
            'target_labels':target_labels,
            'device': resources['device'], 'cpu_threads':resources['cpu_threads'],
            'resource_profile':resources, 'human_confirmation': True}}
        with self.inference.lock:
            if self.inference.training_active:
                raise VideoInferenceBusy('Finish correction-head training before starting analysis')
            if self.inference._has_active_worker():
                raise VideoInferenceBusy('Another local processing worker is active')
            run = video_store.create_run(self.library, mission, source_id,
                {**config, 'tracker_policy': tracker_policy})
            self._spawn(mission, run['id'])
        return {'run': run['id'], 'status': 'queued', 'config': run['config']}

    def save_continuous_policy(self, mission, *, tracker_policy, threshold, tiled, target_labels, resource_profile='auto'):
        from .detector import VEHICLES
        from .hardware import resolve_resource_profile
        settings=self.library.settings(mission)
        if settings['processing_mode']!='continuous' or settings['input_type']!='videos' or settings['analysis_mode']!='video_traffic':
            raise ValueError('A saved automatic-run policy requires a continuous Video Traffic mission')
        if type(threshold) not in (int,float) or not .01<=threshold<=.9:
            raise ValueError('Detector threshold must be between 0.01 and 0.90')
        if type(tiled) is not bool:
            raise ValueError('tiled must be a boolean')
        supported_labels=tuple(VEHICLES.values())
        if (not isinstance(target_labels,list) or not target_labels
                or any(not isinstance(label,str) for label in target_labels)
                or len(set(target_labels))!=len(target_labels)
                or not set(target_labels).issubset(supported_labels)):
            raise ValueError('Choose one or more supported video vehicle classes')
        canonical_labels=[label for label in supported_labels if label in target_labels]
        canonical_tracker=video_store._canonical_tracker_policy(tracker_policy)
        resources=resolve_resource_profile(resource_profile)
        policy={'threshold':float(threshold),'tiled':tiled,'target_labels':canonical_labels,
            'tracker_policy':canonical_tracker,'resource_profile':resource_profile}
        return self.library.set_continuous_video_policy(mission,policy)

    def start_continuous_source(self,mission,source_id,policy=None):
        policy=policy or self.library.settings(mission).get('continuous_video_policy')
        if not policy:
            raise ValueError('Save the explicit detector and tracker policy before enabling automatic video analysis')
        for job in video_store.runs(self.library,mission):
            if job['source_id']==source_id:
                return {'run':job['id'],'status':job['status'],'existing':True}
        return self.start(mission,source_id,policy['tracker_policy'],threshold=policy['threshold'],
            tiled=policy['tiled'],target_labels=policy['target_labels'],
            resource_profile=policy.get('resource_profile','auto'))

    def start_plate_review(self,mission,source_id,*,threshold=.3):
        from .plate_review import _model_pack, _paths
        settings=self.library.settings(mission)
        if settings['input_type']!='videos' or settings['analysis_mode']!='license_plate':
            raise ValueError('Plate crop processing requires a video plate-crop mission')
        if type(threshold) not in (int,float) or not .01<=threshold<=.9:
            raise ValueError('Plate detector threshold must be between 0.01 and 0.90')
        _paths(self.library,mission)
        _,sha=_model_pack(self.plate_pack)
        config={'task':'license_plate','plate_detector':{'version':'plate-yolox-tiny-v1',
            'sha256':sha,'threshold':float(threshold),'device':'CPUExecutionProvider',
            'human_confirmation':True},'tracker_policy':{'max_gap_seconds':[1,1],
            'minimum_iou':0,'max_center_distance':1,'association_margin':0,
            'compatible_label_groups':[]}}
        with self.inference.lock:
            if self.inference.training_active:raise ValueError('Finish correction-head training before starting analysis')
            if self.inference._has_active_worker():raise ValueError('Another local processing worker is active')
            run=video_store.create_run(self.library,mission,source_id,config)
            self._spawn(mission,run['id'])
        return {'run':run['id'],'status':'queued','config':run['config']}

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
            if job['config'].get('task')=='license_plate':
                from .plate_review import _model_pack
                _,sha=_model_pack(self.plate_pack)
                detector=job['config'].get('plate_detector',{})
                if detector.get('sha256')!=sha:raise ValueError('Plate detector changed since this run; create a new video run')
            else:
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
        jobs=video_store.runs(self.library, mission)
        from .hardware import process_usage
        for job in jobs:
            if job['status'] in ('queued','running') and job.get('worker_pid'):
                job['resource_usage']=process_usage(job['worker_pid'])
        return jobs


class VehicleProposalAdapter:
    """Keep model class labels while exposing only detection geometry to tracking."""
    def __init__(self, detector, target_labels=None):
        self.detector = detector
        self.target_labels=set(target_labels) if target_labels is not None else None

    def infer(self, rgb_frame):
        return [{'label': item.get('label') or item.get('suggested_kind') or item.get('kind'),
            'box': item['box'], 'score': item.get('score',item.get('confidence'))}
            for item in self.detector.infer(rgb_frame)
            if self.target_labels is None or (item.get('label') or item.get('suggested_kind') or item.get('kind')) in self.target_labels]


def execute_worker(library, mission, run_id, pack):
    run = video_store.run_detail(library, mission, run_id)
    if run['config'].get('task')=='license_plate':
        from .plate_review import PlateDetector, _model_pack, propose_video_frame
        detector_config=run['config'].get('plate_detector',{})
        plate_pack=Path(pack).parent/'license-plate-yolox'
        _,sha=_model_pack(plate_pack)
        if detector_config.get('sha256')!=sha:raise ValueError('Plate detector model hash changed; create a new run')
        plate_detector=PlateDetector(plate_pack,detector_config['threshold'])
        class EmptyDetector:
            def infer(self,rgb_frame):return []
        def save_frame(decoded):
            propose_video_frame(library,mission,run['source_id'],decoded,plate_detector)
        try:
            return video_store.execute_video_run(library,mission,run_id,EmptyDetector(),
                should_cancel=lambda:_video_cancel_requested(library,mission,run_id),frame_callback=save_frame)
        except Exception as exc:
            with library.connect(mission) as conn:
                row=conn.execute('SELECT status FROM video_runs WHERE mission_id=? AND id=?',(mission,run_id)).fetchone()
            if row and row['status'] in ('queued','running'):
                video_store.fail_worker(library,mission,run_id,f'{type(exc).__name__}: {exc}')
            raise
    detector_config = run['config'].get('detector', {})
    from .detector import VehicleDetector, VERSION
    if detector_config.get('version') != VERSION:
        raise ValueError('Video detector version changed; create a new run')
    detector = VehicleDetector(pack, threshold=detector_config['threshold'],
        tiled=detector_config['tiled'],cpu_threads=detector_config.get('cpu_threads',2))
    if detector.sha != detector_config.get('sha256'):
        raise ValueError('Video detector model hash changed; create a new run')

    def should_cancel():return _video_cancel_requested(library,mission,run_id)

    try:
        return video_store.execute_video_run(library, mission, run_id,
            VehicleProposalAdapter(detector,detector_config.get('target_labels')),
            should_cancel=should_cancel)
    except Exception as exc:
        with library.connect(mission) as conn:
            row = conn.execute('SELECT status FROM video_runs WHERE mission_id=? AND id=?',
                (mission, run_id)).fetchone()
        if row and row['status'] in ('queued', 'running'):
            video_store.fail_worker(library, mission, run_id,
                f'{type(exc).__name__}: {exc}')
        raise


def _video_cancel_requested(library,mission,run_id):
    video_store.prepare(library,mission)
    with library.connect(mission) as conn:
        row=conn.execute('SELECT cancel_requested FROM video_runs WHERE mission_id=? AND id=?',
            (mission,run_id)).fetchone()
    if row is None:raise ValueError('Video run disappeared while its worker was active')
    return bool(row['cancel_requested'])

