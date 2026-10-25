"""Small service-side process controller; framework imports stay in the worker."""
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from . import inference_store
from .store import now


class InferenceBusy(ValueError):
    """A new inference batch can wait until the current worker finishes."""

    retryable = True


class InferenceService:
    def __init__(self,library,pack):
        self.library,self.pack=library,Path(pack).resolve()
        self.lock=threading.Lock();self.processes={};self.training_active=False
        self.recover_after_restart()

    @staticmethod
    def _pid_alive(pid):
        if not pid:return False
        if os.name=='nt':
            import ctypes
            from ctypes import wintypes
            kernel=ctypes.WinDLL('kernel32',use_last_error=True)
            kernel.OpenProcess.restype=wintypes.HANDLE
            kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
            kernel.GetExitCodeProcess.argtypes=[wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD)]
            kernel.CloseHandle.argtypes=[wintypes.HANDLE]
            handle=kernel.OpenProcess(0x1000,False,int(pid))
            if not handle:return ctypes.get_last_error() in (5,)
            code=wintypes.DWORD()
            try:return bool(kernel.GetExitCodeProcess(handle,ctypes.byref(code)) and code.value==259)
            finally:kernel.CloseHandle(handle)
        try:
            os.kill(int(pid),0)
            return True
        except ProcessLookupError:return False
        except PermissionError:return True
        except OSError:return False

    def recover_after_restart(self):
        """Fail runs whose worker vanished; preserve completed per-image records."""
        for mission_row in self.library.missions():
            mission=mission_row['id'];inference_store.prepare(self.library,mission)
            with self.library.connect(mission) as conn:
                active=conn.execute("SELECT id,status,done,total,worker_pid FROM inference_runs WHERE status IN ('queued','running')").fetchall()
                for job in active:
                    if job['status']=='running' and self._pid_alive(job['worker_pid']):continue
                    reason=(f"Service restarted before the worker recorded its process ID; {job['done']} of {job['total']} saved."
                        if job['status']=='queued' else f"Worker process ended after a service restart; {job['done']} of {job['total']} image results are saved.")
                    conn.execute("UPDATE inference_runs SET status='failed',stage='failed',stop_requested=1,error=?,finished=? WHERE id=?",
                        (reason+' Resume this run to process remaining inputs.',now(),job['id']))

    def _spawn(self,mission,run):
        folder=self.library.location(mission)/'inference'/run;folder.mkdir(parents=True,exist_ok=True)
        try:
            with (folder/'worker.log').open('ab') as log:
                process=subprocess.Popen([sys.executable,str(Path(__file__).resolve().parents[2]/'scripts'/'run-inference.py'),'--workspace',str(self.library.root),
                    '--mission',mission,'--run',run,'--pack',str(self.pack)],stdout=log,stderr=subprocess.STDOUT)
            self.processes[(mission,run)]=process
        except Exception as exc:
            with self.library.connect(mission) as conn:
                conn.execute("UPDATE inference_runs SET status='failed',stage='failed',error=?,finished=? WHERE id=?",(str(exc),now(),run))
            raise

    def _has_active_worker(self):
        if any(process.poll() is None for process in self.processes.values()):return True
        for mission_row in self.library.missions():
            mission=mission_row['id'];inference_store.prepare(self.library,mission)
            with self.library.connect(mission) as conn:
                active=conn.execute("SELECT id,status,done,total,worker_pid FROM inference_runs WHERE status IN ('queued','running')").fetchall()
                for job in active:
                    process=self.processes.get((mission,job['id']))
                    if process and process.poll() is None:return True
                    if job['status']=='running' and self._pid_alive(job['worker_pid']):return True
                    conn.execute("UPDATE inference_runs SET status='failed',stage='failed',stop_requested=1,error=?,finished=? WHERE id=?",
                        (f'Worker process is no longer available; {job["done"]} of {job["total"]} image results are saved. Resume this run.',now(),job['id']))
            from . import video_store
            for job in video_store.runs(self.library, mission):
                if job['status'] not in ('queued', 'running'):
                    continue
                process = self.processes.get((mission, job['id']))
                if process is not None and process.poll() is None:
                    return True
                if self._pid_alive(job['worker_pid']):
                    return True
                video_store.fail_worker(self.library, mission, job['id'],
                    f'Video worker is no longer available; {job["frames_done"]} frames are saved. Resume this run.')
        return False

    def start(self,mission,threshold=.18,tiled=True,night_lights=True,night_head_id=None,source_run=None,resource_profile='auto'):
        from .detector import model_pack,VERSION
        from .hardware import resolve_resource_profile
        settings = self.library.settings(mission)
        if settings['input_type'] != 'images' or settings['analysis_mode'] != 'general_traffic':
            raise ValueError('Automatic detection currently supports still-image General Traffic missions only')
        if type(threshold) not in (float,int) or not .01<=threshold<=.9:raise ValueError('Threshold must be 0.01–0.90')
        if type(tiled) is not bool or type(night_lights) is not bool:raise ValueError('Expected boolean settings')
        resources=resolve_resource_profile(resource_profile)
        _,sha=model_pack(self.pack)
        config={'adapter':VERSION,'model_sha256':sha,'threshold':threshold,'tiled':tiled,'night_lights':night_lights,
            'device':resources['device'],'cpu_threads':resources['cpu_threads'],'resource_profile':resources,
            'training_supported':False,'training_capabilities':{'yolox_s':'unsupported','night_light_head':'refit from explicit labels'},'policy':'All automatic results require human confirmation'}
        if night_head_id:
            from .learning import load_head
            _,head_hash=load_head(self.library,mission,night_head_id)
            config.update(night_head_id=night_head_id,night_head_hash=head_hash)
        with self.lock:
            if self.training_active:raise ValueError('Finish correction-head training before starting inference')
            # Only one local worker at a time; avoid contention on 16 GB machines.
            if self._has_active_worker():raise InferenceBusy('An inference worker is already running')
            run=inference_store.create_run(self.library,mission,config,source_run)
            self._spawn(mission,run)
        return {'run':run,'config':config}

    def rerun_roi(self,mission,run,operator):
        from .detector import model_pack,VERSION
        settings=self.library.settings(mission)
        if settings['input_type']!='images' or settings['analysis_mode']!='general_traffic':
            raise ValueError('ROI reruns currently support still-image General Traffic missions only')
        with self.lock:
            if self.training_active:raise ValueError('Finish correction-head training before starting inference')
            if self._has_active_worker():raise InferenceBusy('An inference worker is already running')
            with self.library.connect(mission) as conn:
                row=conn.execute('SELECT config FROM inference_runs WHERE id=?',(run,)).fetchone()
            if not row:raise ValueError('Inference run not found')
            config=json.loads(row['config'])
            _,sha=model_pack(self.pack)
            if config.get('adapter')!=VERSION or config.get('model_sha256')!=sha:
                raise ValueError('The source model changed; start a new run to apply the current ROI')
            if config.get('night_head_id'):
                from .learning import load_head
                _,head_hash=load_head(self.library,mission,config['night_head_id'])
                if head_hash!=config.get('night_head_hash'):
                    raise ValueError('The source correction head changed; start a new run to apply the current ROI')
            result=inference_store.create_roi_rerun(self.library,mission,run,operator)
            self._spawn(mission,result['run'])
        return result

    def start_continuous_batch(self,mission,source_run):
        settings=self.library.settings(mission)
        if settings['processing_mode']!='continuous':raise ValueError('Mission is not configured for continuous processing')
        if settings['input_type']!='images' or settings['analysis_mode']!='general_traffic':
            raise ValueError('Automatic continuous inference currently supports still-image General Traffic missions only')
        for job in inference_store.runs(self.library,mission):
            if json.loads(job['config']).get('source_run_id')!=source_run:continue
            if job['status'] in ('failed','cancelled'):
                return {'run':job['id'],'status':job['status'],'existing':True}
            return {'run':job['id'],'status':job['status'],'existing':True}
        try:
            return self.start(mission,source_run=source_run)
        except Exception:
            # _spawn can fail after the immutable run has already been committed.
            # Return that run so the durable monitor batch tracks the failed run
            # and the operator can resume it instead of creating duplicates.
            for job in inference_store.runs(self.library,mission):
                if json.loads(job['config']).get('source_run_id')==source_run:
                    return {'run':job['id'],'status':job['status'],'existing':True}
            raise

    def resume(self,mission,run):
        from .detector import model_pack,VERSION
        with self.lock:
            if self.training_active:raise ValueError('Finish correction-head training before resuming inference')
            if self._has_active_worker():raise InferenceBusy('An inference worker is already running')
            inference_store.prepare(self.library,mission)
            with self.library.connect(mission) as conn:
                job=conn.execute('SELECT * FROM inference_runs WHERE id=?',(run,)).fetchone()
                if not job:raise ValueError('Inference run not found')
                if job['status'] not in ('failed','cancelled'):raise ValueError('Only failed or stopped runs can be resumed')
                if self._pid_alive(job['worker_pid']):raise ValueError('The previous worker is still stopping; retry in a moment')
                config=json.loads(job['config'])
                if config.get('adapter')!=VERSION:raise ValueError('Adapter version changed; start a new run')
                _,sha=model_pack(self.pack)
                if sha!=config.get('model_sha256'):raise ValueError('Model changed since this run; start a new run')
                completed=conn.execute('SELECT COUNT(*) FROM predictions WHERE inference_id=?',(run,)).fetchone()[0]
                if completed>=job['total']:raise ValueError('This run has no unfinished image inputs')
                conn.execute("UPDATE inference_runs SET status='queued',stage='queued',done=?,stop_requested=0,error=NULL,finished=NULL,worker_pid=NULL WHERE id=?",
                    (completed,run))
            self._spawn(mission,run)
        return {'run':run,'status':'queued','remaining':job['total']-completed}

    def status(self,mission):
        inference_store.prepare(self.library,mission)
        with self.lock:
            for (m,run),process in list(self.processes.items()):
                if m!=mission or process.poll() is None:continue
                with self.library.connect(mission) as conn:
                    conn.execute("UPDATE inference_runs SET status='failed',stage='failed',error=?,finished=? WHERE id=? AND status IN ('queued','running')",
                        (f'Worker exited unexpectedly ({process.returncode}); see inference/{run}/worker.log',now(),run))
            with self.library.connect(mission) as conn:
                active=conn.execute("SELECT id,status,done,total,worker_pid FROM inference_runs WHERE status IN ('queued','running')").fetchall()
                for job in active:
                    if (mission,job['id']) in self.processes:continue
                    if job['status']=='running' and self._pid_alive(job['worker_pid']):continue
                    conn.execute("UPDATE inference_runs SET status='failed',stage='failed',stop_requested=1,error=?,finished=? WHERE id=?",
                        (f'Worker process ended unexpectedly; {job["done"]} of {job["total"]} image results are saved. Resume this run.',now(),job['id']))
        jobs=inference_store.runs(self.library,mission)
        for job in jobs:
            state=inference_store.roi_review_state(self.library,mission,job['id'])
            job['roi_changed_views']=state['changed_views']
            job['roi_stale']=state['stale']
            job['roi_review_required']=state['review_required']
            acknowledgment=state['acknowledgment']
            job['roi_rerun_acknowledgment']=acknowledgment
            if acknowledgment:
                job['roi_rerun_id']=acknowledgment['rerun_run']
                job['roi_rerun_status']=acknowledgment['rerun_status']
                job['roi_rerun_operator']=acknowledgment['operator']
        return jobs

    def stop(self,mission,run):
        inference_store.prepare(self.library,mission)
        with self.library.connect(mission) as conn:
            if not conn.execute('SELECT 1 FROM inference_runs WHERE id=?',(run,)).fetchone():raise ValueError('Run not found')
            conn.execute("UPDATE inference_runs SET stop_requested=1 WHERE id=? AND status IN ('queued','running')",(run,))
        return {'stop_requested':True}
