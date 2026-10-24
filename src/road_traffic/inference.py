"""Small service-side process controller; framework imports stay in the worker."""
import json
from pathlib import Path
import subprocess
import sys
import threading
from . import inference_store
from .store import now


class InferenceService:
    def __init__(self,library,pack):
        self.library,self.pack=library,Path(pack).resolve()
        self.lock=threading.Lock();self.processes={};self.training_active=False

    def start(self,mission,threshold=.18,tiled=True,night_lights=True,night_head_id=None):
        from .detector import model_pack,VERSION
        settings = self.library.settings(mission)
        if settings['input_type'] != 'images' or settings['analysis_mode'] != 'general_traffic':
            raise ValueError('Automatic detection currently supports still-image General Traffic missions only')
        if type(threshold) not in (float,int) or not .01<=threshold<=.9:raise ValueError('Threshold must be 0.01–0.90')
        if type(tiled) is not bool or type(night_lights) is not bool:raise ValueError('Expected boolean settings')
        _,sha=model_pack(self.pack)
        config={'adapter':VERSION,'model_sha256':sha,'threshold':threshold,'tiled':tiled,'night_lights':night_lights,
            'device':'CPUExecutionProvider','training_supported':False,'training_capabilities':{'yolox_s':'unsupported','night_light_head':'refit from explicit labels'},'policy':'All automatic results require human confirmation'}
        if night_head_id:
            from .learning import load_head
            _,head_hash=load_head(self.library,mission,night_head_id)
            config.update(night_head_id=night_head_id,night_head_hash=head_hash)
        with self.lock:
            if self.training_active:raise ValueError('Finish correction-head training before starting inference')
            # Only one local worker at a time; avoid contention on 16 GB machines.
            if any(p.poll() is None for p in self.processes.values()):raise ValueError('An inference worker is already running')
            run=inference_store.create_run(self.library,mission,config)
            folder=self.library.location(mission)/'inference'/run;folder.mkdir(parents=True)
            try:
                with (folder/'worker.log').open('ab') as log:
                    process=subprocess.Popen([sys.executable,str(Path(__file__).resolve().parents[2]/'scripts'/'run-inference.py'),'--workspace',str(self.library.root),
                        '--mission',mission,'--run',run,'--pack',str(self.pack)],stdout=log,stderr=subprocess.STDOUT)
                self.processes[(mission,run)]=process
            except Exception as exc:
                with self.library.connect(mission) as conn:
                    conn.execute("UPDATE inference_runs SET status='failed',stage='failed',error=?,finished=? WHERE id=?",(str(exc),now(),run))
                raise
        return {'run':run,'config':config}

    def status(self,mission):
        with self.lock:
            for (m,run),process in list(self.processes.items()):
                if m!=mission or process.poll() is None:continue
                with self.library.connect(mission) as conn:
                    conn.execute("UPDATE inference_runs SET status='failed',stage='failed',error=?,finished=? WHERE id=? AND status IN ('queued','running')",
                        (f'Worker exited unexpectedly ({process.returncode}); see inference/{run}/worker.log',now(),run))
        return inference_store.runs(self.library,mission)

    def stop(self,mission,run):
        inference_store.prepare(self.library,mission)
        with self.library.connect(mission) as conn:
            if not conn.execute('SELECT 1 FROM inference_runs WHERE id=?',(run,)).fetchone():raise ValueError('Run not found')
            conn.execute("UPDATE inference_runs SET stop_requested=1 WHERE id=? AND status IN ('queued','running')",(run,))
        return {'stop_requested':True}
