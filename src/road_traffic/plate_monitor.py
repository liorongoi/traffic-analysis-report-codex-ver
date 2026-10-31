"""Serialize continuous still-image plate crops through the shared worker slot."""
import threading

from .store import now


class PlateBatchService:
    def __init__(self, library, pack, inference_service):
        self.library=library
        self.pack=pack
        self.inference=inference_service
        self._recover_queued_batches()

    def _recover_queued_batches(self):
        for mission_row in self.library.missions():
            mission=mission_row['id']
            settings=self.library.settings(mission)
            if settings['input_type']!='images' or settings['analysis_mode']!='license_plate':
                continue
            with self.library.connect(mission) as conn:
                conn.execute("UPDATE monitor_batches SET status='pending',error=NULL WHERE status='queued'")

    def start_continuous_batch(self, mission, source_run):
        from .plate_review import _model_pack, _paths
        settings=self.library.settings(mission)
        if settings['processing_mode']!='continuous' or settings['input_type']!='images' or settings['analysis_mode']!='license_plate':
            raise ValueError('Automatic plate crops require a continuous still-image plate mission')
        _paths(self.library,mission)
        _model_pack(self.pack)
        with self.library.connect(mission) as conn:
            row=conn.execute('SELECT status FROM monitor_batches WHERE source_run_id=?',(source_run,)).fetchone()
            if not row:raise ValueError('Continuous image batch not found')
            if row['status']!='pending':return {'run':None,'status':row['status'],'existing':True}

        with self.inference.lock:
            yielded=self.inference._claim_worker_slot()
            try:
                self.inference.auxiliary_worker_active=True
                thread=threading.Thread(target=self._process,args=(mission,source_run,yielded),
                    name=f'roadloom-plate-batch-{source_run[:8]}',daemon=True)
                thread.start()
            except Exception:
                self.inference.auxiliary_worker_active=False
                if yielded:self.inference._resume_training_after_worker()
                raise
        return {'run':None,'status':'queued'}

    def _process(self, mission, source_run, yielded):
        from .plate_review import PlateDetector, _model_pack, propose
        errors=[]
        try:
            pack=self.pack
            detector=PlateDetector(pack,.18,tiled=True)
            with self.library.connect(mission) as conn:
                image_ids=[row['id'] for row in conn.execute(
                    'SELECT id FROM images WHERE run_id=? ORDER BY capture_utc,name',(source_run,))]
            for image_id in image_ids:
                try:
                    propose(self.library,mission,image_id,pack,.18,True,detector=detector)
                except Exception as exc:
                    errors.append(f'{image_id}: {type(exc).__name__}: {exc}')
            status='completed_with_errors' if errors else 'completed'
            error='\n'.join(errors)[:1000] if errors else None
        except Exception as exc:
            status='failed'
            error=f'{type(exc).__name__}: {exc}'[:1000]
        try:
            with self.library.connect(mission) as conn:
                conn.execute('UPDATE monitor_batches SET status=?,error=? WHERE source_run_id=?',
                    (status,error,source_run))
                kind='completed' if status=='completed' else ('completed_with_errors' if status=='completed_with_errors' else 'failed')
                details=error or 'Automatic plate-chip extraction finished; chips are awaiting human review.'
                conn.execute('INSERT OR IGNORE INTO monitor_events(source_run_id,kind,details,created) VALUES (?,?,?,?)',
                    (source_run,kind,details,now()))
        finally:
            with self.inference.lock:
                self.inference.auxiliary_worker_active=False
                if yielded:self.inference._resume_training_after_worker()
