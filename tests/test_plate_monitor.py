import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from road_traffic.inference import InferenceService
from road_traffic.plate_monitor import PlateBatchService
from road_traffic.plate_review import MODEL_SHA256
from road_traffic.store import Library, now


class ContinuousPlateBatchTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.source=self.root/'incoming';self.source.mkdir()
        self.output=self.root/'output';self.output.mkdir()
        Image.new('RGB',(32,24),(30,60,90)).save(self.source/'17052026-1604-camera.jpg','JPEG')
        self.library=Library(self.root/'workspace')
        self.mission=self.library.create('Continuous plate crops','Lebanon','Asia/Beirut',{
            'input_type':'images','analysis_mode':'license_plate','processing_mode':'continuous',
            'source_root':str(self.source),'output_root':str(self.output)})
        self.pack=Path(__file__).resolve().parents[1]/'model-packs'/'license-plate-yolox'
        self.inference=InferenceService(self.library,self.pack)
        self.source_run=self.library.import_folder(self.mission,self.source,'camera')['run']
        with self.library.connect(self.mission) as conn:
            conn.execute("INSERT INTO monitor_batches(source_run_id,status,created) VALUES (?,'pending',?)",
                (self.source_run,now()))

    def tearDown(self):
        self.temp.cleanup()

    def test_new_images_run_crop_only_worker_and_finish_for_human_review(self):
        class Detector:
            sha=MODEL_SHA256
            config_sha256='fixture-config'
            def __init__(self,*args,**kwargs):pass
            def detect(self,image):return []
        service=PlateBatchService(self.library,self.pack,self.inference)
        with patch('road_traffic.plate_review.PlateDetector',Detector):
            result=service.start_continuous_batch(self.mission,self.source_run)
            self.assertEqual(result['status'],'queued')
            deadline=time.monotonic()+3
            while time.monotonic()<deadline:
                with self.library.connect(self.mission) as conn:
                    status=conn.execute('SELECT status FROM monitor_batches WHERE source_run_id=?',
                        (self.source_run,)).fetchone()['status']
                if status=='completed':break
                time.sleep(.01)
        self.assertEqual(status,'completed')
        self.assertFalse(self.inference.auxiliary_worker_active)
        with self.library.connect(self.mission) as conn:
            event=conn.execute("SELECT kind FROM monitor_events WHERE source_run_id=? ORDER BY id DESC LIMIT 1",
                (self.source_run,)).fetchone()
        self.assertEqual(event['kind'],'completed')


if __name__=='__main__':
    unittest.main()
