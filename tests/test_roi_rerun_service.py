from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from PIL import Image

from road_traffic.detector import VERSION
from road_traffic.inference import InferenceService
from road_traffic.inference_store import create_run,roi_review_state
from road_traffic.store import Library


class RoiRerunServiceTests(unittest.TestCase):
    def test_service_reruns_same_inputs_and_spawns_new_run(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);library=Library(root/'workspace')
            mission=library.create('ROI rerun','Israel','Asia/Jerusalem')
            source=root/'source';source.mkdir()
            Image.new('RGB',(8,8),(20,40,60)).save(source/'17052026-1604-camera.jpg','JPEG')
            library.import_folder(mission,source,'camera')
            image=library.images(mission)[0]
            view=library.create_view(mission,'Entrance','Operator')
            library.assign_view(mission,image['id'],view['id'],'Operator')
            polygon=[[.1,.1],[.9,.1],[.9,.9],[.1,.9]]
            region=library.add_view_region(mission,view['id'],'include',polygon,'Operator')
            source_run=create_run(library,mission,{'adapter':VERSION,'model_sha256':'fixture-hash',
                'threshold':.18,'tiled':False,'night_lights':False})
            with library.connect(mission) as conn:
                conn.execute("UPDATE inference_runs SET status='completed',stage='finished',done=1 WHERE id=?",(source_run,))
            library.update_view_region(mission,region['region_key'],'exclude',[[.2,.2],[.8,.2],[.8,.8],[.2,.8]],'Operator B')
            service=InferenceService(library,root/'pack')
            with patch('road_traffic.detector.model_pack',return_value=(root/'pack','fixture-hash')), \
                 patch.object(service,'_spawn') as spawn:
                result=service.rerun_roi(mission,source_run,'Operator C')
            self.assertEqual(result['rerun_of'],source_run)
            self.assertEqual(result['total'],1)
            spawn.assert_called_once_with(mission,result['run'])
            state=roi_review_state(library,mission,source_run)
            self.assertEqual(state['acknowledgment']['rerun_run'],result['run'])
            self.assertEqual(state['acknowledgment']['operator'],'Operator C')

    def test_model_change_blocks_roi_only_rerun(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);library=Library(root/'workspace')
            mission=library.create('ROI model gate','Israel','Asia/Jerusalem')
            source=root/'source';source.mkdir()
            Image.new('RGB',(8,8),(20,40,60)).save(source/'17052026-1604-camera.jpg','JPEG')
            library.import_folder(mission,source,'camera')
            image=library.images(mission)[0]
            view=library.create_view(mission,'Entrance','Operator')
            library.assign_view(mission,image['id'],view['id'],'Operator')
            region=library.add_view_region(mission,view['id'],'include',[[.1,.1],[.9,.1],[.9,.9],[.1,.9]],'Operator')
            source_run=create_run(library,mission,{'adapter':VERSION,'model_sha256':'old-hash',
                'threshold':.18,'tiled':False,'night_lights':False})
            with library.connect(mission) as conn:
                conn.execute("UPDATE inference_runs SET status='completed',stage='finished',done=1 WHERE id=?",(source_run,))
            library.update_view_region(mission,region['region_key'],'exclude',[[.2,.2],[.8,.2],[.8,.8],[.2,.8]],'Operator B')
            service=InferenceService(library,root/'pack')
            with patch('road_traffic.detector.model_pack',return_value=(root/'pack','new-hash')):
                with self.assertRaisesRegex(ValueError,'source model changed'):
                    service.rerun_roi(mission,source_run,'Operator C')
            self.assertIsNone(roi_review_state(library,mission,source_run)['acknowledgment'])


if __name__=='__main__':
    unittest.main()
