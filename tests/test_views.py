import json
from pathlib import Path
import tempfile
import unittest
from PIL import Image
from road_traffic.inference_store import create_run
from road_traffic.store import Library


class CameraViewTests(unittest.TestCase):
    def test_batch_assignment_is_atomic_and_keeps_per_image_history(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);library=Library(root/'workspace')
            mission=library.create('Views','Israel','Asia/Jerusalem')
            source=root/'source';source.mkdir()
            for minute in (1604,1605,1606):
                Image.new('RGB',(8,8),(10,20,30)).save(source/f'17052026-{minute}-view-a.jpg','JPEG')
            library.import_folder(mission,source,'camera')
            images=library.images(mission)
            first=library.create_view(mission,'Gate A','Operator')
            second=library.create_view(mission,'Gate B','Operator')
            initial=library.assign_views(mission,[images[0]['id'],images[1]['id']],first['id'],'Operator')
            self.assertEqual((initial['requested'],initial['changed'],initial['unchanged']),(2,2,0))
            result=library.assign_views(mission,[images[0]['id'],images[1]['id'],images[0]['id']],second['id'],'Operator 2')
            self.assertEqual((result['requested'],result['changed'],result['unchanged']),(2,2,0))
            same=library.assign_views(mission,[images[0]['id'],images[1]['id']],second['id'],'Operator 2')
            self.assertEqual((same['changed'],same['unchanged']),(0,2))
            with self.assertRaisesRegex(ValueError,'not found in this mission'):
                library.assign_views(mission,[images[2]['id'],'missing'],first['id'],'Operator')
            self.assertEqual(library.view_history(mission,images[2]['id']),[])
            self.assertEqual([event['view_name'] for event in library.view_history(mission,images[0]['id'])],
                ['Gate A','Gate B'])
            with library.connect(mission) as conn:
                self.assertEqual(conn.execute('SELECT COUNT(*) FROM view_events WHERE action="images_assigned"').fetchone()[0],4)

    def test_assignments_and_renames_keep_history_and_run_snapshot(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);library=Library(root/'workspace')
            mission=library.create('Views','Israel','Asia/Jerusalem')
            source=root/'source';source.mkdir()
            Image.new('RGB',(8,8),(10,20,30)).save(source/'17052026-1604-view-a.jpg','JPEG')
            Image.new('RGB',(8,8),(30,20,10)).save(source/'17052026-1605-view-b.jpg','JPEG')
            library.import_folder(mission,source,'camera')
            rows=library.images(mission)
            first=library.create_view(mission,'North entrance','Operator A')
            second=library.create_view(mission,'South entrance','Operator A')
            library.assign_view(mission,rows[0]['id'],first['id'],'Operator A')
            library.assign_view(mission,rows[0]['id'],second['id'],'Operator B')
            library.assign_view(mission,rows[1]['id'],first['id'],'Operator A')
            run=create_run(library,mission,{'adapter':'view-fixture'})
            library.rename_view(mission,second['id'],'South approach','Operator B')
            views={v['id']:v for v in library.views(mission)}
            self.assertEqual(views[first['id']]['image_count'],1)
            self.assertEqual(views[second['id']]['image_count'],1)
            self.assertEqual(views[second['id']]['name'],'South approach')
            history=library.view_history(mission,rows[0]['id'])
            self.assertEqual([event['view_name'] for event in history],['North entrance','South approach'])
            with library.connect(mission) as conn:
                snapshot=conn.execute('SELECT view_id,view_revision FROM inference_input_views WHERE inference_id=? AND image_id=?',
                    (run,rows[0]['id'])).fetchone()
            self.assertEqual(tuple(snapshot),(second['id'],2))
            reviewed={row['id']:row for row in library.images(mission,run)}
            self.assertEqual(reviewed[rows[0]['id']]['view_name'],'South entrance')
            with library.connect(mission) as conn:
                self.assertEqual(conn.execute('SELECT COUNT(*) FROM view_events').fetchone()[0],6)


if __name__=='__main__':
    unittest.main()
