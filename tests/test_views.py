import json
from pathlib import Path
import tempfile
import unittest
from road_traffic.inference_store import create_run
from road_traffic.store import Library


class CameraViewTests(unittest.TestCase):
    def test_assignments_and_renames_keep_history_and_run_snapshot(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);library=Library(root/'workspace')
            mission=library.create('Views','Israel','Asia/Jerusalem')
            source=root/'source';source.mkdir()
            (source/'17052026-1604-view-a.jpg').write_bytes(b'view-a')
            (source/'17052026-1605-view-b.jpg').write_bytes(b'view-b')
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
