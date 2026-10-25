import json
from pathlib import Path
import tempfile
import unittest
from PIL import Image
from road_traffic.inference_store import create_run
from road_traffic.store import Library


class CameraViewTests(unittest.TestCase):
    def test_legacy_view_database_gets_additive_merge_column_and_trigger(self):
        with tempfile.TemporaryDirectory() as temp:
            library=Library(Path(temp)/'workspace')
            mission=library.create('Legacy views','Israel','Asia/Jerusalem')
            with library.connect(mission) as conn:
                conn.execute('DROP TRIGGER one_way_view_merge')
                conn.execute('ALTER TABLE views DROP COLUMN merged_into')
            self.assertEqual(library.views(mission),[])
            with library.connect(mission) as conn:
                columns={row['name'] for row in conn.execute('PRAGMA table_info(views)')}
                self.assertIn('merged_into',columns)
                trigger=conn.execute("SELECT 1 FROM sqlite_master WHERE type='trigger' AND name='one_way_view_merge'").fetchone()
                self.assertIsNotNone(trigger)

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

    def test_merge_moves_current_members_and_preserves_view_and_run_history(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);library=Library(root/'workspace')
            mission=library.create('Views','Israel','Asia/Jerusalem')
            source=root/'source';source.mkdir()
            for minute in (1604,1605):
                Image.new('RGB',(16,16),(minute%255,20,30)).save(source/f'17052026-{minute}-view-a.jpg','JPEG')
            library.import_folder(mission,source,'camera')
            rows=library.images(mission)
            old=library.create_view(mission,'North','Operator')
            target=library.create_view(mission,'South','Operator')
            library.assign_view(mission,rows[0]['id'],old['id'],'Operator')
            library.assign_view(mission,rows[1]['id'],target['id'],'Operator')
            library.add_view_region(mission,old['id'],'include',[[.1,.1],[.8,.1],[.8,.8],[.1,.8]],'Operator')
            library.add_view_region(mission,target['id'],'exclude',[[.2,.2],[.7,.2],[.7,.7],[.2,.7]],'Operator')
            run=create_run(library,mission,{'adapter':'view-fixture'})

            result=library.merge_views(mission,old['id'],target['id'],'Reviewer')

            self.assertEqual(result['moved_images'],1)
            self.assertTrue(result['source_regions_retained'])
            self.assertTrue(result['target_regions_apply_to_reassigned_images'])
            with library.connect(mission) as conn:
                current={row['image_id']:row['view_id'] for row in conn.execute('''SELECT h.image_id,h.view_id
                    FROM image_view_history h JOIN (SELECT image_id,MAX(revision) revision
                    FROM image_view_history WHERE mission_id=? GROUP BY image_id) latest
                    ON latest.image_id=h.image_id AND latest.revision=h.revision WHERE h.mission_id=?''',
                    (mission,mission))}
            self.assertEqual(current[rows[0]['id']],target['id'])
            self.assertEqual(current[rows[1]['id']],target['id'])
            saved={row['id']:row for row in library.images(mission,run)}
            self.assertEqual(saved[rows[0]['id']]['view_id'],old['id'])
            views={row['id']:row for row in library.views(mission)}
            self.assertEqual(views[old['id']]['merged_into'],target['id'])
            self.assertEqual(views[old['id']]['image_count'],0)
            self.assertEqual(len(library.view_regions(mission,old['id'])),1)
            self.assertEqual(len(library.view_regions(mission,target['id'])),1)
            history=library.view_history(mission,rows[0]['id'])
            self.assertEqual([entry['view_id'] for entry in history],[old['id'],target['id']])
            with library.connect(mission) as conn:
                actions=[row['action'] for row in conn.execute('SELECT action FROM view_events ORDER BY id')]
            self.assertIn('merged_into',actions)
            self.assertIn('merged_from',actions)
            with self.assertRaisesRegex(ValueError,'already merged'):
                library.merge_views(mission,old['id'],target['id'],'Reviewer')
            with self.assertRaisesRegex(ValueError,'Camera view not found'):
                library.assign_view(mission,rows[0]['id'],old['id'],'Reviewer')


if __name__=='__main__':
    unittest.main()
