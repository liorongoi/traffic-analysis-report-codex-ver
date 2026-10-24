import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from PIL import Image

from road_traffic.inference_store import create_run
from road_traffic.regions import filter_proposals, point_in_polygon, validate_polygon
from road_traffic.store import Library


class RegionGeometryTests(unittest.TestCase):
    def test_polygon_validation_and_boundary_inclusion(self):
        polygon=validate_polygon([[.1,.1],[.9,.1],[.9,.9],[.1,.9]])
        self.assertTrue(point_in_polygon((.1,.5),polygon))
        self.assertTrue(point_in_polygon((.5,.5),polygon))
        self.assertFalse(point_in_polygon((.95,.5),polygon))
        with self.assertRaisesRegex(ValueError,'normalized'):
            validate_polygon([[0,0],[1.1,0],[1,1]])
        with self.assertRaisesRegex(ValueError,'area'):
            validate_polygon([[0,0],[.5,.5],[1,1]])

    def test_inclusion_union_and_exclusion_precedence_filter_proposals(self):
        include={'kind':'include','polygon':[[.1,.1],[.9,.1],[.9,.9],[.1,.9]]}
        exclusion={'kind':'exclude','polygon':[[.4,.4],[.6,.4],[.6,.6],[.4,.6]]}
        proposals=[{'box':[.2,.2,.1,.1]},{'box':[.45,.45,.1,.1]},{'box':[.95,.95,.02,.02]}]
        kept,summary=filter_proposals(proposals,[include,exclusion])
        self.assertEqual(kept,[proposals[0]])
        self.assertEqual(summary['excluded'],2)
        self.assertTrue(summary['exclusion_overrides_inclusion'])


class RegionHistoryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.library=Library(self.root/'workspace')
        self.mission=self.library.create('Regions','Israel','Asia/Jerusalem')
        source=self.root/'source';source.mkdir()
        Image.new('RGB',(8,8),(20,40,60)).save(source/'17052026-1604-camera-a.jpg','JPEG')
        self.library.import_folder(self.mission,source,'camera')
        self.image=self.library.images(self.mission)[0]
        self.view=self.library.create_view(self.mission,'Entrance','Operator A')
        self.library.assign_view(self.mission,self.image['id'],self.view['id'],'Operator A')

    def tearDown(self):
        self.temp.cleanup()

    def test_region_edits_are_append_only_and_deleted_regions_disappear(self):
        polygon=[[.1,.1],[.9,.1],[.9,.9],[.1,.9]]
        created=self.library.add_view_region(self.mission,self.view['id'],'include',polygon,'Operator A')
        updated=self.library.update_view_region(self.mission,created['region_key'],'exclude',[[.2,.2],[.8,.2],[.8,.8],[.2,.8]],'Operator B')
        deleted=self.library.delete_view_region(self.mission,created['region_key'],'Operator C')
        self.assertEqual((created['revision'],updated['revision'],deleted['revision']),(1,2,3))
        self.assertEqual(self.library.view_regions(self.mission,self.view['id']),[])
        history=self.library.view_region_history(self.mission,self.view['id'])
        self.assertEqual([row['action'] for row in history],['upsert','upsert','delete'])
        self.assertEqual([row['operator'] for row in history],['Operator A','Operator B','Operator C'])
        with self.library.connect(self.mission) as conn:
            with self.assertRaisesRegex(sqlite3.IntegrityError,'immutable'):
                conn.execute('UPDATE view_region_history SET operator=? WHERE region_key=?',('tamper',created['region_key']))

    def test_inference_run_keeps_region_snapshot_after_later_edits(self):
        polygon=[[.1,.1],[.9,.1],[.9,.9],[.1,.9]]
        region=self.library.add_view_region(self.mission,self.view['id'],'include',polygon,'Operator A')
        run=create_run(self.library,self.mission,{'adapter':'region-fixture'})
        self.library.update_view_region(self.mission,region['region_key'],'exclude',[[.2,.2],[.8,.2],[.8,.8],[.2,.8]],'Operator B')
        with self.library.connect(self.mission) as conn:
            config=json.loads(conn.execute('SELECT config FROM inference_runs WHERE id=?',(run,)).fetchone()['config'])
        snapshot=config['view_regions_snapshot'][self.view['id']]
        self.assertEqual(snapshot[0]['revision'],1)
        self.assertEqual(snapshot[0]['kind'],'include')
        self.assertEqual(snapshot[0]['polygon'],polygon)


if __name__=='__main__':
    unittest.main()
