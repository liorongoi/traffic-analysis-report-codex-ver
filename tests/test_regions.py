import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from PIL import Image

from road_traffic.inference_store import create_run,roi_drift,roi_review_state,create_roi_rerun
from road_traffic.regions import classify_road_relation, filter_proposals, mask_analysis_regions, point_in_polygon, validate_polygon
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

    def test_road_surface_is_reference_geometry_and_does_not_filter_detections(self):
        road={'kind':'include','role':'road_surface','polygon':[[.2,.6],[.8,.6],[.95,1],[.05,1]]}
        rows=[{'kind':'vehicle','box':[.4,.55,.1,.12],'evidence':'yolox-s'},
              {'kind':'vehicle','box':[.4,.47,.1,.1],'evidence':'yolox-s'},
              {'kind':'vehicle','box':[.9,.1,.05,.1],'evidence':'yolox-s'}]
        kept,summary=filter_proposals(rows,[road])
        self.assertEqual(kept,rows)
        self.assertEqual([row['road_relation']['label'] for row in kept],['on_road','near_road','off_road'])
        self.assertEqual(summary['road_surface_regions'],1)
        self.assertEqual(classify_road_relation([.1,.1,.1,.1],[])['label'],'unknown')
        unreferenced=[{'kind':'candidate','box':[.1,.1,.1,.1],'evidence':'yolox-s'}]
        filter_proposals(unreferenced,[])
        self.assertEqual(unreferenced[0]['road_relation']['label'],'unknown')

    def test_analysis_inclusions_mask_model_input_but_road_surface_does_not(self):
        import numpy as np
        class CV2Stub:
            @staticmethod
            def fillPoly(mask, polygons, value):
                points=polygons[0]
                mask[points[:,1].min():points[:,1].max()+1,points[:,0].min():points[:,0].max()+1]=value
        rgb=np.full((20,20,3),220,dtype=np.uint8)
        polygon=[[.25,.25],[.75,.25],[.75,.75],[.25,.75]]
        analysis=[{'kind':'include','role':'analysis','polygon':polygon}]
        masked,summary=mask_analysis_regions(rgb,analysis,CV2Stub)
        self.assertTrue(summary['applied'])
        self.assertTrue(np.all(masked[0,0]==114))
        self.assertTrue(np.all(masked[10,10]==220))
        unchanged,summary=mask_analysis_regions(rgb,[{'kind':'include','role':'road_surface','polygon':polygon}],CV2Stub)
        self.assertFalse(summary['applied'])
        self.assertTrue(np.array_equal(unchanged,rgb))


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

    def test_named_regions_are_unique_and_names_remain_in_append_only_history(self):
        polygon=[[.1,.1],[.9,.1],[.9,.9],[.1,.9]]
        region=self.library.add_view_region(self.mission,self.view['id'],'include',polygon,'Operator A','Northbound lane')
        renamed=self.library.update_view_region(self.mission,region['region_key'],'include',polygon,'Operator B','Bus bay')
        self.assertEqual((region['name'],renamed['name']),('Northbound lane','Bus bay'))
        self.library.add_view_region(self.mission,self.view['id'],'exclude',polygon,'Operator C','Loading area')
        with self.assertRaisesRegex(ValueError,'unique within this camera view'):
            self.library.add_view_region(self.mission,self.view['id'],'include',polygon,'Operator D','BUS BAY')
        history=self.library.view_region_history(self.mission,self.view['id'])
        self.assertEqual({item['name'] for item in history},{'Northbound lane','Bus bay','Loading area'})
        self.assertEqual([(item['revision'],item['name']) for item in history if item['region_key']==region['region_key']],
            [(1,'Northbound lane'),(2,'Bus bay')])

    def test_old_region_table_gets_additive_name_column_migration(self):
        with self.library.connect(self.mission) as conn:
            conn.execute('ALTER TABLE view_region_history DROP COLUMN name')
            conn.execute('ALTER TABLE view_region_history DROP COLUMN role')
        with self.library.connect(self.mission) as conn:
            columns={row['name'] for row in conn.execute('PRAGMA table_info(view_region_history)')}
        self.assertIn('name',columns)
        self.assertIn('role',columns)
        created=self.library.add_view_region(self.mission,self.view['id'],'include',
            [[.1,.1],[.9,.1],[.9,.9],[.1,.9]],'Operator','Legacy migration lane')
        self.assertEqual(created['name'],'Legacy migration lane')
        self.assertEqual(created['role'],'analysis')

    def test_road_surface_role_is_versioned_and_must_use_include_geometry(self):
        polygon=[[.1,.1],[.9,.1],[.9,.9],[.1,.9]]
        with self.assertRaisesRegex(ValueError,'inclusion mode'):
            self.library.add_view_region(self.mission,self.view['id'],'exclude',polygon,'Operator','Road',role='road_surface')
        region=self.library.add_view_region(self.mission,self.view['id'],'include',polygon,'Operator','Road',role='road_surface')
        updated=self.library.update_view_region(self.mission,region['region_key'],'include',polygon,'Operator B','Main road')
        self.assertEqual((region['role'],updated['role']),('road_surface','road_surface'))
        self.assertEqual([row['role'] for row in self.library.view_region_history(self.mission,self.view['id'])],
            ['road_surface','road_surface'])

    def test_inference_run_keeps_region_snapshot_after_later_edits(self):
        polygon=[[.1,.1],[.9,.1],[.9,.9],[.1,.9]]
        region=self.library.add_view_region(self.mission,self.view['id'],'include',polygon,'Operator A')
        run=create_run(self.library,self.mission,{'adapter':'region-fixture'})
        self.assertEqual(roi_drift(self.library,self.mission,run),[])
        self.library.update_view_region(self.mission,region['region_key'],'exclude',[[.2,.2],[.8,.2],[.8,.8],[.2,.8]],'Operator B')
        self.assertEqual(roi_drift(self.library,self.mission,run),[{'view_id':self.view['id'],'view_name':'Entrance'}])
        with self.library.connect(self.mission) as conn:
            config=json.loads(conn.execute('SELECT config FROM inference_runs WHERE id=?',(run,)).fetchone()['config'])
        snapshot=config['view_regions_snapshot'][self.view['id']]
        self.assertEqual(snapshot[0]['revision'],1)
        self.assertEqual(snapshot[0]['kind'],'include')
        self.assertEqual(snapshot[0]['name'],'Include region 1')
        self.assertEqual(snapshot[0]['polygon'],polygon)

    def test_unrelated_view_roi_change_does_not_flag_run(self):
        other=self.library.create_view(self.mission,'Other camera','Operator A')
        region=self.library.add_view_region(self.mission,other['id'],'include',[[.1,.1],[.9,.1],[.9,.9],[.1,.9]],'Operator A')
        run=create_run(self.library,self.mission,{'adapter':'roi-scope-fixture'})
        self.library.update_view_region(self.mission,region['region_key'],'exclude',[[.2,.2],[.8,.2],[.8,.8],[.2,.8]],'Operator B')
        self.assertEqual(roi_drift(self.library,self.mission,run),[])

    def test_roi_acknowledgment_creates_same_input_rerun_and_tracks_later_changes(self):
        polygon=[[.1,.1],[.9,.1],[.9,.9],[.1,.9]]
        region=self.library.add_view_region(self.mission,self.view['id'],'include',polygon,'Operator A')
        source=create_run(self.library,self.mission,{'adapter':'roi-fixture','threshold':.2})
        with self.library.connect(self.mission) as conn:
            conn.execute("UPDATE inference_runs SET status='completed',stage='finished',done=1 WHERE id=?",(source,))
        self.library.update_view_region(self.mission,region['region_key'],'exclude',[[.2,.2],[.8,.2],[.8,.8],[.2,.8]],'Operator B')
        state=roi_review_state(self.library,self.mission,source)
        self.assertTrue(state['review_required'])
        rerun=create_roi_rerun(self.library,self.mission,source,'Operator C')
        self.assertEqual((rerun['rerun_of'],rerun['status'],rerun['total']),(source,'queued',1))
        self.assertFalse(roi_review_state(self.library,self.mission,source)['review_required'])
        self.assertEqual(roi_review_state(self.library,self.mission,source)['acknowledgment']['operator'],'Operator C')
        with self.library.connect(self.mission) as conn:
            old=conn.execute('SELECT config FROM inference_runs WHERE id=?',(source,)).fetchone()
            new=conn.execute('SELECT config FROM inference_runs WHERE id=?',(rerun['run'],)).fetchone()
            old_inputs=conn.execute('SELECT image_id,sha256 FROM inference_inputs WHERE inference_id=?',(source,)).fetchall()
            new_inputs=conn.execute('SELECT image_id,sha256 FROM inference_inputs WHERE inference_id=?',(rerun['run'],)).fetchall()
            self.assertEqual(old_inputs,new_inputs)
            self.assertEqual(json.loads(old['config'])['view_regions_snapshot'][self.view['id']][0]['revision'],1)
            self.assertEqual(json.loads(new['config'])['view_regions_snapshot'][self.view['id']][0]['revision'],2)
            with self.assertRaisesRegex(sqlite3.IntegrityError,'immutable'):
                conn.execute('UPDATE roi_rerun_acknowledgments SET operator=? WHERE rerun_run=?',('tampered',rerun['run']))
        with self.assertRaisesRegex(ValueError,'already has an acknowledged rerun'):
            create_roi_rerun(self.library,self.mission,source,'Operator D')
        self.library.update_view_region(self.mission,region['region_key'],'exclude',[[.3,.3],[.7,.3],[.7,.7],[.3,.7]],'Operator D')
        self.assertTrue(roi_review_state(self.library,self.mission,source)['review_required'])


if __name__=='__main__':
    unittest.main()
