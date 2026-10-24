import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from PIL import Image
from unittest.mock import patch
from road_traffic.store import Library
from road_traffic.inference_store import create_run,enrich,runs
from road_traffic.detector import iou,suppress,illumination,night_proposals,model_pack


class DetectorMathTests(unittest.TestCase):
    def test_duplicate_vehicle_classes_do_not_double_count(self):
        a={'box':[.1,.1,.2,.2],'label':'car','score':.8}
        b={'box':[.11,.11,.2,.2],'label':'truck','score':.6}
        self.assertGreater(iou(a['box'],b['box']),.7)
        self.assertEqual(suppress([b,a]),[a])
    def test_bright_monochrome_night_is_not_skipped(self):
        import numpy as np
        self.assertTrue(illumination(np.full((30,30,3),120,np.uint8))['low_light'])
        colorful=np.zeros((30,30,3),np.uint8);colorful[:,:,1]=220;colorful[:,:,0]=120
        self.assertFalse(illumination(colorful)['low_light'])
    def test_single_to_pair_is_not_two_confirmed_vehicles(self):
        import numpy as np
        rgb=np.zeros((100,200,3),np.uint8);rgb[50:53,80:83]=255
        single,info=night_proposals(rgb,None,[])
        self.assertEqual(len(single),1);self.assertEqual(single[0]['pairing'],'single')
        rgb[50:53,90:93]=255
        pair,info=night_proposals(rgb,None,[])
        self.assertEqual(len(pair),1);self.assertEqual(pair[0]['pairing'],'hypothesis')
        self.assertEqual(len(pair[0]['light_components']),2)
        self.assertEqual(pair[0]['kind'],'light')
    def test_persistent_lights_and_detector_overlap_excluded(self):
        import numpy as np
        rgb=np.zeros((100,200,3),np.uint8);rgb[50:53,80:83]=255
        self.assertEqual(night_proposals(rgb,np.ones((100,200),np.uint8),[])[0],[])
        box={'box':[.3,.4,.2,.2]}
        self.assertEqual(night_proposals(rgb,None,[box])[0],[])
    def test_missing_model_fails_without_fetch(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(FileNotFoundError):model_pack(root)


class InferenceLedgerTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();root=Path(self.temp.name)
        self.lib=Library(root/'workspace');self.m=self.lib.create('Test','Israel','Asia/Jerusalem')
        source=root/'source';source.mkdir();Image.new('RGB',(8,8),(25,50,75)).save(source/'17052026-1604-test.jpg','JPEG')
        self.lib.import_folder(self.m,source,'camera');self.image=self.lib.images(self.m)[0]['id']
        self.run=create_run(self.lib,self.m,{'model':'explicit-test-double'})
        self.pred=[{'kind':'candidate','label':'car','box':[.1,.1,.2,.2],'score':.7,'prediction_id':'fixture-1','origin':'model'}]
        with self.lib.connect(self.m) as conn:
            conn.execute('INSERT INTO predictions VALUES (?,?,?,?,?,?)',(self.run,self.image,json.dumps(self.pred),'{}',.1,None))
            conn.execute("UPDATE inference_runs SET status='completed',done=1 WHERE id=?",(self.run,))
    def tearDown(self):self.temp.cleanup()
    def test_model_proposals_appear_without_becoming_reviewed(self):
        row=self.lib.images(self.m)[0]
        self.assertEqual(row['revision'],0);self.assertEqual(row['review_status'],'unreviewed')
        self.assertEqual(json.loads(row['annotations']),self.pred)
    def test_correction_survives_restart_and_keeps_original(self):
        changed=[{**self.pred[0],'kind':'vehicle','label':'truck','reviewed':True}]
        self.lib.review(self.m,self.image,0,changed,'Test operator',prediction_run=self.run)
        reopened=Library(self.lib.root);row=reopened.images(self.m)[0]
        self.assertEqual(json.loads(row['annotations'])[0]['label'],'truck')
        self.assertEqual(row['predictions'][0]['label'],'car')
        self.assertEqual(row['review_prediction_run'],self.run)
        with self.lib.connect(self.m) as conn:
            changes=json.loads(conn.execute('SELECT changes FROM review_inference').fetchone()[0])
            self.assertEqual(changes['before'],self.pred);self.assertEqual(changes['after'],changed)
            with self.assertRaises(sqlite3.IntegrityError):conn.execute('DELETE FROM predictions')
    def test_rejection_preserves_original(self):
        changed=[{**self.pred[0],'rejected':True}]
        self.lib.review(self.m,self.image,0,changed,'Test operator',prediction_run=self.run)
        row=self.lib.images(self.m)[0]
        self.assertTrue(json.loads(row['annotations'])[0]['rejected'])
        self.assertNotIn('rejected',row['predictions'][0])
    def test_failed_lineage_save_is_atomic(self):
        with self.assertRaises(ValueError):self.lib.review(self.m,self.image,0,self.pred,'Operator',prediction_run='not-a-real-run')
        self.assertEqual(self.lib.images(self.m)[0]['revision'],0)
    def test_new_run_does_not_replace_review(self):
        self.lib.review(self.m,self.image,0,[],'Operator',prediction_run=self.run)
        newer=create_run(self.lib,self.m,{'model':'second-fixture'})
        with self.lib.connect(self.m) as conn:
            conn.execute('INSERT INTO predictions VALUES (?,?,?,?,?,?)',(newer,self.image,'[]','{}',.1,None))
            conn.execute("UPDATE inference_runs SET status='completed' WHERE id=?",(newer,))
        row=self.lib.images(self.m)[0]
        self.assertEqual(row['prediction_run'],self.run);self.assertEqual(row['annotations'],'[]')
        selected=self.lib.images(self.m,newer)[0]
        self.assertEqual(selected['review_prediction_run'],self.run);self.assertEqual(selected['prediction_run'],newer)
    def test_duplicate_worker_blocked(self):
        create_run(self.lib,self.m,{})
        with self.assertRaises(ValueError):create_run(self.lib,self.m,{})
    def test_mission_run_isolation(self):
        other=self.lib.create('Other','Latvia','Europe/Riga')
        with self.assertRaises(ValueError):self.lib.images(other,self.run)
