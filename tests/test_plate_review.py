import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from PIL import Image
from road_traffic.plate_review import _connect, _paths, crop_file, list_crops, review, _sha, propose
from unittest.mock import patch
from road_traffic.store import Library


class PlateCropReviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();root=Path(self.tmp.name)
        self.output=root/'output';self.output.mkdir()
        self.lib=Library(root/'workspace')
        self.mission=self.lib.create('Plate crops','Israel','Asia/Jerusalem',{
            'input_type':'images','analysis_mode':'license_plate','output_root':str(self.output)})
        source=root/'source';source.mkdir()
        Image.new('RGB',(32,24),(30,60,90)).save(source/'17052026-1604-camera.jpg','JPEG')
        self.lib.import_folder(self.mission,source,'camera')
        self.image=self.lib.images(self.mission)[0]
        _,folder,db=_paths(self.lib,self.mission)
        self.crop_id='00000000-0000-4000-8000-000000000001'
        crop=folder/(self.crop_id+'.jpg');Image.new('RGB',(12,5),(255,255,255)).save(crop,'JPEG')
        with _connect(db) as conn:
            conn.execute('INSERT INTO crops VALUES (?,?,?,?,?,?,?,?,?,?)',(
                self.crop_id,self.image['id'],self.image['name'],self.image['sha256'],crop.name,_sha(crop),
                json.dumps({'x':2,'y':3,'width':12,'height':5}),.8,'a'*64,'2026-10-25T00:00:00+00:00'))
    def tearDown(self):self.tmp.cleanup()

    def test_crop_is_saved_under_configured_output_and_checksum_verified(self):
        crop=crop_file(self.lib,self.mission,self.crop_id)
        self.assertTrue(crop.is_relative_to((self.output/'plate-crops'/self.mission).resolve()))
        self.assertEqual(_sha(crop),list_crops(self.lib,self.mission)[0]['crop_sha256'])

    def test_human_decisions_are_append_only_and_revision_checked(self):
        approved=review(self.lib,self.mission,self.crop_id,0,'approved','Operator A')
        self.assertEqual(approved['revision'],1)
        self.assertEqual(list_crops(self.lib,self.mission)[0]['decision'],'approved')
        rejected=review(self.lib,self.mission,self.crop_id,1,'rejected','Operator B')
        self.assertEqual(rejected['revision'],2)
        with self.assertRaisesRegex(ValueError,'changed since it was opened'):
            review(self.lib,self.mission,self.crop_id,1,'approved','Operator C')
        _,_,db=_paths(self.lib,self.mission)
        with _connect(db) as conn:
            self.assertEqual(conn.execute('SELECT COUNT(*) FROM reviews').fetchone()[0],2)
            with self.assertRaisesRegex(sqlite3.IntegrityError,'append-only'):
                conn.execute('UPDATE reviews SET decision=? WHERE crop_id=?',('approved',self.crop_id))

    @patch('road_traffic.plate_review.PlateDetector')
    def test_detector_proposals_are_saved_as_crop_images_without_text_fields(self, detector):
        _,folder,db=_paths(self.lib,self.mission)
        with _connect(db) as conn:conn.execute('DELETE FROM crops WHERE id=?',(self.crop_id,))
        (folder/(self.crop_id+'.jpg')).unlink()
        detector.return_value.sha='b'*64
        detector.return_value.detect.return_value=[{'box':[.25,.25,.5,.25],'confidence':.91,'model_sha256':'b'*64}]
        rows=propose(self.lib,self.mission,self.image['id'],Path('model-packs/license-plate-yolox'))
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]['decision'],None)
        self.assertEqual(rows[0]['source_name'],self.image['name'])
        self.assertTrue(crop_file(self.lib,self.mission,rows[0]['id']).is_file())
        self.assertNotIn('text',rows[0]);self.assertNotIn('ocr',rows[0])
        repeated=propose(self.lib,self.mission,self.image['id'],Path('model-packs/license-plate-yolox'))
        self.assertEqual(len(repeated),1)

    def test_reviews_reject_missing_operator_and_invalid_decision(self):
        with self.assertRaisesRegex(ValueError,'Operator'):
            review(self.lib,self.mission,self.crop_id,0,'approved',' ')
        with self.assertRaisesRegex(ValueError,'Choose approved or rejected'):
            review(self.lib,self.mission,self.crop_id,0,'read-text','Operator')


if __name__=='__main__':unittest.main()
