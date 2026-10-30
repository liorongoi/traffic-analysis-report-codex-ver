import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch
from PIL import Image
import numpy as np
from road_traffic.plate_review import (_connect,_paths,crop_file,context_file,list_crops,
    review,_sha,propose,propose_video_frame,list_crops_page,export_crops_csv,PlateDetector,
    _laplacian_variance)
from road_traffic.store import Library
from road_traffic.video_store import register_video
from road_traffic.video_timeline import VideoTimestamp
from road_traffic.video_decoder import DecodedVideoFrame


class PlateCropReviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();root=Path(self.tmp.name)
        self.output=root/'output';self.output.mkdir()
        self.lib=Library(root/'workspace')
        self.mission=self.lib.create('Plate crops','Israel','Asia/Jerusalem',{
            'input_type':'images','analysis_mode':'license_plate','output_root':str(self.output)})
        source=root/'source';source.mkdir();self.source=source
        Image.new('RGB',(32,24),(30,60,90)).save(source/'17052026-1604-camera.jpg','JPEG')
        self.lib.import_folder(self.mission,source,'camera')
        self.image=self.lib.images(self.mission)[0]
        _,folder,db=_paths(self.lib,self.mission)
        self.crop_id='00000000-0000-4000-8000-000000000001'
        crop=folder/(self.crop_id+'.jpg');Image.new('RGB',(12,5),(255,255,255)).save(crop,'JPEG')
        with _connect(db) as conn:
            conn.execute('''INSERT INTO crops(id,image_id,source_name,source_sha256,crop_file,
                crop_sha256,bbox,confidence,model_sha256,created,mission_country_context,country_context_basis)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)''',(self.crop_id,self.image['id'],self.image['name'],
                self.image['sha256'],crop.name,_sha(crop),json.dumps({'x':2,'y':3,'width':12,'height':5}),
                .8,'a'*64,'2026-10-25T00:00:00+00:00','Israel','mission_configuration'))

    def tearDown(self):self.tmp.cleanup()

    def test_database_connection_closes_after_context(self):
        _,_,db=_paths(self.lib,self.mission)
        with _connect(db) as conn:self.assertEqual(conn.execute('SELECT 1').fetchone()[0],1)
        with self.assertRaises(sqlite3.ProgrammingError):conn.execute('SELECT 1')

    def test_crop_pages_are_bounded_stable_and_do_not_expose_text(self):
        _,folder,db=_paths(self.lib,self.mission)
        with _connect(db) as conn:
            for number in (2,3):
                crop_id=f'00000000-0000-4000-8000-{number:012d}'
                path=folder/(crop_id+'.jpg');Image.new('RGB',(12,5),(number,60,90)).save(path,'JPEG')
                conn.execute('''INSERT INTO crops(id,image_id,source_name,source_sha256,crop_file,crop_sha256,
                    bbox,confidence,model_sha256,created) VALUES (?,?,?,?,?,?,?,?,?,?)''',
                    (crop_id,self.image['id'],f'{number}.jpg',str(number)*64,path.name,_sha(path),'{}',
                    .8,'a'*64,f'2026-10-25T00:00:0{number}+00:00'))
            # Old installations may contain prior OCR columns; API reads must ignore them.
            conn.execute("ALTER TABLE crops ADD COLUMN ocr_text TEXT NOT NULL DEFAULT ''")
            conn.execute("UPDATE crops SET ocr_text='SHOULD NOT LEAK' WHERE id=?",(self.crop_id,))
        first=list_crops_page(self.lib,self.mission,offset=0,limit=2)
        second=list_crops_page(self.lib,self.mission,offset=2,limit=2)
        self.assertEqual((first['total'],len(first['items']),len(second['items'])),(3,2,1))
        self.assertEqual([x['id'] for x in first['items']+second['items']],
            [x['id'] for x in list_crops(self.lib,self.mission)])
        self.assertNotIn('ocr_text',first['items'][0])
        self.assertNotIn('verified_text',first['items'][0])
        with self.assertRaisesRegex(ValueError,'between 1 and 100'):
            list_crops_page(self.lib,self.mission,limit=101)

    def test_crop_is_saved_under_output_and_checksum_verified(self):
        crop=crop_file(self.lib,self.mission,self.crop_id)
        self.assertTrue(crop.is_relative_to((self.output/'plate-crops'/self.mission).resolve()))
        self.assertEqual(_sha(crop),list_crops(self.lib,self.mission)[0]['crop_sha256'])

    def test_crop_decisions_are_append_only_and_revision_checked(self):
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
    def test_image_detection_saves_crop_without_loading_any_recognizer(self,detector):
        _,folder,db=_paths(self.lib,self.mission)
        with _connect(db) as conn:conn.execute('DELETE FROM crops WHERE id=?',(self.crop_id,))
        (folder/(self.crop_id+'.jpg')).unlink()
        detector.return_value.sha='b'*64
        detector.return_value.detect.return_value=[{'box':[.25,.25,.5,.25],'confidence':.91,'model_sha256':'b'*64}]
        rows=propose(self.lib,self.mission,self.image['id'],Path('model-packs/license-plate-yolox'))
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]['decision'],None)
        self.assertEqual(rows[0]['source_name'],self.image['name'])
        self.assertEqual((rows[0]['mission_country_context'],rows[0]['country_context_basis']),
            ('Israel','mission_configuration'))
        self.assertTrue(crop_file(self.lib,self.mission,rows[0]['id']).is_file())
        self.assertNotIn('ocr_text',rows[0])
        repeated=propose(self.lib,self.mission,self.image['id'],Path('model-packs/license-plate-yolox'))
        self.assertEqual(len(repeated),1)

    def test_export_contains_crop_review_lineage_only(self):
        review(self.lib,self.mission,self.crop_id,0,'approved','Reviewer')
        result=export_crops_csv(self.lib,self.mission);self.assertEqual(result['row_count'],1)
        text=Path(result['path']).read_text(encoding='utf-8-sig')
        self.assertIn('crop_review_decision',text)
        self.assertNotIn('ocr_',text.lower());self.assertNotIn('verified_text',text)

    def test_missing_plate_runtime_dependency_fails_clearly(self):
        with patch('importlib.import_module',side_effect=ImportError("No module named 'cv2'")):
            with self.assertRaisesRegex(ValueError,'plate-detection dependency'):
                PlateDetector(Path('model-packs/license-plate-yolox'))

    def test_blur_reference_has_no_automatic_threshold(self):
        class CV:
            COLOR_RGB2GRAY=1;CV_64F=2
            @staticmethod
            def cvtColor(image,_):return image[:,:,0]
            @staticmethod
            def Laplacian(gray,_):return gray.astype(float)
        value=_laplacian_variance(CV,np.dstack([np.arange(100).reshape(10,10)]*3))
        self.assertGreater(value,0)

    def test_video_frame_crops_keep_pts_utc_and_context(self):
        root=Path(self.tmp.name);source_root=root/'videos';source_root.mkdir()
        video=source_root/'camera.mp4';video.write_bytes(b'fixture video bytes')
        output=root/'video-output';output.mkdir()
        video_mission=self.lib.create('Video plate crops','Israel','Asia/Jerusalem',{
            'input_type':'videos','analysis_mode':'license_plate','source_root':str(source_root),
            'output_root':str(output)})
        origin=datetime(2026,5,17,16,4,tzinfo=timezone.utc)
        source=register_video(self.lib,video_mission,'camera.mp4',recording_start_utc=origin,
            origin_provenance='verified recorder metadata')
        class Detector:
            sha='c'*64
            calls=0
            def detect(self,image):
                self.calls+=1
                return [{'box':[.25,.25,.5,.25],'confidence':.9,'model_sha256':self.sha}]
        detector=Detector()
        decoded=DecodedVideoFrame(index=7,timestamp=VideoTimestamp(150,1,30),
            image=np.zeros((24,32,3),dtype=np.uint8))
        ids=propose_video_frame(self.lib,video_mission,source['id'],decoded,detector)
        self.assertEqual(len(ids),1)
        row=list_crops(self.lib,video_mission)[0]
        self.assertEqual((row['source_kind'],row['frame_index'],row['pts']),('video',7,150))
        self.assertEqual((row['time_base_num'],row['time_base_den']),(1,30))
        self.assertEqual((row['mission_country_context'],row['country_context_basis']),
            ('Israel','mission_configuration'))
        self.assertTrue(row['capture_utc'].startswith('2026-05-17T16:04:05'))
        self.assertTrue(context_file(self.lib,video_mission,row['id']).is_file())
        self.assertEqual(propose_video_frame(self.lib,video_mission,source['id'],decoded,detector),[])
        self.assertEqual(detector.calls,1)
        self.assertNotIn('ocr_text',row)

if __name__=='__main__':unittest.main()
