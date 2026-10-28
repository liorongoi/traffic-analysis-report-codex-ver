import json
import csv
from pathlib import Path
import sqlite3
import tempfile
import unittest
from PIL import Image
from road_traffic.plate_review import (_connect, _paths, crop_file, list_crops,
    normalize_plate_text, review, review_text, _sha, propose, ocr_quality_report,
    _normalized_edit_similarity, _edit_distance, list_crops_page,export_crops_csv,PlateDetector)
from road_traffic.plate_review import _laplacian_variance
from unittest.mock import patch
from road_traffic.store import Library
from datetime import datetime, timezone
from road_traffic.video_store import register_video
from road_traffic.video_timeline import VideoTimestamp
from road_traffic.video_decoder import DecodedVideoFrame


class PlateCropReviewTests(unittest.TestCase):
    def test_search_normalization_preserves_order_and_leading_zeroes(self):
        self.assertEqual(normalize_plate_text('٠٠١٢-ب/34\u200f'), '0012ب34')
        self.assertEqual(normalize_plate_text('۱۲٣٤'), '1234')
        self.assertEqual(normalize_plate_text('A-0012'), 'a0012')

    def test_legacy_plate_reviews_gain_search_keys_without_losing_immutable_history(self):
        with tempfile.TemporaryDirectory() as temp:
            db=Path(temp)/'legacy.sqlite3'
            conn=sqlite3.connect(db)
            conn.executescript('''
                PRAGMA foreign_keys=ON;
                CREATE TABLE crops(id TEXT PRIMARY KEY,image_id TEXT NOT NULL,source_name TEXT NOT NULL,
                    source_sha256 TEXT NOT NULL,crop_file TEXT NOT NULL UNIQUE,crop_sha256 TEXT NOT NULL,
                    bbox TEXT NOT NULL,confidence REAL NOT NULL,model_sha256 TEXT NOT NULL,created TEXT NOT NULL,
                    ocr_text TEXT NOT NULL DEFAULT '',ocr_confidence REAL NOT NULL DEFAULT 0,
                    ocr_model_sha256 TEXT NOT NULL DEFAULT '');
                CREATE TABLE reviews(id INTEGER PRIMARY KEY AUTOINCREMENT,crop_id TEXT NOT NULL REFERENCES crops(id),
                    revision INTEGER NOT NULL,decision TEXT NOT NULL,operator TEXT NOT NULL,created TEXT NOT NULL,
                    UNIQUE(crop_id,revision));
                CREATE TABLE text_reviews(id INTEGER PRIMARY KEY AUTOINCREMENT,crop_id TEXT NOT NULL REFERENCES crops(id),
                    revision INTEGER NOT NULL,decision TEXT NOT NULL,verified_text TEXT NOT NULL DEFAULT '',
                    operator TEXT NOT NULL,created TEXT NOT NULL,UNIQUE(crop_id,revision));
                CREATE TRIGGER immutable_plate_text_reviews_update BEFORE UPDATE ON text_reviews
                    BEGIN SELECT RAISE(ABORT,'plate text reviews are append-only'); END;
                INSERT INTO crops(id,image_id,source_name,source_sha256,crop_file,crop_sha256,bbox,
                    confidence,model_sha256,created,ocr_text) VALUES
                    ('crop','image','source.jpg','a','crop.jpg','b','{}',.8,'c','2026-01-01','AB-0012');
                INSERT INTO text_reviews(crop_id,revision,decision,verified_text,operator,created)
                    VALUES ('crop',1,'corrected','٠٠١٢-ب/34','Operator','2026-01-01');
            ''')
            conn.commit();conn.close()
            with _connect(db) as migrated:
                row=migrated.execute('SELECT * FROM text_reviews').fetchone()
                self.assertEqual(row['verified_text'],'٠٠١٢-ب/34')
                self.assertEqual(row['normalized_text'],'0012ب34')
                crop=migrated.execute('SELECT ocr_search_text FROM crops WHERE id=?',('crop',)).fetchone()
                self.assertEqual(crop['ocr_search_text'],'ab0012')
            with _connect(db) as migrated:
                with self.assertRaisesRegex(sqlite3.IntegrityError,'append-only'):
                    migrated.execute('UPDATE text_reviews SET verified_text=? WHERE id=1',('different',))

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
            conn.execute('''INSERT INTO crops(id,image_id,source_name,source_sha256,crop_file,
                crop_sha256,bbox,confidence,model_sha256,created) VALUES (?,?,?,?,?,?,?,?,?,?)''',(
                self.crop_id,self.image['id'],self.image['name'],self.image['sha256'],crop.name,_sha(crop),
                json.dumps({'x':2,'y':3,'width':12,'height':5}),.8,'a'*64,'2026-10-25T00:00:00+00:00'))
    def tearDown(self):self.tmp.cleanup()

    def test_crop_database_connection_closes_after_context(self):
        _,_,db=_paths(self.lib,self.mission)
        with _connect(db) as connection:
            self.assertEqual(connection.execute('SELECT 1').fetchone()[0],1)
        with self.assertRaises(sqlite3.ProgrammingError):
            connection.execute('SELECT 1')

    def test_crop_review_pages_are_bounded_stable_and_searchable(self):
        _,folder,db=_paths(self.lib,self.mission)
        with _connect(db) as conn:
            for number in (2,3):
                crop_id=f'00000000-0000-4000-8000-{number:012d}'
                path=folder/(crop_id+'.jpg');Image.new('RGB',(12,5),(number,60,90)).save(path,'JPEG')
                conn.execute('''INSERT INTO crops(id,image_id,source_name,source_sha256,crop_file,
                    crop_sha256,bbox,confidence,model_sha256,created,ocr_text,ocr_search_text)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?)''',(crop_id,self.image['id'],f'{number}.jpg',
                    str(number)*64,path.name,_sha(path),'{}',.8,'a'*64,f'2026-10-25T00:00:0{number}+00:00',
                    'AB0012' if number==2 else 'CD0034','ab0012' if number==2 else 'cd0034'))
        first=list_crops_page(self.lib,self.mission,offset=0,limit=2)
        second=list_crops_page(self.lib,self.mission,offset=2,limit=2)
        self.assertEqual((first['total'],len(first['items']),len(second['items'])),(3,2,1))
        self.assertEqual([item['id'] for item in first['items']+second['items']],
            [item['id'] for item in list_crops(self.lib,self.mission)])
        searched=list_crops_page(self.lib,self.mission,limit=1,search='ab-0012')
        self.assertEqual((searched['total'],searched['items'][0]['ocr_text']),(1,'AB0012'))
        with self.assertRaisesRegex(ValueError,'between 1 and 100'):
            list_crops_page(self.lib,self.mission,limit=101)

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
    def test_detector_proposals_are_saved_with_unverified_ocr_suggestions(self, detector):
        _,folder,db=_paths(self.lib,self.mission)
        with _connect(db) as conn:conn.execute('DELETE FROM crops WHERE id=?',(self.crop_id,))
        (folder/(self.crop_id+'.jpg')).unlink()
        detector.return_value.sha='b'*64
        detector.return_value.recognizer.read.return_value={'text':'AB1234','confidence':.82,'model_sha256':'d'*64}
        detector.return_value.global_recognizer.read.return_value={'text':'CD5678','confidence':.97,'model_sha256':'e'*64}
        detector.return_value.detect.return_value=[{'box':[.25,.25,.5,.25],'confidence':.91,'model_sha256':'b'*64}]
        rows=propose(self.lib,self.mission,self.image['id'],Path('model-packs/license-plate-yolox'))
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]['decision'],None)
        self.assertEqual(rows[0]['source_name'],self.image['name'])
        self.assertTrue(crop_file(self.lib,self.mission,rows[0]['id']).is_file())
        self.assertEqual(rows[0]['ocr_text'],'AB1234');self.assertEqual(rows[0]['ocr_confidence'],.82)
        self.assertEqual(rows[0]['ocr_alternative_text'],'CD5678')
        self.assertEqual(rows[0]['ocr_alternative_confidence'],.97)
        self.assertEqual(rows[0]['ocr_alternative_model_sha256'],'e'*64)
        self.assertEqual(rows[0]['ocr_alternative_search_text'],'cd5678')
        self.assertEqual(list_crops(self.lib,self.mission,search='CD-5678')[0]['ocr_alternative_text'],'CD5678')
        self.assertEqual(rows[0]['ocr_search_text'],'ab1234')
        self.assertEqual(rows[0]['text_decision'],None)
        repeated=propose(self.lib,self.mission,self.image['id'],Path('model-packs/license-plate-yolox'))
        self.assertEqual(len(repeated),1)

    def test_text_acceptance_and_correction_are_separate_append_only_reviews(self):
        accepted=review_text(self.lib,self.mission,self.crop_id,0,'accepted','٠٠١٢-ب/34','Operator A')
        self.assertEqual(accepted['verified_text'],'٠٠١٢-ب/34')
        self.assertEqual(accepted['normalized_text'],'0012ب34')
        self.assertEqual(len(list_crops(self.lib,self.mission,search='0012-ب/34')),1)
        self.assertEqual(len(list_crops(self.lib,self.mission,search='٠٠١٢-ب/34')),1)
        corrected=review_text(self.lib,self.mission,self.crop_id,1,'corrected','AB1284','Operator B')
        row=list_crops(self.lib,self.mission)[0]
        self.assertEqual((row['text_decision'],row['verified_text'],row['text_revision']),('corrected','AB1284',2))
        self.assertEqual(row['normalized_text'],'ab1284')
        self.assertEqual(list_crops(self.lib,self.mission,search='ab1284')[0]['verified_text'],'AB1284')
        with self.assertRaisesRegex(ValueError,'changed since it was opened'):
            review_text(self.lib,self.mission,self.crop_id,1,'accepted','AB1284','Operator C')
        _,_,db=_paths(self.lib,self.mission)
        with _connect(db) as conn:
            with self.assertRaisesRegex(sqlite3.IntegrityError,'append-only'):
                conn.execute('UPDATE text_reviews SET verified_text=? WHERE crop_id=?',('OTHER',self.crop_id))

    def test_plate_ocr_report_scores_only_approved_human_labels_and_discloses_limits(self):
        review(self.lib,self.mission,self.crop_id,0,'approved','Reviewer')
        review_text(self.lib,self.mission,self.crop_id,0,'corrected','AB-0012','Reviewer')
        _,_,db=_paths(self.lib,self.mission)
        with _connect(db) as conn:
            conn.execute('''UPDATE crops SET ocr_text=?,ocr_model_sha256=?,ocr_alternative_text=?,
                ocr_alternative_model_sha256=? WHERE id=?''',('AB0012','a'*64,'AB0013','b'*64,self.crop_id))
        report=ocr_quality_report(self.lib,self.mission)
        self.assertEqual(report['country'],'Israel')
        self.assertEqual(report['reviewed_crops'],1)
        self.assertIn('not a protected held-out evaluation',report['scope'])
        primary=next(row for row in report['models'] if row['candidate']=='primary')
        alternate=next(row for row in report['models'] if row['candidate']=='global_latin')
        self.assertEqual((primary['exact_matches'],primary['exact_match_rate']),(1,1.0))
        self.assertEqual(alternate['exact_matches'],0)
        self.assertAlmostEqual(alternate['mean_normalized_edit_similarity'],5/6)
        self.assertAlmostEqual(alternate['character_error_rate'],1/6)
        self.assertEqual((alternate['edit_errors'],alternate['reference_characters']),(1,6))
        # A second observation of the same plate must not count as another
        # independent string in the research-inspired grouped summary.
        with _connect(db) as conn:
            conn.execute('''INSERT INTO crops(id,image_id,source_name,source_sha256,crop_file,crop_sha256,bbox,
                confidence,model_sha256,created,ocr_text,ocr_model_sha256) SELECT ?,image_id,source_name,
                source_sha256,?, ?,bbox,confidence,model_sha256,created,'WRONG','a'||substr(ocr_model_sha256,2)
                FROM crops WHERE id=?''',('00000000-0000-4000-8000-000000000002','second.jpg','0'*64,self.crop_id))
            second='00000000-0000-4000-8000-000000000002'
            conn.execute("INSERT INTO reviews(crop_id,revision,decision,operator,created) VALUES (?,1,'approved','Reviewer','now')",(second,))
            conn.execute("INSERT INTO text_reviews(crop_id,revision,decision,verified_text,normalized_text,operator,created) VALUES (?,1,'corrected','AB-0012','ab0012','Reviewer','now')",(second,))
        report=ocr_quality_report(self.lib,self.mission)
        primary=next(row for row in report['models'] if row['candidate']=='primary')
        self.assertEqual(primary['reviewed_crops'],2)
        self.assertEqual(primary['string_grouped']['reviewed_crops'],1)
        self.assertEqual(primary['string_grouped']['exact_matches'],0)
        self.assertEqual(primary['unique_verified_strings'],1)
        self.assertAlmostEqual(_normalized_edit_similarity('۱۲-AB','12AC'),.75)
        self.assertEqual(_edit_distance('A','ABCDEF'),5)

    def test_plate_ocr_report_excludes_pending_rejected_or_unreadable_text(self):
        report=ocr_quality_report(self.lib,self.mission)
        self.assertEqual(report['reviewed_crops'],0)
        self.assertEqual(report['models'],[])

    def test_plate_review_export_preserves_unicode_and_lineage_in_unique_csv(self):
        review(self.lib,self.mission,self.crop_id,0,'approved','Reviewer')
        review_text(self.lib,self.mission,self.crop_id,0,'corrected','٠٠١٢-ب/34','Reviewer')
        first=export_crops_csv(self.lib,self.mission)
        with Path(first['path']).open(encoding='utf-8-sig',newline='') as stream:
            row=next(csv.DictReader(stream))
        self.assertEqual(first['row_count'],1)
        self.assertEqual(row['verified_text'],'٠٠١٢-ب/34')
        self.assertEqual(row['normalized_text'],'0012ب34')
        self.assertEqual(row['text_decision'],'corrected')
        self.assertEqual(row['crop_review_operator'],'Reviewer')
        self.assertEqual(row['crop_sha256'],_sha(Path(self.output)/'plate-crops'/self.mission/(self.crop_id+'.jpg')))
        self.assertEqual(len(first['sha256']),64)
        second=export_crops_csv(self.lib,self.mission)
        self.assertNotEqual(first['path'],second['path'])

    def test_plate_detector_reports_missing_offline_runtime_dependency_clearly(self):
        with patch('builtins.__import__',side_effect=ModuleNotFoundError("No module named 'cv2'",name='cv2')):
            with self.assertRaisesRegex(ValueError,r"offline runtime is missing a required plate-detection dependency \(module 'cv2'\)"):
                PlateDetector(Path('unused-model-pack'))

    def test_plate_blur_reference_uses_laplacian_variance_without_thresholding(self):
        import numpy as np
        class FakeCV:
            COLOR_RGB2GRAY=1
            CV_64F=2
            @staticmethod
            def cvtColor(image,code):return image
            @staticmethod
            def Laplacian(image,dtype):return image
        crop=np.array([[0,1,2],[1,2,3],[2,3,4]],dtype=np.float32)
        self.assertAlmostEqual(_laplacian_variance(FakeCV,np,crop),float(np.var(crop)))
        self.assertIsNone(_laplacian_variance(FakeCV,np,np.empty((0,0))))

    def test_reviews_reject_missing_operator_and_invalid_decision(self):
        with self.assertRaisesRegex(ValueError,'Operator'):
            review(self.lib,self.mission,self.crop_id,0,'approved',' ')
        with self.assertRaisesRegex(ValueError,'Choose approved or rejected'):
            review(self.lib,self.mission,self.crop_id,0,'read-text','Operator')

    def test_video_frame_crops_keep_pts_utc_context_and_ocr_fields(self):
        import numpy as np
        from road_traffic.plate_review import propose_video_frame, context_file
        root=Path(self.tmp.name);source_root=root/'videos';source_root.mkdir()
        video=source_root/'camera.mp4';video.write_bytes(b'fixture video bytes')
        output=root/'video-output';output.mkdir()
        video_mission=self.lib.create('Video plate crops','Israel','Asia/Jerusalem',{
            'input_type':'videos','analysis_mode':'license_plate','source_root':str(source_root),
            'output_root':str(output)})
        origin=datetime(2026,5,17,16,4,tzinfo=timezone.utc)
        source=register_video(self.lib,video_mission,'camera.mp4',recording_start_utc=origin,
            origin_provenance='verified recorder metadata')
        class Recognizer:
            def read(self,image):return {'text':'','confidence':0.0,'model_sha256':'d'*64}
        class Detector:
            sha='c'*64
            calls=0
            recognizer=Recognizer()
            global_recognizer=Recognizer()
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
        self.assertTrue(row['capture_utc'].startswith('2026-05-17T16:04:05'))
        self.assertTrue(context_file(self.lib,video_mission,row['id']).is_file())
        self.assertEqual(propose_video_frame(self.lib,video_mission,source['id'],decoded,detector),[])
        self.assertEqual(detector.calls,1)
        self.assertEqual(row['ocr_text'],'');self.assertEqual(row['text_decision'],None)


if __name__=='__main__':unittest.main()
