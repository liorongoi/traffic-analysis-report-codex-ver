import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from fractions import Fraction
import tempfile
import unittest

from road_traffic.store import Library, now
from road_traffic.video_clips import export_event_clip, export_file, exports
from road_traffic.video_store import create_run, register_video


def write_video(path):
    script = r"""import av, numpy as np, sys
from fractions import Fraction
container=av.open(sys.argv[1],mode='w',format='mp4')
stream=container.add_stream('libx264',rate=5)
stream.width=stream.height=64
stream.pix_fmt='yuv420p'
stream.time_base=Fraction(1,5)
pts=[0,1,4,5,9,10,14,15,19,20]
for index,value in enumerate(pts):
 image=np.full((64,64,3),index*20,dtype=np.uint8)
 frame=av.VideoFrame.from_ndarray(image,format='rgb24')
 frame.pts=value
 frame.time_base=Fraction(1,5)
 for packet in stream.encode(frame): container.mux(packet)
for packet in stream.encode(None): container.mux(packet)
container.close()
"""
    subprocess.run([sys.executable,'-c',script,str(path)],check=True,env=os.environ.copy())


class VideoClipTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source_root = self.root / 'source'
        self.source_root.mkdir()
        self.output_root = self.root / 'output'
        self.output_root.mkdir()
        self.video_path = self.source_root / 'event.mp4'
        write_video(self.video_path)
        self.library = Library(self.root / 'workspace')
        self.mission = self.library.create('Clip mission', 'Israel', 'Asia/Jerusalem', {
            'input_type':'videos','analysis_mode':'video_traffic',
            'source_root':str(self.source_root),'output_root':str(self.output_root),
        })
        self.source = register_video(self.library, self.mission, 'event.mp4')
        self.run = create_run(self.library, self.mission, self.source['id'], {
            'detector':{'version':'test','sha256':'abc'},
            'tracker_policy':{'max_gap_seconds':[1,2],'minimum_iou':.2,
                'max_center_distance':.2,'association_margin':.05,'compatible_label_groups':[]},
        })
        self.event_id = 'event-clip-fixture'
        with self.library.connect(self.mission) as conn:
            conn.execute('''INSERT INTO video_events(id,mission_id,run_id,source_id,track_id,
                first_pts,first_time_base_num,first_time_base_den,last_pts,last_time_base_num,
                last_time_base_den,first_offset_num,first_offset_den,last_offset_num,last_offset_den,
                first_utc,last_utc,labels,observation_count,association_uncertain,uncertainty_reasons,created)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                (self.event_id,self.mission,self.run['id'],self.source['id'],'track-a',
                 8192,1,10240,30720,1,10240,4,5,3,1,None,None,json.dumps(['car']),6,0,'[]',now()))

    def tearDown(self):
        self.temp.cleanup()

    def test_clip_export_uses_explicit_context_and_preserves_lineage(self):
        clip = export_event_clip(self.library,self.mission,self.event_id,.2,.2,'Operator')
        path = export_file(self.library,self.mission,clip['id'])
        self.assertEqual(clip['status'],'completed')
        self.assertEqual(clip['frame_count'],6)
        self.assertEqual(clip['pre_seconds'],.2)
        self.assertEqual(clip['post_seconds'],.2)
        self.assertEqual(clip['source_sha256'],self.source['sha256'])
        self.assertEqual(clip['operator'],'Operator')
        self.assertTrue(path.is_relative_to(self.output_root.resolve()))
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),clip['sha256'])
        inspect_script = r"""import av, json, sys
container=av.open(sys.argv[1])
video=container.streams.video[0]
timestamps=[float(frame.pts*frame.time_base) for frame in container.decode(video)]
print(json.dumps(timestamps))
"""
        result = subprocess.run([sys.executable,'-c',inspect_script,str(path)],
            check=True,capture_output=True,text=True,env=os.environ.copy())
        timestamps = [Fraction(str(value)) for value in json.loads(result.stdout)]
        self.assertEqual(len(timestamps),6)
        self.assertEqual([b-a for a,b in zip(timestamps,timestamps[1:])],
            [Fraction(1,5),Fraction(4,5),Fraction(1,5),Fraction(4,5),Fraction(1,5)])
        self.assertEqual(exports(self.library,self.mission)[0]['id'],clip['id'])

    def test_clip_export_requires_explicit_operator_and_context_bounds(self):
        with self.assertRaisesRegex(ValueError,'Operator is required'):
            export_event_clip(self.library,self.mission,self.event_id,.2,.2,' ')
        with self.assertRaisesRegex(ValueError,'between 0 and 600'):
            export_event_clip(self.library,self.mission,self.event_id,601,.2,'Operator')

    def test_download_refuses_tampered_export(self):
        clip = export_event_clip(self.library,self.mission,self.event_id,.2,.2,'Operator')
        Path(clip['output_path']).write_bytes(b'tampered')
        with self.assertRaisesRegex(ValueError,'SHA-256 check'):
            export_file(self.library,self.mission,clip['id'])
