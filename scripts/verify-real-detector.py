"""Reproducible real-model checks on the authorized Fintraffic source, not accuracy scoring."""
import argparse
import hashlib
import json
from pathlib import Path
import socket
import sys
import time
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from PIL import Image
from road_traffic.detector import VehicleDetector,illumination,night_proposals
p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('--output',type=Path,required=True)
args=p.parse_args();manifest=json.loads((args.source/'manifest.json').read_text())
project=Path(__file__).resolve().parents[1]
checks=[]
# Block Python outbound connections for model initialization/scoring. This is not an OS firewall/air-gap test.
with patch.object(socket.socket,'connect',side_effect=RuntimeError('Outbound networking prohibited by real-model check')):
    detector=VehicleDetector(project/'model-packs'/'yolox-s')
    for row in [manifest['downloads'][0],manifest['downloads'][70]]:
        path=args.source/row['file']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
        with Image.open(path) as image:rgb=np.asarray(image.convert('RGB'))
        started=time.perf_counter();pred=detector.infer(rgb);elapsed=time.perf_counter()-started
        for a in pred:
            x,y,w,h=a['box'];assert 0<=x<=1 and 0<=y<=1 and w>0 and h>0 and x+w<=1.000001 and y+h<=1.000001
            assert np.isfinite(a['score']) and 0<=a['score']<=1 and a['kind']=='candidate'
        if '072525' in path.name:assert len(pred)>0,'Expected real daytime detector output'
        info=illumination(rgb)
        if row is manifest['downloads'][0]:assert info['low_light'],'Bright monochrome night frame must reach light branch'
        checks.append({'file':row['file'],'model_proposals':len(pred),'elapsed_seconds':elapsed,'illumination':info})
    empty=np.zeros((640,640,3),np.uint8)
    assert detector.infer(empty)==[],'Blank frame generated a vehicle'
result={'status':'passed','model_sha256':detector.sha,'providers':detector.session.get_providers(),
    'python_outbound_connections_blocked':True,'native_os_network_isolation_tested':False,'blank_frame_proposals':0,
    'samples':checks,'accuracy':'Not measured; these are functional checks, not independent ground-truth evaluation.'}
args.output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
