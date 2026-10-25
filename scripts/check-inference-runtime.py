"""Offline, CPU-native installer check. No network access or package installation."""
import hashlib
import json
from pathlib import Path
import sys
root=Path(sys.argv[1]).resolve()
sys.path.insert(0,str(root/'src'))
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
from road_traffic.detector import VehicleDetector
model=VehicleDetector(root/'model-packs'/'yolox-s',tiled=False)
result=model.infer(np.zeros((640,640,3),dtype=np.uint8))
if result:raise SystemExit('Vehicle model produced proposals for a blank smoke frame')
from road_traffic.plate_review import PlateDetector, _model_pack
plate_pack=root/'model-packs'/'license-plate-yolox'
_,plate_sha=_model_pack(plate_pack)
plate_model=PlateDetector(plate_pack)
plate_result=plate_model.detect(np.zeros((640,640,3),dtype=np.uint8))
if plate_result:raise SystemExit('Plate-region model produced proposals for a blank smoke frame')
print(json.dumps({'status':'passed','provider':model.session.get_providers(),'model_sha256':model.sha,
    'plate_model_sha256':plate_sha,'numpy':np.__version__,'opencv':cv2.__version__,
    'onnxruntime':ort.__version__,'blank_vehicle_proposals':len(result),
    'blank_plate_region_proposals':len(plate_result),'plate_text_recognition':'excluded'}))
