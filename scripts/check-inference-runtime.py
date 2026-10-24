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
print(json.dumps({'status':'passed','provider':model.session.get_providers(),'model_sha256':model.sha,
    'numpy':np.__version__,'opencv':cv2.__version__,'onnxruntime':ort.__version__,'blank_image_proposals':len(result)}))
