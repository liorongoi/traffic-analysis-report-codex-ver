"""Build-time staging only. Never downloads a runtime or a package.
Input must be a complete portable Windows x64 CPython 3.11+ runtime containing tzdata.
Native Windows execution and installer compilation remain separate validation steps.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

p=argparse.ArgumentParser()
p.add_argument('--runtime',type=Path,required=True)
args=p.parse_args()
project=Path(__file__).resolve().parents[1]
runtime=args.runtime.resolve(strict=True)
for name in ('python.exe','python311.dll' if (runtime/'python311.dll').exists() else 'python312.dll'):
    if not (runtime/name).is_file():p.error('Complete Windows Python 3.11/3.12 runtime required: '+name)
if not any(runtime.rglob('tzdata/zoneinfo/Asia/Jerusalem')):
    p.error('Offline tzdata package is required inside the private runtime')
for module in ('numpy','PIL','cv2','onnxruntime'):
    if not any(p.is_dir() for p in runtime.rglob(module)):
        p.error('Offline Windows inference package is missing: '+module)
pack=project/'model-packs'/'yolox-s'
if not (pack/'model.onnx').is_file() or not (pack/'LICENSE').is_file():
    p.error('Verified YOLOX-S model pack and license must be staged first')
pack_manifest=json.loads((pack/'manifest.json').read_text(encoding='utf-8'))
for filename in ('model.onnx','LICENSE','upstream-README.md'):
    with (pack/filename).open('rb') as stream:
        actual=hashlib.file_digest(stream,'sha256').hexdigest()
    if actual!=pack_manifest['files'][filename]['sha256']:
        p.error('Model pack integrity check failed: '+filename)
stage=project/'staging'/'windows'
if stage.exists():p.error('Staging already exists; inspect it before rebuilding')
stage.mkdir(parents=True)
shutil.copytree(runtime,stage/'runtime')
for name in ('src','web'):
    shutil.copytree(project/name,stage/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
(stage/'scripts').mkdir()
for name in ('Start-Traffic-Analysis.cmd','verify-install.py','launch.py','run-inference.py','check-inference-runtime.py'):
    shutil.copy2(project/'scripts'/name,stage/'scripts'/name)
shutil.copytree(pack,stage/'model-packs'/'yolox-s')
(stage/'PREVIEW-LIMITATIONS.txt').write_text('Local YOLOX-S inference and human review preview. Continuous monitoring, LAN authentication, XLSX and training are not implemented. Experimental night lights require human review. Windows execution must be verified on a Windows target.\n',encoding='utf-8')
manifest={}
for path in sorted(stage.rglob('*')):
    if path.is_file():
        with path.open('rb') as stream:
            h=hashlib.file_digest(stream,'sha256').hexdigest()
        manifest[path.relative_to(stage).as_posix()]=h
(stage/'payload-manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print('Staged offline PREVIEW media:',stage)
print('Not a finished installer. Compile installer/windows/traffic-analysis.iss on Windows, inspect extracted payload and test on a no-admin Windows account.')
