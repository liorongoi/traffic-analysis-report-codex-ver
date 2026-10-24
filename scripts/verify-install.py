"""Installed, network-free integrity check; standard-library imports only."""
import hashlib
import json
from pathlib import Path
import sys
import subprocess

root=Path(sys.argv[1]).resolve()
receipt=root/'verified-receipt.json'
receipt.unlink(missing_ok=True)
try:
    manifest=root/'payload-manifest.json'
    entries=json.loads(manifest.read_text(encoding='utf-8'))
    for relative,expected in entries.items():
        path=(root/relative).resolve(strict=True)
        if not path.is_relative_to(root) or not path.is_file():raise ValueError('Invalid payload path: '+relative)
        h=hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024*1024),b''):h.update(block)
        if h.hexdigest()!=expected:raise ValueError('Checksum mismatch: '+relative)
    # ZoneInfo must resolve exclusively from the installed runtime's data pack.
    from zoneinfo import ZoneInfo
    for zone in ('Asia/Jerusalem','Asia/Tehran','Asia/Beirut','Europe/Riga','Europe/Helsinki'):
        ZoneInfo(zone)
    # Framework checks happen here, during installation, not every normal launch.
    check=subprocess.run([sys.executable,'-I',str(root/'scripts'/'check-inference-runtime.py'),str(root)],capture_output=True,text=True,encoding='utf-8',errors='replace')
    (root/'inference-runtime-validation.txt').write_text(check.stdout+'\n'+check.stderr,encoding='utf-8')
    if check.returncode:raise ValueError('Native inference runtime check failed. See inference-runtime-validation.txt')
    receipt.write_text(json.dumps({'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),'files':len(entries)}),encoding='utf-8')
except Exception as exc:
    (root/'verification-error.txt').write_text(str(exc),encoding='utf-8')
    raise
