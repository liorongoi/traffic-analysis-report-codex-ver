"""Current-user offline browser launcher. No package managers or network downloads."""
from pathlib import Path
import hashlib
import json
import os
import sys
import threading
import webbrowser

root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'src'))

def verify_install_receipt(application_root):
    """Require the install-time content receipt when running an installed bundle."""
    manifest_path=application_root/'payload-manifest.json'
    if not manifest_path.exists():
        return  # Developer checkout; packaged installs always carry the manifest.
    receipt_path=application_root/'verified-receipt.json'
    try:
        receipt=json.loads(receipt_path.read_text(encoding='utf-8'))
        entries=json.loads(manifest_path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError('Installed files have not passed offline verification. Rerun the installer.') from exc
    digest=hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if (receipt.get('manifest_sha256')!=digest or receipt.get('files')!=len(entries)):
        raise RuntimeError('The installed payload changed after verification. Rerun the installer.')


def main():
    verify_install_receipt(root)
    from road_traffic.server import serve
    workspace=Path(os.environ.get('LOCALAPPDATA',str(Path.home())))/'RoadTrafficAnalysis'/'Workspace'
    # This preview intentionally uses an existing browser; a bundled desktop engine is a later deliverable.
    # Trigger opening after binding succeeds, rather than after an arbitrary startup delay.
    serve(workspace,root/'web',8765,open_browser=True)


if __name__=='__main__':
    main()
