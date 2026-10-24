"""Current-user offline browser launcher. No package managers or network downloads."""
from pathlib import Path
import os
import sys
import threading
import webbrowser

root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'src'))
from road_traffic.server import serve
workspace=Path(os.environ.get('LOCALAPPDATA',str(Path.home())))/'RoadTrafficAnalysis'/'Workspace'
# This preview intentionally uses an existing browser; a bundled desktop engine is a later deliverable.
# Trigger opening after binding succeeds, rather than after an arbitrary startup delay.
serve(workspace,root/'web',8765,open_browser=True)
