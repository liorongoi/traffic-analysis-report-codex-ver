"""Private-runtime worker bootstrap for one durable video-analysis run."""
from pathlib import Path
import runpy
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
runpy.run_module('road_traffic.video_worker', run_name='__main__')

