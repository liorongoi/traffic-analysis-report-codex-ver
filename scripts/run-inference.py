"""Private-runtime worker bootstrap, including embedded Python with isolated sys.path."""
from pathlib import Path
import runpy
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
runpy.run_module('road_traffic.worker',run_name='__main__')
