"""Private-runtime worker bootstrap for one durable video-analysis run."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


def main():
    from road_traffic.video_worker import main as worker_main
    worker_main()


if __name__ == '__main__':
    main()

