"""Separate worker process for one persisted video-analysis run."""
import argparse
from pathlib import Path

from .store import Library
from .video_inference import execute_worker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', required=True)
    parser.add_argument('--mission', required=True)
    parser.add_argument('--run', required=True)
    parser.add_argument('--pack', required=True)
    args = parser.parse_args()
    execute_worker(Library(args.workspace), args.mission, args.run, Path(args.pack))


if __name__ == '__main__':
    main()
