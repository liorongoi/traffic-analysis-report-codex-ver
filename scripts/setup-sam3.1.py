#!/usr/bin/env python3
"""Install the optional anonymous SAM 3.1 CPU runtime and local public weights.

Run with the installed RoadLoom private runtime:
  runtime/python.exe scripts/setup-sam3.1.py
This changes the installed runtime only; SAM remains outside the core installer.
"""
import argparse
from pathlib import Path
import runpy
import subprocess
import sys

TORCH_INDEX = "https://download.pytorch.org/whl/cpu"
TORCH_PACKAGES = ("torch==2.10.0", "torchvision==0.25.0")
TRANSFORMER_PACKAGES = ("transformers==5.22.1", "safetensors", "huggingface_hub")
SUPPORT_PACKAGES = (
    "filelock", "typing_extensions", "sympy", "networkx", "jinja2", "fsspec", "setuptools",
    "httpx", "hf_xet", "packaging", "pyyaml", "regex", "requests", "tokenizers", "tqdm",
)


def install_runtime(model_dir, python=None, runner=subprocess.run):
    python = str(python or sys.executable)
    version = runner([python, "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
        check=True, capture_output=True, text=True).stdout.strip()
    try:
        major, minor = (int(part) for part in version.split(".", 1))
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Could not determine the app runtime's Python version") from exc
    if (major, minor) < (3, 11):
        raise RuntimeError("Optional SAM 3.1 setup requires the installed app runtime to use Python 3.11 or newer")
    runtime = Path(model_dir).expanduser().resolve() / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    common = [python, "-m", "pip", "install", "--upgrade", "--disable-pip-version-check", "--no-input",
        "--target", str(runtime)]
    runner(common + ["--no-deps", "--index-url", TORCH_INDEX, *TORCH_PACKAGES], check=True)
    runner(common + ["--no-deps", *TRANSFORMER_PACKAGES], check=True)
    runner(common + list(SUPPORT_PACKAGES), check=True)


def _default_destination():
    downloader = runpy.run_path(str(Path(__file__).with_name("download-sam3.1.py")))
    return downloader["default_destination"]()


def setup(destination=None, runtime_installer=None, downloader=None):
    target = Path(destination).expanduser().resolve() if destination else _default_destination()
    installer = runtime_installer or install_runtime
    installer(target)
    if downloader is None:
        downloader = runpy.run_path(str(Path(__file__).with_name("download-sam3.1.py")))["download"]
    runtime = str(target / "runtime")
    sys.path.insert(0, runtime)
    try:
        return downloader(target)
    finally:
        sys.path.remove(runtime)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path,
        help="Optional local model folder; defaults to the current user's RoadLoom model directory")
    args = parser.parse_args()
    try:
        destination = setup(args.destination)
    except Exception as exc:
        print(f"Optional SAM 3.1 setup failed: {exc}", file=sys.stderr)
        return 1
    print(f"Optional SAM runtime and BF16 weights are ready at: {destination}")
    print("No account, sign-in, token, or core-installer rebuild was used.")
    print("Close and reopen RoadLoom before using SAM road masks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
