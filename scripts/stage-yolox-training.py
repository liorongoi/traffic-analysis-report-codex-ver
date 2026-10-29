#!/usr/bin/env python3
"""Bundle a prepared Windows CUDA/YOLOX environment for offline fine-tuning.

This is a staging-only tool. It does not install packages or access the network.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile


PROBE = r'''import json, platform, struct, torch, onnx, cv2
import yolox, yolox.tools.train, yolox.tools.export_onnx
available = bool(torch.cuda.is_available())
device = None
if available:
    torch.cuda.init()
    device = torch.cuda.get_device_name(0)
    torch.zeros((1,), device="cuda").item()
    torch.cuda.synchronize()
print(json.dumps({
    "python": platform.python_version(),
    "platform": platform.system(),
    "architecture_bits": struct.calcsize("P") * 8,
    "torch": torch.__version__,
    "cuda_build": torch.version.cuda,
    "cuda_available": available,
    "cuda_device": device,
    "onnx": onnx.__version__,
    "opencv": cv2.__version__,
    "yolox": getattr(yolox, "__version__", "source-checkout"),
}))'''


def _inside(path, root):
    return path == root or root in path.parents


def _sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _manifest_files(root):
    result = {}
    for path in sorted(Path(root).rglob('*')):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        result[relative] = {'bytes': path.stat().st_size, 'sha256': _sha256(path)}
    return result


def _baseline_sha(project):
    source = (Path(project) / 'src/road_traffic/yolox_training.py').read_text(encoding='utf-8')
    match = re.search(r"^BASELINE_SHA256\s*=\s*['\"]([0-9a-f]{64})['\"]", source, re.MULTILINE)
    if not match:
        raise ValueError('RoadLoom YOLOX trainer has no valid baseline checkpoint hash.')
    return match.group(1)


def _probe(runtime, yolox_root, *, require_cuda=False, runner=subprocess.run):
    python = Path(runtime) / 'python.exe'
    if not python.is_file():
        raise ValueError('Training runtime must contain python.exe.')
    root = Path(yolox_root)
    for relative in ('yolox/__init__.py', 'tools/train.py', 'tools/export_onnx.py'):
        if not (root / relative).is_file():
            raise ValueError(f'YOLOX source checkout is incomplete: missing {relative}')
    env = os.environ.copy()
    env['PYTHONPATH'] = str(root) + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    try:
        result = runner([str(python), '-c', PROBE], cwd=str(root), env=env,
            capture_output=True, text=True, check=False)
    except OSError as exc:
        raise ValueError('Could not run the staged Windows Python. Prepare and stage this bundle on Windows.') from exc
    if result.returncode:
        detail = (result.stderr or result.stdout or '').strip()[-1200:]
        raise ValueError('Training runtime preflight failed; install CUDA-enabled PyTorch, ONNX, OpenCV and the YOLOX source dependencies before staging. ' + detail)
    try:
        info = json.loads(result.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise ValueError('Training runtime preflight did not return valid version metadata.') from exc
    if not info.get('cuda_build'):
        raise ValueError('Training runtime must use a CUDA-enabled PyTorch build.')
    if info.get('platform') != 'Windows' or info.get('architecture_bits') != 64:
        raise ValueError('Training runtime must be Windows x64.')
    if info.get('python') and not info['python'].startswith(('3.11.', '3.12.')):
        raise ValueError('Training runtime must use Python 3.11 or 3.12.')
    if info.get('yolox') != '0.3.0':
        raise ValueError('Training runtime must use the RoadLoom-compatible YOLOX 0.3.0 source.')
    if require_cuda and not info.get('cuda_available'):
        raise ValueError('CUDA is unavailable on this staging computer; run the CUDA preflight on the intended training workstation.')
    return info


def stage_training_bundle(runtime, yolox_root, output, *, project_root=None,
        require_cuda=False, runner=subprocess.run, zip_bundle=False):
    """Copy a prevalidated portable runtime, YOLOX source and RoadLoom CLI.

    No dependency installation, network access, authentication or source-media
    copying occurs. The resulting directory can be copied to the offline PC.
    """
    runtime = Path(runtime).resolve(strict=True)
    yolox_root = Path(yolox_root).resolve(strict=True)
    project = Path(project_root or Path(__file__).resolve().parents[1]).resolve(strict=True)
    output = Path(output).resolve()
    inputs = (runtime, yolox_root, project)
    if any(_inside(output, root) for root in inputs):
        raise ValueError('Training bundle output must be outside the runtime, YOLOX checkout and project.')
    if any(_inside(root, output) for root in inputs):
        raise ValueError('Training bundle output must not contain any source input.')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Training bundle destination must be missing or empty.')
    for relative in ('scripts/finetune-yolox.py', 'src/road_traffic/yolox_training.py',
            'model-packs/yolox-s/yolox_s_coco_pretrained.pth'):
        if not (project / relative).is_file():
            raise ValueError(f'RoadLoom training source is incomplete: missing {relative}')

    baseline = project / 'model-packs/yolox-s/yolox_s_coco_pretrained.pth'
    if _sha256(baseline) != _baseline_sha(project):
        raise ValueError('The bundled YOLOX-S checkpoint does not match its recorded SHA-256.')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=output.name + '.staging-', dir=output.parent))
    try:
        shutil.copytree(runtime, temporary / 'runtime', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        shutil.copytree(yolox_root, temporary / 'yolox',
            ignore=shutil.ignore_patterns('.git', '__pycache__', '*.pyc'))
        source_tools = temporary / 'yolox' / 'tools'
        package_tools = temporary / 'yolox' / 'yolox' / 'tools'
        if package_tools.exists():
            raise ValueError('YOLOX source already contains yolox/tools; refusing to overlay trainer modules.')
        shutil.copytree(source_tools, package_tools,
            ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        runtime_info = _probe(temporary / 'runtime', temporary / 'yolox',
            require_cuda=require_cuda, runner=runner)

        roadloom = temporary / 'roadloom'
        (roadloom / 'scripts').mkdir(parents=True)
        shutil.copy2(project / 'scripts/finetune-yolox.py', roadloom / 'scripts/finetune-yolox.py')
        shutil.copytree(project / 'src', roadloom / 'src',
            ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        model = roadloom / 'model-packs/yolox-s'
        model.mkdir(parents=True)
        shutil.copy2(baseline, model)

        (temporary / 'verify-training.cmd').write_text(
            '@echo off\r\nsetlocal\r\n'
            'set "PYTHONPATH=%~dp0yolox;%PYTHONPATH%"\r\n'
            '"%~dp0runtime\\python.exe" -c "import torch; assert torch.cuda.is_available(), \'CUDA is unavailable\'; torch.cuda.init(); torch.zeros((1,), device=\'cuda\').item(); torch.cuda.synchronize(); print(torch.cuda.get_device_name(0))"\r\n'
            'exit /b %errorlevel%\r\n', encoding='utf-8')
        (temporary / 'finetune.cmd').write_text(
            '@echo off\r\nsetlocal\r\n'
            'set "PYTHONPATH=%~dp0yolox;%PYTHONPATH%"\r\n'
            '"%~dp0runtime\\python.exe" "%~dp0roadloom\\scripts\\finetune-yolox.py" --yolox-root "%~dp0yolox" %*\r\n'
            'exit /b %errorlevel%\r\n', encoding='utf-8')
        (temporary / 'README-offline-training.txt').write_text(
            'RoadLoom optional offline YOLOX-S fine-tuning bundle\n\n'
            'Run verify-training.cmd on the offline workstation first. It must run a CUDA tensor operation.\n'
            'Then run, for example:\n'
            '  finetune.cmd --dataset-zip D:\\RoadLoom\\reviewed-training.zip --output D:\\RoadLoom\\training-output --epochs 50 --batch-size 4\n\n'
            'The app dataset export protects training/validation/test roles; this trainer uses only training and validation. '
            'The held-out test split is not used. Import the resulting roadloom-model-pack folder in RoadLoom to select it for future still-image runs.\n'
            'This optional bundle is separate from the single application installer. It contains a prepared Python runtime and may be large.\n',
            encoding='utf-8')
        manifest = {
            'format': 'RoadLoom offline YOLOX training bundle v1',
            'runtime': runtime_info,
            'cuda_required_for_training': True,
            'network_access_during_bundle_creation': False,
            'training_dataset': 'not included',
            'held_out_test_used': False,
            'files': _manifest_files(temporary),
        }
        (temporary / 'bundle-manifest.json').write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
        if output.exists():
            output.rmdir()
        temporary.replace(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    archive = None
    if zip_bundle:
        archive = shutil.make_archive(str(output), 'zip', root_dir=output.parent, base_dir=output.name)
    return {'output': str(output), 'manifest': str(output / 'bundle-manifest.json'), 'archive': archive,
        'runtime': runtime_info}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', required=True, help='Prepared portable Windows Python runtime with CUDA PyTorch')
    parser.add_argument('--yolox-root', required=True, help='Local YOLOX source checkout; no clone/download is performed')
    parser.add_argument('--output', required=True, help='New directory outside the source inputs')
    parser.add_argument('--project-root', help='RoadLoom project containing the training CLI and baseline')
    parser.add_argument('--require-cuda', action='store_true', help='Require an attached CUDA GPU during staging validation')
    parser.add_argument('--zip', action='store_true', dest='zip_bundle', help='Create a transfer ZIP beside the bundle directory')
    args = parser.parse_args()
    result = stage_training_bundle(args.runtime, args.yolox_root, args.output,
        project_root=args.project_root, require_cuda=args.require_cuda, zip_bundle=args.zip_bundle)
    print(f"Offline YOLOX training bundle: {result['output']}")
    print(f"Runtime: Python {result['runtime']['python']}, PyTorch {result['runtime']['torch']}, CUDA build {result['runtime']['cuda_build']}")
    print(f"CUDA available on staging computer: {result['runtime']['cuda_available']}")
    if result['archive']:
        print(f"Transfer archive: {result['archive']}")


if __name__ == '__main__':
    main()
