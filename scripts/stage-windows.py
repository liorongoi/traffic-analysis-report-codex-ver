"""Build-time staging only. Never downloads a runtime or a package.

Input must be a complete portable Windows x64 CPython 3.11/3.12 runtime with
all pinned dependencies and tzdata already installed. Staging does not prove
native execution or installer correctness; those remain separate Windows gates.
"""
import argparse
import hashlib
import json
from email.parser import Parser
from pathlib import Path
import re
import shutil
import struct


REQUIRED_MODULES = ('numpy', 'PIL', 'cv2', 'onnxruntime', 'watchdog', 'av', 'tzdata')
REQUIRED_SCRIPTS = ('Start-Traffic-Analysis.cmd', 'verify-install.py', 'launch.py',
    'run-inference.py', 'run-video-analysis.py', 'check-inference-runtime.py')


def sha256_file(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def parse_lock(lock_path):
    pins = {}
    for line_number, raw in enumerate(Path(lock_path).read_text(encoding='utf-8').splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        match = re.fullmatch(r'([A-Za-z0-9_.-]+)==([A-Za-z0-9_.+!-]+)', line)
        if not match:
            raise ValueError(f'Windows runtime lock must contain exact name==version pins (line {line_number})')
        name = re.sub(r'[-_.]+', '-', match.group(1)).lower()
        if name in pins:
            raise ValueError(f'Duplicate package in Windows runtime lock: {match.group(1)}')
        pins[name] = match.group(2)
    if not pins:
        raise ValueError('Windows runtime lock contains no exact package pins')
    return pins


def validate_windows_python(runtime, lock_path):
    runtime = Path(runtime).resolve(strict=True)
    executable = runtime / 'python.exe'
    if not executable.is_file():
        raise ValueError('Complete Windows Python 3.11/3.12 runtime required: python.exe')
    dlls = [runtime / 'python311.dll', runtime / 'python312.dll']
    dll = next((path for path in dlls if path.is_file()), None)
    if dll is None:
        raise ValueError('Complete Windows Python 3.11/3.12 runtime DLL is required')
    with executable.open('rb') as stream:
        header = stream.read(4096)
    if len(header) < 0x40 or header[:2] != b'MZ':
        raise ValueError('python.exe is not a Windows PE executable')
    pe_offset = struct.unpack_from('<I', header, 0x3c)[0]
    if pe_offset + 6 > len(header) or header[pe_offset:pe_offset + 4] != b'PE\0\0':
        raise ValueError('python.exe has an invalid PE header')
    machine = struct.unpack_from('<H', header, pe_offset + 4)[0]
    if machine != 0x8664:
        raise ValueError('A Windows x64 Python runtime is required')
    site_packages = runtime / 'Lib' / 'site-packages'
    module_paths = {'PIL': 'PIL', 'cv2': 'cv2'}
    missing = [module for module in REQUIRED_MODULES
        if not (site_packages / module_paths.get(module, module)).is_dir()]
    if missing:
        raise ValueError('Offline Windows runtime is missing packages: ' + ', '.join(missing))
    if not (site_packages / 'tzdata' / 'zoneinfo' / 'Asia' / 'Jerusalem').is_file():
        raise ValueError('Offline tzdata zone database is incomplete')
    expected = parse_lock(lock_path)
    installed = {}
    for metadata_path in site_packages.glob('*.dist-info/METADATA'):
        metadata = Parser().parsestr(metadata_path.read_text(encoding='utf-8', errors='replace'))
        name, version = metadata.get('Name'), metadata.get('Version')
        if name and version:
            installed[re.sub(r'[-_.]+', '-', name).lower()] = version
    mismatches = [f'{name}=={version} (found {installed.get(name, "missing")})'
        for name, version in expected.items() if installed.get(name) != version]
    if mismatches:
        raise ValueError('Offline Windows runtime does not match its pinned lock: ' + ', '.join(mismatches))
    return runtime


def validate_model_pack(pack, kind):
    pack = Path(pack)
    manifest_path = pack / 'manifest.json'
    if not manifest_path.is_file():
        raise ValueError(f'{kind} model pack manifest is missing')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if not isinstance(manifest, dict):
        raise ValueError(f'{kind} model pack manifest must be a JSON object')
    if kind == 'vehicle':
        expected = manifest.get('files')
        required = ('model.onnx', 'LICENSE', 'upstream-README.md')
        if not isinstance(expected, dict):
            raise ValueError('Vehicle model pack manifest has no files map')
    elif kind == 'plate':
        model_name = manifest.get('model')
        ocr = manifest.get('ocr_artifacts')
        if model_name != 'plate_yolox_tiny_640.onnx' or manifest.get('ocr_assets_included') is not True or not isinstance(ocr, dict):
            raise ValueError('Plate model manifest must include the detector and matched OCR model/dictionary')
        expected = {model_name: {'sha256': manifest.get('sha256'), 'bytes': manifest.get('bytes')}, **ocr}
        required = (model_name, 'plate_ocr_arabic_ppocrv3_mobile.onnx', 'plate_ocr_arabic_dict.json', 'LICENSE', 'NOTICE', 'README.md')
    else:
        raise ValueError('Unknown model pack kind')
    for name in required:
        path = pack / name
        if not path.is_file():
            raise ValueError(f'{kind} model pack is missing {name}')
        if name in expected:
            entry = expected[name]
            if not isinstance(entry, dict) or sha256_file(path) != entry.get('sha256'):
                raise ValueError(f'{kind} model pack checksum failed: {name}')
            if path.stat().st_size != entry.get('bytes'):
                raise ValueError(f'{kind} model pack size failed: {name}')
    return manifest


def payload_manifest(root):
    root = Path(root).resolve(strict=True)
    result = {}
    for path in sorted(root.rglob('*')):
        if path.is_file() and path.name != 'payload-manifest.json':
            result[path.relative_to(root).as_posix()] = sha256_file(path)
    return result


def stage(project, runtime):
    project = Path(project).resolve(strict=True)
    lock_path = project / 'requirements-inference-windows.lock.txt'
    if not lock_path.is_file():
        raise ValueError('Pinned Windows runtime lock is missing')
    runtime = validate_windows_python(runtime, lock_path)
    vehicle_pack = project / 'model-packs' / 'yolox-s'
    plate_pack = project / 'model-packs' / 'license-plate-yolox'
    validate_model_pack(vehicle_pack, 'vehicle')
    validate_model_pack(plate_pack, 'plate')
    for name in REQUIRED_SCRIPTS:
        if not (project / 'scripts' / name).is_file():
            raise ValueError('Required installed entry point is missing: scripts/' + name)

    output = project / 'staging' / 'windows'
    if output.exists():
        raise ValueError(f'Staging already exists; inspect it before rebuilding: {output}')
    output.mkdir(parents=True)
    try:
        shutil.copytree(runtime, output / 'runtime')
        for name in ('src', 'web', 'docs'):
            shutil.copytree(project / name, output / name,
                ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        shutil.copy2(project / 'README.md', output / 'README.md')
        shutil.copy2(lock_path, output / lock_path.name)
        shutil.copytree(project / 'scripts', output / 'scripts',
            ignore=shutil.ignore_patterns('__pycache__', '*.pyc', 'run-dev.sh', 'stage-windows.py',
                'verify-real-detector.py'))
        shutil.copytree(vehicle_pack, output / 'model-packs' / 'yolox-s')
        shutil.copytree(plate_pack, output / 'model-packs' / 'license-plate-yolox')
        (output / 'PREVIEW-LIMITATIONS.txt').write_text(
            'This is a preview build, not an accepted production deployment.\n'
            'Plate handling locates regions, saves chips, and proposes Latin and Arabic/Persian letters with digits with local OCR. '
            'Operators separately approve/correct/reject text; Persian digit recognition and target-country accuracy are unvalidated.\n'
            'Vehicle and plate detector accuracy, Windows execution, offline install/update, '
            'LAN authentication, GPU support, and target-hardware performance are not accepted.\n'
            'The application includes still-image monitoring, CSV/XLSX export, video event review, '
            'and plate crop review. See docs/ for current limits and open decisions.\n',
            encoding='utf-8')
        manifest = payload_manifest(output)
        (output / 'payload-manifest.json').write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    except BaseException:
        shutil.rmtree(output, ignore_errors=True)
        raise
    print('Staged offline preview media:', output)
    print('The bundle includes vehicle and plate detection/OCR models plus all runtime worker entry points.')
    print('Compile installer/windows/traffic-analysis.iss on Windows, inspect extracted payload, then test offline on a no-admin Windows account.')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', type=Path, required=True,
        help='Complete portable Windows x64 CPython runtime with pinned packages already installed')
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    try:
        stage(args.project, args.runtime)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
