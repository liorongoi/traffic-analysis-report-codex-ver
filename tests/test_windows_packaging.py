import hashlib
import json
from pathlib import Path
import runpy
import struct
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
STAGER = runpy.run_path(str(ROOT / 'scripts' / 'stage-windows.py'))
LAUNCHER = runpy.run_path(str(ROOT / 'scripts' / 'launch.py'))


class WindowsPackagingTests(unittest.TestCase):
    def test_single_roadloom_installer_keeps_upgrade_identity_and_branded_icon(self):
        installers = list((ROOT / 'installer').rglob('*.iss'))
        self.assertEqual([path.relative_to(ROOT).as_posix() for path in installers],
            ['installer/windows/traffic-analysis.iss'])
        script = installers[0].read_text(encoding='utf-8')
        self.assertIn('AppId=RoadTrafficAnalysisIndependent', script)
        self.assertIn('AppName=RoadLoom Preview', script)
        self.assertIn('DefaultDirName={localappdata}\\Programs\\RoadTrafficAnalysis', script)
        self.assertIn('OutputBaseFilename=RoadLoom-Preview-Setup', script)
        self.assertIn('SetupIconFile=..\\..\\web\\roadloom.ico', script)
        self.assertIn('IconFilename: "{app}\\web\\roadloom.ico"', script)
        workflow = (ROOT / '.github' / 'workflows' / 'windows-preview-package.yml').read_text(encoding='utf-8')
        self.assertIn('RoadLoom-Preview-Setup.exe', workflow)
        self.assertNotIn('Traffic-Analysis-Preview-Setup.exe', workflow)
        self.assertIn("'upgrade-preservation-check.txt'", workflow)
        self.assertIn("if ((Get-Content -Raw $sentinel) -ne $env:GITHUB_RUN_ID)", workflow)
        self.assertIn('Reinstall changed existing workspace data.', workflow)
        self.assertIn('Uninstall removed existing workspace data.', workflow)
        self.assertIn('"-o$outputDir" $script', workflow)
        self.assertNotIn('--output-dir=', workflow)

    @staticmethod
    def fake_windows_runtime(runtime, lock_path):
        runtime.mkdir(parents=True, exist_ok=True)
        executable = bytearray(512)
        executable[:2] = b'MZ'
        struct.pack_into('<I', executable, 0x3c, 0x80)
        executable[0x80:0x84] = b'PE\0\0'
        struct.pack_into('<H', executable, 0x84, 0x8664)
        (runtime / 'python.exe').write_bytes(executable)
        (runtime / 'python312.dll').write_bytes(b'MZ runtime')
        site = runtime / 'Lib' / 'site-packages'
        for module in STAGER['REQUIRED_MODULES']:
            package = site / {'PIL': 'PIL', 'cv2': 'cv2'}.get(module, module)
            package.mkdir(parents=True)
        zone = site / 'tzdata' / 'zoneinfo' / 'Asia' / 'Jerusalem'
        zone.parent.mkdir(parents=True, exist_ok=True)
        zone.write_bytes(b'zone fixture')
        for name, version in STAGER['parse_lock'](lock_path).items():
            metadata = site / f'{name}-{version}.dist-info'
            metadata.mkdir()
            (metadata / 'METADATA').write_text(f'Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n',
                encoding='utf-8')

    @staticmethod
    def fake_model_pack(pack, kind):
        pack.mkdir(parents=True)
        if kind == 'vehicle':
            files = {}
            for name, content in (('model.onnx', b'vehicle'), ('LICENSE', b'vehicle license'),
                    ('yolox_s_coco_pretrained.pth', b'vehicle checkpoint'),
                    ('upstream-README.md', b'vehicle provenance')):
                (pack / name).write_bytes(content)
                files[name] = {'sha256': hashlib.sha256(content).hexdigest(), 'bytes': len(content)}
            manifest = {'files': files}
        else:
            model = b'plate region model'
            (pack / 'plate_yolox_tiny_640.onnx').write_bytes(model)
            for name in ('LICENSE', 'NOTICE', 'README.md'):
                (pack / name).write_text(name, encoding='utf-8')
            manifest = {'model': 'plate_yolox_tiny_640.onnx',
                'sha256': hashlib.sha256(model).hexdigest(), 'bytes': len(model)}
        (pack / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')

    def test_staged_model_packs_include_verified_plate_detector_only(self):
        vehicle = STAGER['validate_model_pack'](ROOT / 'model-packs' / 'yolox-s', 'vehicle')
        plate = STAGER['validate_model_pack'](ROOT / 'model-packs' / 'license-plate-yolox', 'plate')
        self.assertIn('model.onnx', vehicle['files'])
        self.assertEqual(plate['model'], 'plate_yolox_tiny_640.onnx')
        self.assertNotIn('ocr_artifacts', plate)

    def test_plate_pack_refuses_manifest_missing_detector(self):
        with tempfile.TemporaryDirectory() as temp:
            pack = Path(temp)
            (pack / 'manifest.json').write_text(json.dumps({'model': 'wrong-name.onnx'}), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'identify the plate-region detector'):
                STAGER['validate_model_pack'](pack, 'plate')

    def test_windows_runtime_staging_requires_x64_and_offline_dependencies(self):
        with tempfile.TemporaryDirectory() as temp:
            runtime = Path(temp)
            lock = ROOT / 'requirements-inference-windows.lock.txt'
            self.fake_windows_runtime(runtime, lock)
            self.assertEqual(STAGER['validate_windows_python'](runtime, lock), runtime.resolve())
            metadata = next((runtime / 'Lib' / 'site-packages').glob('numpy-*.dist-info/METADATA'))
            metadata.write_text('Metadata-Version: 2.1\nName: numpy\nVersion: 0\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'numpy==.*found 0'):
                STAGER['validate_windows_python'](runtime, lock)
            metadata.write_text('Metadata-Version: 2.1\nName: numpy\nVersion: 2.5.4\n', encoding='utf-8')
            executable = bytearray((runtime / 'python.exe').read_bytes())
            struct.pack_into('<H', executable, 0x84, 0x14c)
            (runtime / 'python.exe').write_bytes(executable)
            with self.assertRaisesRegex(ValueError, 'Windows x64'):
                STAGER['validate_windows_python'](runtime, lock)

    def test_launch_receipt_is_bound_to_the_payload_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.assertIsNone(LAUNCHER['verify_install_receipt'](root))
            manifest = {'src/app.py': 'abc'}
            data = (json.dumps(manifest) + '\n').encode()
            (root / 'payload-manifest.json').write_bytes(data)
            receipt = {'manifest_sha256': hashlib.sha256(data).hexdigest(), 'files': 1}
            (root / 'verified-receipt.json').write_text(json.dumps(receipt), encoding='utf-8')
            self.assertIsNone(LAUNCHER['verify_install_receipt'](root))
            (root / 'payload-manifest.json').write_text(json.dumps({'src/app.py': 'changed'}), encoding='utf-8')
            with self.assertRaisesRegex(RuntimeError, 'payload changed'):
                LAUNCHER['verify_install_receipt'](root)

    def test_complete_preview_payload_contains_workers_both_models_and_docs(self):
        with tempfile.TemporaryDirectory() as temp:
            project = Path(temp) / 'project'
            for folder in ('src/road_traffic', 'web', 'docs', 'scripts'):
                (project / folder).mkdir(parents=True, exist_ok=True)
            (project / 'README.md').write_text('project', encoding='utf-8')
            for name in STAGER['REQUIRED_SCRIPTS']:
                (project / 'scripts' / name).write_text('entry point', encoding='utf-8')
            lock = ROOT / 'requirements-inference-windows.lock.txt'
            (project / lock.name).write_bytes(lock.read_bytes())
            (project / 'web' / 'app.js').write_text('web app', encoding='utf-8')
            (project / 'web' / 'data').mkdir()
            (project / 'web' / 'data' / 'timezone-cities.json').write_text('{"cities":[]}', encoding='utf-8')
            (project / 'web' / 'data' / 'timezone-countries.json').write_text('{"features":[]}', encoding='utf-8')
            (project / 'web' / 'data' / 'timezone-zones.json').write_text('{"features":[]}', encoding='utf-8')
            (project / 'docs' / '07-deployment.md').write_text('deployment notes', encoding='utf-8')
            self.fake_model_pack(project / 'model-packs' / 'yolox-s', 'vehicle')
            self.fake_model_pack(project / 'model-packs' / 'license-plate-yolox', 'plate')
            runtime = Path(temp) / 'runtime'
            self.fake_windows_runtime(runtime, lock)
            cache = runtime / 'Lib' / 'site-packages' / 'onnxruntime' / 'tools' / '__pycache__'
            cache.mkdir(parents=True)
            (cache / 'unused.cpython-312.pyc').write_bytes(b'cache')
            staged = STAGER['stage'](project, runtime)
            expected = ('scripts/run-inference.py', 'scripts/run-video-analysis.py',
                'model-packs/license-plate-yolox/plate_yolox_tiny_640.onnx',
                'model-packs/license-plate-yolox/NOTICE', 'web/app.js', 'docs/07-deployment.md',
                'requirements-inference-windows.lock.txt', 'web/data/timezone-cities.json',
                'web/data/timezone-countries.json', 'web/data/timezone-zones.json')
            for relative in expected:
                self.assertTrue((staged / relative).is_file(), relative)
            self.assertFalse((staged / 'runtime/Lib/site-packages/onnxruntime/tools/__pycache__').exists())
            manifest = json.loads((staged / 'payload-manifest.json').read_text(encoding='utf-8'))
            self.assertIn('PREVIEW-LIMITATIONS.txt', manifest)
            self.assertNotIn('payload-manifest.json', manifest)

    def test_install_verifier_hashes_payload_and_runs_both_model_smoke_gate(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            payload = root / 'src' / 'app.py'
            payload.parent.mkdir()
            payload.write_text('offline payload', encoding='utf-8')
            scripts = root / 'scripts'
            scripts.mkdir()
            smoke = scripts / 'check-inference-runtime.py'
            smoke.write_text("print('vehicle and plate CPU smoke passed')\n", encoding='utf-8')
            manifest = {
                'src/app.py': hashlib.sha256(payload.read_bytes()).hexdigest(),
                'scripts/check-inference-runtime.py': hashlib.sha256(smoke.read_bytes()).hexdigest(),
            }
            (root / 'payload-manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
            verifier = ROOT / 'scripts' / 'verify-install.py'
            result = subprocess.run([sys.executable, str(verifier), str(root)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            receipt = json.loads((root / 'verified-receipt.json').read_text(encoding='utf-8'))
            self.assertEqual(receipt['files'], 2)
            self.assertIn('vehicle and plate CPU smoke passed',
                (root / 'inference-runtime-validation.txt').read_text(encoding='utf-8'))
            payload.write_text('modified payload', encoding='utf-8')
            failed = subprocess.run([sys.executable, str(verifier), str(root)], capture_output=True, text=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertFalse((root / 'verified-receipt.json').exists())
            self.assertIn('Checksum mismatch', (root / 'verification-error.txt').read_text(encoding='utf-8'))


if __name__ == '__main__':
    unittest.main()
