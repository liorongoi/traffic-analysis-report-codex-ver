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
    @staticmethod
    def fake_windows_runtime(runtime):
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

    @staticmethod
    def fake_model_pack(pack, kind):
        pack.mkdir(parents=True)
        if kind == 'vehicle':
            files = {}
            for name, content in (('model.onnx', b'vehicle'), ('LICENSE', b'vehicle license'),
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
                'sha256': hashlib.sha256(model).hexdigest(), 'bytes': len(model),
                'ocr_assets_included': False}
        (pack / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')

    def test_staged_model_packs_include_verified_plate_only_detector(self):
        vehicle = STAGER['validate_model_pack'](ROOT / 'model-packs' / 'yolox-s', 'vehicle')
        plate = STAGER['validate_model_pack'](ROOT / 'model-packs' / 'license-plate-yolox', 'plate')
        self.assertIn('model.onnx', vehicle['files'])
        self.assertIs(plate['ocr_assets_included'], False)

    def test_plate_pack_refuses_manifest_that_enables_recognition_assets(self):
        with tempfile.TemporaryDirectory() as temp:
            pack = Path(temp)
            (pack / 'manifest.json').write_text(json.dumps({
                'model': 'plate_yolox_tiny_640.onnx', 'ocr_assets_included': True}), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'exclude OCR assets'):
                STAGER['validate_model_pack'](pack, 'plate')

    def test_windows_runtime_staging_requires_x64_and_offline_dependencies(self):
        with tempfile.TemporaryDirectory() as temp:
            runtime = Path(temp)
            self.fake_windows_runtime(runtime)
            self.assertEqual(STAGER['validate_windows_python'](runtime), runtime.resolve())
            executable = bytearray((runtime / 'python.exe').read_bytes())
            struct.pack_into('<H', executable, 0x84, 0x14c)
            (runtime / 'python.exe').write_bytes(executable)
            with self.assertRaisesRegex(ValueError, 'Windows x64'):
                STAGER['validate_windows_python'](runtime)

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
            (project / 'web' / 'app.js').write_text('web app', encoding='utf-8')
            (project / 'docs' / '07-deployment.md').write_text('deployment notes', encoding='utf-8')
            self.fake_model_pack(project / 'model-packs' / 'yolox-s', 'vehicle')
            self.fake_model_pack(project / 'model-packs' / 'license-plate-yolox', 'plate')
            runtime = Path(temp) / 'runtime'
            self.fake_windows_runtime(runtime)
            staged = STAGER['stage'](project, runtime)
            expected = ('scripts/run-inference.py', 'scripts/run-video-analysis.py',
                'model-packs/license-plate-yolox/plate_yolox_tiny_640.onnx',
                'model-packs/license-plate-yolox/NOTICE', 'web/app.js', 'docs/07-deployment.md')
            for relative in expected:
                self.assertTrue((staged / relative).is_file(), relative)
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
