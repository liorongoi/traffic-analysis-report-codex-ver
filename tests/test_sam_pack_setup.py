import runpy
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch


class SamPackSetupTests(unittest.TestCase):
    def test_downloader_forces_anonymous_bf16_only_public_snapshot(self):
        script = Path(__file__).resolve().parents[1] / 'scripts' / 'download-sam3.1.py'
        module = runpy.run_path(str(script))
        calls = []

        def snapshot_download(**kwargs):
            calls.append(kwargs)
            folder = Path(kwargs['local_dir'])
            for name in kwargs['allow_patterns']:
                path = folder / name
                path.parent.mkdir(parents=True, exist_ok=True)
                if name == 'model.safetensors':
                    with path.open('wb') as stream:
                        stream.truncate(module['MINIMUM_WEIGHT_BYTES'] + 1)
                else:
                    path.write_text('{}', encoding='utf-8')
            return str(folder)

        with tempfile.TemporaryDirectory() as temporary, patch.dict(
                sys.modules, {'huggingface_hub': SimpleNamespace(snapshot_download=snapshot_download)}):
            destination = module['download'](Path(temporary) / 'sam3')
            self.assertTrue((destination / 'model.safetensors').is_file())
            self.assertEqual(calls[0]['repo_id'], 'unsloth/sam3.1')
            self.assertIs(calls[0]['token'], False)
            self.assertIn('*FP8*', calls[0]['ignore_patterns'])
            self.assertIn('*INT8*', calls[0]['ignore_patterns'])
            self.assertNotIn('sam3.1-INT8.safetensors', calls[0]['allow_patterns'])

    def test_setup_installs_optional_runtime_before_downloading_weights(self):
        script = Path(__file__).resolve().parents[1] / 'scripts' / 'setup-sam3.1.py'
        module = runpy.run_path(str(script))
        calls = []

        def install_runtime(destination):
            calls.append(('runtime', Path(destination)))

        def download(destination):
            calls.append(('weights', Path(destination)))
            return Path(destination)

        with tempfile.TemporaryDirectory() as temporary:
            model_dir = (Path(temporary) / 'model').resolve()
            result = module['setup'](model_dir, install_runtime, download)
            self.assertEqual(result, model_dir)
        self.assertEqual(calls, [('runtime', model_dir), ('weights', model_dir)])

    def test_runtime_setup_pins_cpu_model_stack_and_rejects_old_python(self):
        script = Path(__file__).resolve().parents[1] / 'scripts' / 'setup-sam3.1.py'
        module = runpy.run_path(str(script))
        calls = []

        def runner(command, **kwargs):
            calls.append((command, kwargs))
            if '-c' in command:
                return SimpleNamespace(stdout='3.12')
            return SimpleNamespace(stdout='')

        with tempfile.TemporaryDirectory() as temporary:
            runtime_dir = (Path(temporary) / 'model').resolve()
            module['install_runtime'](runtime_dir, python='roadloom-python', runner=runner)
            self.assertTrue(runtime_dir.joinpath('runtime').is_dir())
        self.assertEqual(calls[0][0][0:3], ['roadloom-python', '-c', "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"])
        installs = [command for command, _ in calls[1:]]
        self.assertIn('torch==2.10.0', installs[0])
        self.assertIn('torchvision==0.25.0', installs[0])
        self.assertIn('--index-url', installs[0])
        self.assertIn('--target', installs[0])
        self.assertIn(str(runtime_dir / 'runtime'), installs[0])
        self.assertIn('https://download.pytorch.org/whl/cpu', installs[0])
        self.assertIn('transformers==5.22.1', installs[1])
        self.assertIn('huggingface_hub', installs[1])

        def old_python(command, **kwargs):
            return SimpleNamespace(stdout='3.10')

        with self.assertRaisesRegex(RuntimeError, 'Python 3.11'):
            module['install_runtime'](Path(temporary) / 'model', python='old-python', runner=old_python)


if __name__ == '__main__':
    unittest.main()
