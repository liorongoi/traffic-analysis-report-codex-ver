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


if __name__ == '__main__':
    unittest.main()
