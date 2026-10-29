import importlib.util
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'stage-yolox-training.py'
SPEC = importlib.util.spec_from_file_location('stage_yolox_training', SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class TrainingBundleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.runtime = self.root / 'runtime-input'
        self.runtime.mkdir()
        (self.runtime / 'python.exe').write_bytes(b'fake-runtime')
        self.yolox = self.root / 'yolox-source'
        (self.yolox / 'yolox').mkdir(parents=True)
        (self.yolox / 'tools').mkdir()
        (self.yolox / 'yolox' / '__init__.py').write_text('__version__="0.3.0"')
        (self.yolox / 'tools' / 'train.py').write_text('')
        (self.yolox / 'tools' / 'export_onnx.py').write_text('')
        self.project = self.root / 'project'
        (self.project / 'scripts').mkdir(parents=True)
        (self.project / 'scripts' / 'finetune-yolox.py').write_text('cli')
        (self.project / 'src' / 'road_traffic').mkdir(parents=True)
        (self.project / 'src' / 'road_traffic' / '__init__.py').write_text('')
        baseline=b'baseline'
        digest=hashlib.sha256(baseline).hexdigest()
        (self.project / 'src' / 'road_traffic' / 'yolox_training.py').write_text(
            f"BASELINE_SHA256 = '{digest}'\n")
        (self.project / 'model-packs' / 'yolox-s').mkdir(parents=True)
        (self.project / 'model-packs' / 'yolox-s' / 'yolox_s_coco_pretrained.pth').write_bytes(baseline)
        self.output = self.root / 'transfer' / 'RoadLoom-Training'
        self.info = {'python':'3.11.9','platform':'Windows','architecture_bits':64,
            'torch':'2.5.1+cu121','cuda_build':'12.1',
            'cuda_available':True,'cuda_device':'Test GPU','onnx':'1.17.0',
            'opencv':'4.10.0','yolox':'0.3.0'}
        self.runner = Mock(return_value=subprocess.CompletedProcess([],0,
            stdout=json.dumps(self.info),stderr=''))

    def tearDown(self):
        self.temp.cleanup()

    def test_stage_copies_offline_runtime_source_and_manifest(self):
        result = MODULE.stage_training_bundle(self.runtime,self.yolox,self.output,
            project_root=self.project,require_cuda=True,runner=self.runner)
        self.assertTrue((self.output/'runtime'/'python.exe').is_file())
        self.assertTrue((self.output/'roadloom'/'scripts'/'finetune-yolox.py').is_file())
        self.assertTrue((self.output/'roadloom'/'model-packs'/'yolox-s'/'yolox_s_coco_pretrained.pth').is_file())
        self.assertTrue((self.output/'yolox'/'yolox'/'tools'/'train.py').is_file())
        self.assertTrue((self.output/'finetune.cmd').is_file())
        manifest=json.loads((self.output/'bundle-manifest.json').read_text())
        self.assertFalse(manifest['network_access_during_bundle_creation'])
        self.assertEqual(manifest['runtime']['cuda_device'],'Test GPU')
        self.assertIn('runtime/python.exe',manifest['files'])
        self.assertIsNone(result['archive'])

    def test_cuda_requirement_rejects_staging_machine_without_device(self):
        info={**self.info,'cuda_available':False,'cuda_device':None}
        runner=Mock(return_value=subprocess.CompletedProcess([],0,stdout=json.dumps(info),stderr=''))
        with self.assertRaisesRegex(ValueError,'CUDA is unavailable'):
            MODULE.stage_training_bundle(self.runtime,self.yolox,self.output,
                project_root=self.project,require_cuda=True,runner=runner)

    def test_rejects_incompatible_yolox_python_and_platform_versions(self):
        for changed, message in (
                ({'yolox':'0.4.0'},'YOLOX 0.3.0'),
                ({'python':'3.13.0'},'Python 3.11 or 3.12'),
                ({'platform':'Linux'},'Windows x64')):
            info={**self.info,**changed}
            runner=Mock(return_value=subprocess.CompletedProcess([],0,stdout=json.dumps(info),stderr=''))
            with self.subTest(changed=changed), self.assertRaisesRegex(ValueError,message):
                MODULE.stage_training_bundle(self.runtime,self.yolox,self.output,
                    project_root=self.project,runner=runner)

    def test_rejects_modified_baseline_checkpoint(self):
        (self.project/'model-packs'/'yolox-s'/'yolox_s_coco_pretrained.pth').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'does not match its recorded SHA-256'):
            MODULE.stage_training_bundle(self.runtime,self.yolox,self.output,
                project_root=self.project,runner=self.runner)

    def test_preflight_failure_removes_partial_staging_directory(self):
        runner=Mock(return_value=subprocess.CompletedProcess([],1,stdout='',stderr='missing module'))
        with self.assertRaisesRegex(ValueError,'preflight failed'):
            MODULE.stage_training_bundle(self.runtime,self.yolox,self.output,
                project_root=self.project,runner=runner)
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.output.parent.glob(self.output.name+'.staging-*')),[])

    def test_staging_does_not_overwrite_nonempty_destination(self):
        self.output.mkdir(parents=True)
        (self.output/'keep.txt').write_text('existing')
        with self.assertRaisesRegex(ValueError,'must be missing or empty'):
            MODULE.stage_training_bundle(self.runtime,self.yolox,self.output,
                project_root=self.project,runner=self.runner)

    def test_bundle_cannot_be_created_inside_inputs(self):
        output=self.runtime/'bundle'
        with self.assertRaisesRegex(ValueError,'outside the runtime'):
            MODULE.stage_training_bundle(self.runtime,self.yolox,output,
                project_root=self.project,runner=self.runner)

    def test_optional_transfer_zip_is_created_beside_bundle(self):
        result=MODULE.stage_training_bundle(self.runtime,self.yolox,self.output,
            project_root=self.project,runner=self.runner,zip_bundle=True)
        archive=Path(result['archive'])
        self.assertTrue(archive.is_file())
        self.assertEqual(archive.parent,self.output.parent.resolve())

    def test_incomplete_yolox_source_is_rejected_before_copy(self):
        (self.yolox/'tools'/'train.py').unlink()
        with self.assertRaisesRegex(ValueError,'YOLOX source checkout is incomplete'):
            MODULE.stage_training_bundle(self.runtime,self.yolox,self.output,
                project_root=self.project,runner=self.runner)


if __name__=='__main__':
    unittest.main()
