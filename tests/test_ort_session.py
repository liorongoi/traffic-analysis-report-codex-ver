import unittest

from road_traffic.ort_session import create_session


class Options:
    pass


class FakeSession:
    def __init__(self, providers, fail_run=False):
        self.providers = providers
        self.fail_run = fail_run

    def get_providers(self):
        return self.providers

    def run(self, outputs, inputs):
        if self.fail_run:
            raise RuntimeError('fixture execution failure')
        return ['ok']


class FakeOrt:
    class ExecutionMode:
        ORT_SEQUENTIAL = 'sequential'

    SessionOptions = Options

    def __init__(self, available, dml_init=False, dml_run=False):
        self.available = available
        self.dml_init = dml_init
        self.dml_run = dml_run
        self.calls = []

    def get_available_providers(self):
        return self.available

    def InferenceSession(self, model, *, sess_options, providers):
        self.calls.append((sess_options, providers))
        if providers[0] == 'DmlExecutionProvider':
            if self.dml_init:
                raise RuntimeError('fixture initialization failure')
            return FakeSession(providers, self.dml_run)
        return FakeSession(providers)


class AdaptiveSessionTests(unittest.TestCase):
    def test_cpu_only_host_uses_cpu(self):
        ort = FakeOrt(['CPUExecutionProvider'])
        session = create_session(ort, 'model.onnx', 3)
        self.assertEqual(session.provider, 'CPUExecutionProvider')
        self.assertEqual(session.options.intra_op_num_threads, 3)
        self.assertEqual(session.run(None, {}), ['ok'])

    def test_directml_is_configured_with_cpu_fallback(self):
        ort = FakeOrt(['DmlExecutionProvider', 'CPUExecutionProvider'])
        session = create_session(ort, 'model.onnx', 2)
        self.assertEqual(session.provider, 'DmlExecutionProvider')
        options = ort.calls[0][0]
        self.assertFalse(options.enable_mem_pattern)
        self.assertEqual(options.execution_mode, 'sequential')
        self.assertEqual(ort.calls[0][1], ['DmlExecutionProvider', 'CPUExecutionProvider'])

    def test_initialization_failure_falls_back_to_cpu(self):
        ort = FakeOrt(['DmlExecutionProvider', 'CPUExecutionProvider'], dml_init=True)
        session = create_session(ort, 'model.onnx')
        self.assertEqual(session.provider, 'CPUExecutionProvider')
        self.assertIn('initialization failed', session.fallback_reason)

    def test_execution_failure_retries_on_cpu(self):
        ort = FakeOrt(['DmlExecutionProvider', 'CPUExecutionProvider'], dml_run=True)
        session = create_session(ort, 'model.onnx')
        self.assertEqual(session.run(None, {}), ['ok'])
        self.assertEqual(session.provider, 'CPUExecutionProvider')
        self.assertIn('execution failed', session.fallback_reason)


if __name__ == '__main__':
    unittest.main()
