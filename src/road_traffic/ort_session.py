"""Serialized ONNX Runtime sessions with automatic DirectML and CPU fallback."""
from threading import RLock


class AdaptiveSession:
    """Prefer DirectML when present; retry on CPU if setup or execution fails."""
    def __init__(self, ort, model, cpu_threads=1):
        self.ort = ort
        self.model = str(model)
        self.cpu_threads = cpu_threads
        self._lock = RLock()
        self.fallback_reason = None
        self._session = None
        available = set(ort.get_available_providers())
        if 'DmlExecutionProvider' in available:
            try:
                options = self._options(directml=True)
                self._session = ort.InferenceSession(self.model, sess_options=options,
                    providers=['DmlExecutionProvider', 'CPUExecutionProvider'])
                if 'DmlExecutionProvider' in self._session.get_providers():
                    self.provider = 'DmlExecutionProvider'
                    return
                self.fallback_reason = 'DirectML was advertised but not selected by ONNX Runtime'
            except Exception as exc:
                self.fallback_reason = f'DirectML initialization failed: {exc}'
        self._use_cpu()

    def _options(self, directml=False):
        options = self.ort.SessionOptions()
        options.intra_op_num_threads = self.cpu_threads
        options.inter_op_num_threads = 1
        if directml:
            options.enable_mem_pattern = False
            options.execution_mode = self.ort.ExecutionMode.ORT_SEQUENTIAL
        self.options = options
        return options

    def _use_cpu(self):
        self._session = self.ort.InferenceSession(self.model,
            sess_options=self._options(), providers=['CPUExecutionProvider'])
        self.provider = 'CPUExecutionProvider'

    def run(self, *args, **kwargs):
        with self._lock:
            try:
                return self._session.run(*args, **kwargs)
            except Exception as exc:
                if self.provider != 'DmlExecutionProvider':
                    raise
                self.fallback_reason = f'DirectML execution failed: {exc}'
                self._use_cpu()
                return self._session.run(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._session, name)


def create_session(ort, model, cpu_threads=1):
    return AdaptiveSession(ort, model, cpu_threads)
