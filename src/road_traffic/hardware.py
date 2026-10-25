"""On-demand host inventory without loading vision models or downloading data."""
import ctypes
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import tempfile
import time


def _linux_memory():
    values = {}
    try:
        for line in Path('/proc/meminfo').read_text().splitlines():
            match = re.match(r'^(MemTotal|MemAvailable):\s+(\d+)\s+kB$', line)
            if match:
                values[match.group(1)] = int(match.group(2)) * 1024
    except OSError:
        return None, None
    return values.get('MemTotal'), values.get('MemAvailable')


def _windows_memory():
    class MemoryStatus(ctypes.Structure):
        _fields_ = [
            ('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong),
            ('ullTotalPhys', ctypes.c_ulonglong), ('ullAvailPhys', ctypes.c_ulonglong),
            ('ullTotalPageFile', ctypes.c_ulonglong), ('ullAvailPageFile', ctypes.c_ulonglong),
            ('ullTotalVirtual', ctypes.c_ulonglong), ('ullAvailVirtual', ctypes.c_ulonglong),
            ('ullAvailExtendedVirtual', ctypes.c_ulonglong),
        ]
    status = MemoryStatus()
    status.dwLength = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None, None
    return status.ullTotalPhys, status.ullAvailPhys


def _macos_memory():
    try:
        total = int(subprocess.run(['sysctl', '-n', 'hw.memsize'], check=True,
            capture_output=True, text=True, timeout=3).stdout.strip())
        report = subprocess.run(['vm_stat'], check=True, capture_output=True,
            text=True, timeout=3).stdout
        page = re.search(r'page size of (\d+) bytes', report)
        if not page:
            return total, None
        page_size = int(page.group(1))
        names = ('Pages free', 'Pages inactive', 'Pages speculative')
        counts = [int(match.group(1).replace('.', '')) for name in names
            if (match := re.search(rf'^{re.escape(name)}:\s+(\d+)\.?$', report, re.M))]
        available = sum(counts) * page_size if len(counts) == len(names) else None
        return total, available
    except (OSError, subprocess.SubprocessError, ValueError):
        return None, None


def _memory():
    system = platform.system()
    if system == 'Windows':
        return _windows_memory()
    if system == 'Darwin':
        return _macos_memory()
    if system == 'Linux':
        return _linux_memory()
    return None, None


def _cpu_model():
    candidates = [platform.processor(), os.environ.get('PROCESSOR_IDENTIFIER', '')]
    if platform.system() == 'Darwin':
        try:
            candidates.insert(0, subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'],
                check=True, capture_output=True, text=True, timeout=3).stdout.strip())
        except (OSError, subprocess.SubprocessError):
            pass
    elif platform.system() == 'Linux':
        try:
            text = Path('/proc/cpuinfo').read_text(errors='replace')
            match = re.search(r'^model name\s*:\s*(.+)$', text, re.M)
            if match:
                candidates.insert(0, match.group(1).strip())
        except OSError:
            pass
    return next((item.strip() for item in candidates if item and item.strip()), 'Unknown')


def _gpu_inventory():
    executable = shutil.which('nvidia-smi')
    if executable:
        try:
            result = subprocess.run([executable,
                '--query-gpu=name,memory.total,memory.free,driver_version',
                '--format=csv,noheader,nounits'], check=True, capture_output=True,
                text=True, timeout=5)
            devices = []
            for line in result.stdout.splitlines():
                fields = [value.strip() for value in line.split(',')]
                if len(fields) != 4:
                    continue
                name, total, free, driver = fields
                devices.append({'name': name, 'vram_mib': _number(total),
                    'free_vram_mib': _number(free), 'driver': driver})
            if devices:
                return {'devices': devices, 'source': 'nvidia-smi'}
        except (OSError, subprocess.SubprocessError):
            pass
    if platform.system() == 'Windows':
        ps = shutil.which('powershell.exe') or shutil.which('powershell')
        if ps:
            command = "Get-CimInstance Win32_VideoController | Select-Object Name,AdapterRAM,DriverVersion | ConvertTo-Json -Compress"
            try:
                result = subprocess.run([ps, '-NoProfile', '-NonInteractive', '-Command', command],
                    check=True, capture_output=True, text=True, timeout=8)
                raw = json.loads(result.stdout or 'null')
                entries = raw if isinstance(raw, list) else [raw] if raw else []
                return {'devices': [{'name': item.get('Name') or 'Unknown',
                    'vram_bytes_reported': item.get('AdapterRAM'),
                    'driver': item.get('DriverVersion')} for item in entries],
                    'source': 'Windows WMI; adapter memory may be capped by the API'}
            except (OSError, subprocess.SubprocessError, ValueError, TypeError):
                pass
    return {'devices': [], 'source': 'No supported GPU inventory utility found'}


def _number(value):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _runtime_providers():
    if importlib.util.find_spec('onnxruntime') is None:
        return {'state': 'not_installed', 'available': []}
    try:
        import onnxruntime
        providers = list(onnxruntime.get_available_providers())
        return {'state': 'advertised_not_exercised', 'available': providers}
    except Exception as exc:
        return {'state': 'import_failed', 'available': [],
            'error': f'{type(exc).__name__}: {exc}'[:240]}


def _storage_performance(storage_path, sample_bytes=8 * 1024 * 1024):
    """Measure a small write+fsync and immediate read on the app-data volume."""
    fd = None
    filename = None
    try:
        fd, filename = tempfile.mkstemp(prefix='.traffic-analysis-storage-probe-', dir=storage_path)
        block = b'\xa5' * min(1024 * 1024, sample_bytes)
        remaining = sample_bytes
        started = time.perf_counter()
        while remaining:
            view = memoryview(block)[:min(len(block), remaining)]
            while view:
                written = os.write(fd, view)
                if written <= 0:
                    raise OSError('Storage probe stopped before the sample was written')
                view = view[written:]
            remaining -= min(len(block), remaining)
        os.fsync(fd)
        write_seconds = max(time.perf_counter() - started, 1e-9)

        os.lseek(fd, 0, os.SEEK_SET)
        started = time.perf_counter()
        remaining = sample_bytes
        while remaining:
            chunk = os.read(fd, min(1024 * 1024, remaining))
            if not chunk:
                raise OSError('Storage probe read back a truncated sample')
            remaining -= len(chunk)
        read_seconds = max(time.perf_counter() - started, 1e-9)
        mib = sample_bytes / (1024 ** 2)
        return {
            'state': 'measured',
            'sample_mib': round(mib, 2),
            'write_mib_s': round(mib / write_seconds, 2),
            'read_mib_s': round(mib / read_seconds, 2),
            'method': 'temporary app-data file; write includes fsync; immediate read may be served from OS cache',
        }
    except (OSError, ValueError) as exc:
        return {'state': 'unavailable', 'reason': f'{type(exc).__name__}: {exc}'[:240]}
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        if filename:
            try:
                os.unlink(filename)
            except OSError:
                pass


def _resource_profiles(total_bytes, available_bytes, logical_cores=None):
    """Return advisory host-memory budgets for the specified 16/48 GiB classes."""
    if not total_bytes or available_bytes is None:
        return None, None
    gib = 1024 ** 3
    total = total_bytes / gib
    available = available_bytes / gib
    profiles = {}
    for name, fraction, reserve_fraction, minimum_reserve in (
        ('conservative', .50, .25, 2.0),
        ('expanded', .75, .15, 4.0),
    ):
        reserve = max(minimum_reserve, total * reserve_fraction)
        profiles[name] = {
            'target_memory_class_gib': 16 if name == 'conservative' else 48,
            'analysis_memory_gib': round(max(0.0, min(total * fraction, available - reserve)), 2),
            'reserved_headroom_gib': round(reserve, 2),
            'cpu_threads': min(1 if name == 'conservative' else 4, max(1, logical_cores or 1)),
            'workers': 1,
            'device_recommendation': 'cpu until a model execution probe succeeds',
            'guidance': 'advisory RAM budget; selected profile applies the listed ONNX CPU thread limit',
        }
    return profiles, ('expanded' if total >= 40 else 'conservative')


def resolve_resource_profile(requested='auto'):
    """Resolve a profile to an ONNX CPU thread limit and current RAM estimate."""
    if requested not in ('auto', 'conservative', 'expanded'):
        raise ValueError('Resource profile must be auto, conservative, or expanded')
    total, available = _memory()
    cores = os.cpu_count() or 1
    profiles, recommended = _resource_profiles(total, available, cores)
    selected = recommended if requested == 'auto' else requested
    profile = profiles[selected] if profiles else None
    if profile:
        return {'requested': requested, 'selected': selected,
            'cpu_threads': profile['cpu_threads'],
            'analysis_memory_gib': profile['analysis_memory_gib'],
            'reserved_headroom_gib': profile['reserved_headroom_gib'],
            'workers': 1, 'device': 'CPUExecutionProvider'}
    return {'requested': requested, 'selected': selected,
        'cpu_threads': min(1 if selected == 'conservative' else 4, cores),
        'analysis_memory_gib': None, 'reserved_headroom_gib': None,
        'workers': 1, 'device': 'CPUExecutionProvider'}


def inventory(storage_path):
    """Inspect host capacity and measure small on-demand app-storage I/O; provider listing does not claim a successful GPU run."""
    total, available = _memory()
    disk = None
    try:
        usage = shutil.disk_usage(Path(storage_path))
        disk = {'total_bytes': usage.total, 'free_bytes': usage.free}
    except OSError:
        pass
    gib = 1024 ** 3
    report = {
        'operating_system': f'{platform.system()} {platform.release()}'.strip(),
        'cpu': {'model': _cpu_model(), 'logical_cores': os.cpu_count()},
        'memory': {'total_bytes': total, 'available_bytes': available,
            'total_gib': round(total / gib, 2) if total else None,
            'available_gib': round(available / gib, 2) if available is not None else None,
            'available_method': {'Windows': 'GlobalMemoryStatusEx', 'Darwin': 'free + inactive + speculative pages estimate',
                'Linux': '/proc/meminfo MemAvailable'}.get(platform.system(), 'unavailable')},
        'gpus': _gpu_inventory(),
        'onnx_runtime': _runtime_providers(),
        'storage': disk,
        'storage_performance': _storage_performance(storage_path),
        'gpu_execution': 'not verified by provider listing',
    }
    profiles, recommended = _resource_profiles(total, available, os.cpu_count())
    report['resource_profiles'] = profiles
    report['recommended_profile'] = recommended
    report['conservative_profile'] = profiles['conservative'] if profiles else None
    return report
