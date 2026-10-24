"""Offline domain types; importing this module loads no vision frameworks."""
import re
from datetime import datetime, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


def capture_time(name, preset='camera'):
    if '/' in name or '\\' in name:
        raise ValueError('Provide a filename, not a path')
    if preset == 'camera':
        match = re.fullmatch(r'(\d{8})-(\d{4})-(.+)', name)
        if not match:
            raise ValueError('Expected DDMMYYYY-HHmm-<suffix>')
        result = datetime.strptime(match[1] + match[2], '%d%m%Y%H%M')
        precision = 'minute'
    elif preset == 'fintraffic':
        match = re.fullmatch(r'C\d+_(\d{8}T\d{6}Z)\.jpg', name, re.I)
        if not match:
            raise ValueError('Expected explicit Fintraffic filename format')
        result = datetime.strptime(match[1], '%Y%m%dT%H%M%SZ')
        precision = 'second'
    else:
        raise ValueError('Unknown capture-time preset')
    return result.replace(tzinfo=timezone.utc), precision


def named_zone(name):
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ValueError('Timezone unavailable: install the bundled timezone pack') from exc


def budget(ram_gib, available_gib):
    if ram_gib <= 0 or not 0 <= available_gib <= ram_gib:
        raise ValueError('Invalid memory observation')
    reserve = max(2, ram_gib * .25)
    return {'ram_gib': ram_gib, 'analysis_gib': round(max(0, min(ram_gib * .60, available_gib - reserve)), 2),
            'workers': 1 if ram_gib < 32 else 2, 'device': 'cpu',
            'note': 'Conservative proposal; GPU requires a successful adapter probe'}
