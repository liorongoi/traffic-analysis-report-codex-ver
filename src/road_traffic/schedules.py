"""Mission-local scan windows, evaluated against each capture's local wall time."""
from datetime import date, datetime, time, timedelta
from typing import NamedTuple
from .domain import named_zone


class CompiledWindow(NamedTuple):
    weekdays: frozenset
    start_date: date | None
    end_date: date | None
    start_time: time
    end_time: time


def validate_window(window):
    if not isinstance(window, dict):
        raise ValueError('Schedule window must be an object')
    weekdays = window.get('weekdays')
    if not isinstance(weekdays, list) or not weekdays or any(type(day) is not int or day < 0 or day > 6 for day in weekdays):
        raise ValueError('Choose one or more weekdays (Monday=0 through Sunday=6)')
    if len(set(weekdays)) != len(weekdays):
        raise ValueError('Schedule weekdays must be unique')
    try:
        start_time = time.fromisoformat(window['start_time'])
        end_time = time.fromisoformat(window['end_time'])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('Schedule times must use HH:MM local time') from exc
    if start_time.second or start_time.microsecond or end_time.second or end_time.microsecond:
        raise ValueError('Schedule times must use HH:MM precision')
    if start_time == end_time:
        raise ValueError('Schedule start and end times must differ')
    start_date = window.get('start_date') or None
    end_date = window.get('end_date') or None
    try:
        start_date = date.fromisoformat(start_date) if start_date else None
        end_date = date.fromisoformat(end_date) if end_date else None
    except (TypeError, ValueError) as exc:
        raise ValueError('Schedule date range must use YYYY-MM-DD') from exc
    if start_date and end_date and start_date > end_date:
        raise ValueError('Schedule start date must not be after its end date')
    return {'weekdays': sorted(weekdays), 'start_date': start_date.isoformat() if start_date else None,
        'end_date': end_date.isoformat() if end_date else None,
        'start_time': start_time.strftime('%H:%M'), 'end_time': end_time.strftime('%H:%M')}


def compile_windows(windows):
    compiled=[]
    for raw in windows:
        window=validate_window(raw)
        compiled.append(CompiledWindow(frozenset(window['weekdays']),
            date.fromisoformat(window['start_date']) if window['start_date'] else None,
            date.fromisoformat(window['end_date']) if window['end_date'] else None,
            time.fromisoformat(window['start_time']),time.fromisoformat(window['end_time'])))
    return compiled


def _date_matches(day, window):
    return (day.weekday() in window.weekdays and
        (not window.start_date or window.start_date <= day) and
        (not window.end_date or day <= window.end_date))


def eligible_compiled(capture_utc, zone, windows):
    """Evaluate a UTC capture against prevalidated local-time windows.

    Boundaries are start-inclusive/end-exclusive. For overnight windows, the
    selected weekday and date range apply to the date on which the window starts.
    DST folds and gaps follow the IANA zone's actual UTC-to-local conversion.
    """
    if not windows:
        return True
    instant = datetime.fromisoformat(capture_utc) if isinstance(capture_utc, str) else capture_utc
    if instant.tzinfo is None:
        raise ValueError('Capture timestamp must include a timezone')
    local = instant.astimezone(named_zone(zone))
    local_time = local.timetz().replace(tzinfo=None)
    for window in windows:
        start=window.start_time
        end=window.end_time
        if start < end:
            if start <= local_time < end and _date_matches(local.date(), window):
                return True
        elif local_time >= start and _date_matches(local.date(), window):
            return True
        elif local_time < end and _date_matches(local.date() - timedelta(days=1), window):
            return True
    return False


def eligible(capture_utc, zone, windows):
    return eligible_compiled(capture_utc, zone, compile_windows(windows))
