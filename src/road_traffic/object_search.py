"""Constrained, offline interpretation and search over saved image evidence."""
from __future__ import annotations

import hashlib
import json
import re
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


_CATEGORIES = {
    'car': ('car', ('car', 'cars', 'sedan', 'sedans', 'מכונית', 'מכוניות')),
    'truck': ('truck', ('truck', 'trucks', 'pickup', 'pickups', 'lorry', 'lorries',
                        'semi-trailer', 'semi-trailers', 'semitrailer', 'semitrailers',
                        'משאית', 'משאיות', 'טנדר', 'טנדרים')),
    'bus': ('bus', ('bus', 'buses', 'coach', 'coaches', 'אוטובוס', 'אוטובוסים')),
    'motorcycle': ('motorcycle', ('motorcycle', 'motorcycles', 'motorbike', 'motorbikes', 'אופנוע', 'אופנועים')),
    'bicycle': ('bicycle', ('bicycle', 'bicycles', 'bike', 'bikes', 'אופניים')),
}
_COLOR_ALIASES = {
    **{color: color for color in ('red', 'blue', 'green', 'yellow', 'white', 'black', 'gray', 'silver', 'brown', 'orange')},
    'grey': 'gray',
    **{word: 'red' for word in ('אדום', 'אדומה', 'אדומים', 'אדומות')},
    **{word: 'blue' for word in ('כחול', 'כחולה', 'כחולים', 'כחולות')},
    **{word: 'green' for word in ('ירוק', 'ירוקה', 'ירוקים', 'ירוקות')},
    **{word: 'yellow' for word in ('צהוב', 'צהובה', 'צהובים', 'צהובות')},
    **{word: 'white' for word in ('לבן', 'לבנה', 'לבנים', 'לבנות')},
    **{word: 'black' for word in ('שחור', 'שחורה', 'שחורים', 'שחורות')},
    **{word: 'gray' for word in ('אפור', 'אפורה', 'אפורים', 'אפורות')},
    **{word: 'silver' for word in ('כסוף', 'כסופה', 'כסופים', 'כסופות')},
    **{word: 'brown' for word in ('חום', 'חומה', 'חומים', 'חומות')},
    **{word: 'orange' for word in ('כתום', 'כתומה', 'כתומים', 'כתומות')},
}
_SUBTYPE_ALIASES = {'pickup': 'pickup', 'pickups': 'pickup', 'טנדר': 'pickup', 'טנדרים': 'pickup',
                    'semi-trailer': 'semi_trailer', 'semi-trailers': 'semi_trailer',
                    'semitrailer': 'semi_trailer', 'semitrailers': 'semi_trailer',
                    'sedan': 'sedan', 'sedans': 'sedan', 'suv': 'suv', 'suvs': 'suv', 'van': 'van', 'vans': 'van', 'other': 'other'}
_FILLER = {'find', 'show', 'locate', 'search', 'for', 'of', 'the', 'a', 'an', 'some', 'any', 'all', 'every',
           'each', 'occurrence', 'occurrences', 'vehicle', 'vehicles',
           'object', 'objects', 'that', 'are', 'is', 'with', 'having', 'in', 'at', 'from', 'during', 'captured',
           'image', 'images', 'please', 'and', 'or', 'not', 'without', 'except', 'exclude', 'excluding',
           'on', 'between', 'to', 'through', 'near', 'by', 'יום', 'ביום', 'בימי', 'בין', 'עד', 'ל',
           'מצא', 'מצאי', 'חפש', 'חפשי', 'הצג', 'הציגי', 'את', 'של', 'בתמונה', 'בתמונות', 'במשימה'}


_WEEKDAY_FORMS = {
    **{word: 0 for word in ('monday', 'mondays', 'mon')},
    **{word: 1 for word in ('tuesday', 'tuesdays', 'tue', 'tues')},
    **{word: 2 for word in ('wednesday', 'wednesdays', 'wed')},
    **{word: 3 for word in ('thursday', 'thursdays', 'thu', 'thur', 'thurs')},
    **{word: 4 for word in ('friday', 'fridays', 'fri')},
    **{word: 5 for word in ('saturday', 'saturdays', 'sat')},
    **{word: 6 for word in ('sunday', 'sundays', 'sun')},
    'שני': 0, 'שלישי': 1, 'רביעי': 2, 'חמישי': 3, 'שישי': 4, 'שבת': 5, 'ראשון': 6,
}
_WEEKDAY_PATTERN = re.compile(r'(?<!\w)(' + '|'.join(sorted(map(re.escape, _WEEKDAY_FORMS), key=len, reverse=True)) + r')(?!\w)', re.I)
_TIME_RANGE_PATTERN = re.compile(r'(?<!\w)(?:between|from|בין)\s*(\d{1,2}:\d{2})\s*(?:and|to|through|ל|עד|-)\s*(\d{1,2}:\d{2})(?!\w)', re.I)
_SINGLE_TIME_BOUND_PATTERN = re.compile(r'(?<!\w)(after|before|אחרי|לפני)\s*(\d{1,2}:\d{2})(?!\w)', re.I)
_DATE_RANGE_PATTERN = re.compile(r'(?<!\w)(?:between|from)\s*(20\d{2}-\d{2}-\d{2})\s*(?:and|to|through)\s*(20\d{2}-\d{2}-\d{2})(?!\w)', re.I)
_DATE_PATTERN = re.compile(r'(?<!\w)(20\d{2}-\d{2}-\d{2})(?!\w)')


def _parse_time(value):
    try:
        hour, minute = (int(part) for part in value.split(':'))
        return time(hour, minute)
    except (ValueError, TypeError):
        raise ValueError('Use a valid 24-hour time such as 17:30') from None


def _extract_temporal_filter(query):
    text = query.casefold()
    result = {}
    time_match = _TIME_RANGE_PATTERN.search(text)
    if time_match:
        start = _parse_time(time_match.group(1))
        end = _parse_time(time_match.group(2))
        if start == end:
            raise ValueError('A search time range must have different start and end times')
        result['start_time'] = start.strftime('%H:%M')
        result['end_time'] = end.strftime('%H:%M')
        text = text[:time_match.start()] + ' ' + text[time_match.end():]
    single_time_matches=list(_SINGLE_TIME_BOUND_PATTERN.finditer(text))
    if len(single_time_matches)>1 or (single_time_matches and time_match):
        raise ValueError('Use one local-time bound or one explicit time range')
    if single_time_matches:
        match=single_time_matches[0]
        boundary=_parse_time(match.group(2))
        action=match.group(1).casefold()
        if action in {'after','אחרי'}:
            if boundary!=time(0,0):
                result['start_time']=boundary.strftime('%H:%M')
                result['end_time']='00:00'
        elif boundary==time(0,0):
            result['empty_time_range']=True
        else:
            result['start_time']='00:00'
            result['end_time']=boundary.strftime('%H:%M')
        text=text[:match.start()]+' '+text[match.end():]

    date_match = _DATE_RANGE_PATTERN.search(text)
    if date_match:
        start_date = date.fromisoformat(date_match.group(1))
        end_date = date.fromisoformat(date_match.group(2))
        if end_date < start_date:
            raise ValueError('Search date range ends before it starts')
        result['start_date'] = start_date.isoformat()
        result['end_date'] = end_date.isoformat()
        text = text[:date_match.start()] + ' ' + text[date_match.end():]
    else:
        matches = list(_DATE_PATTERN.finditer(text))
        if len(matches) > 1:
            raise ValueError('Use one date or an explicit between/from date range')
        if matches:
            requested = date.fromisoformat(matches[0].group(1)).isoformat()
            result['start_date'] = result['end_date'] = requested
            text = text[:matches[0].start()] + ' ' + text[matches[0].end():]

    weekdays = set()
    for match in _WEEKDAY_PATTERN.finditer(text):
        weekdays.add(_WEEKDAY_FORMS[match.group(1).casefold()])
    if weekdays:
        result['weekdays'] = sorted(weekdays)
        text = _WEEKDAY_PATTERN.sub(' ', text)
    return text, result


def _extract_view_filter(text, scope_snapshot, selected_view):
    assignments = (scope_snapshot or {}).get('view_assignments') or []
    matched = {}
    spans = []
    connectors = r'(?:at|in|near|by|ב)'
    for assignment in assignments:
        name = assignment.get('view_name')
        identifier = assignment.get('view_id')
        if not name or not identifier:
            continue
        name_pattern = r'\s+'.join(re.escape(part) for part in name.casefold().split())
        pattern = re.compile(r'(?<!\w)' + connectors + r'\s+' + name_pattern + r'(?!\w)', re.I)
        for match in pattern.finditer(text.casefold()):
            matched[identifier] = name
            spans.append(match.span())
    if len(matched) > 1:
        raise ValueError('Search one named camera view at a time')
    parsed_view = next(iter(matched), None)
    if selected_view and scope_snapshot is not None and selected_view not in {item.get('view_id') for item in assignments}:
        raise ValueError('Selected view is not used by this saved run')
    if selected_view and parsed_view and selected_view != parsed_view:
        raise ValueError('The query camera view conflicts with the selected view')
    chosen = selected_view or parsed_view
    for start, end in sorted(spans, reverse=True):
        text = text[:start] + ' ' + text[end:]
    return text, chosen


def _extract_region_filter(text, scope_snapshot, selected_view):
    """Resolve a named ROI only from immutable geometry in the selected run."""
    regions_by_view=(scope_snapshot or {}).get('regions_by_view') or {}
    candidates=[]
    for view_id, regions in regions_by_view.items():
        if selected_view and view_id!=selected_view:
            continue
        for region in regions:
            name=(region.get('name') or '').strip()
            if name:
                candidates.append((view_id,region,name))
    matches=[]
    lowered=text.casefold()
    for view_id,region,name in candidates:
        name_pattern=r'\s+'.join(re.escape(part) for part in name.casefold().split())
        pattern=re.compile(r'(?<!\w)(?:(?:in|inside|within|at|on)\s+'
            r'(?:(?:the\s+)?(?:(?:saved\s+|analysis\s+)?region|ROI|polygon|area)\s+(?:named\s+)?)?'
            r'|(?:בתוך|באזור|בתחום|על)\s+(?:(?:אזור|שטח|מצולע)\s+(?:בשם\s+)?)?|ב)'
            +name_pattern+r'(?!\w)',re.I)
        for match in pattern.finditer(lowered):
            matches.append((view_id,region,name,match.span()))
    # Prefer the longest full region name where names share a prefix. If the
    # same phrase names different view-specific polygons, require a view filter.
    by_span={}
    for item in matches:
        by_span.setdefault(item[3],[]).append(item)
    chosen=[]
    for span,items in sorted(by_span.items()):
        longest=max(item[3][1]-item[3][0] for item in items)
        best=[item for item in items if item[3][1]-item[3][0]==longest]
        if len({(item[0],item[1].get('id')) for item in best})>1:
            raise ValueError('Choose a camera view to disambiguate this named analysis region')
        item=best[0]
        if not chosen or item[3][0]>=chosen[-1][3][1]:
            chosen.append(item)
    identities={(item[0],item[1].get('id')) for item in chosen}
    if len(identities)>1:
        raise ValueError('Search one named analysis region at a time')
    if not chosen:
        return text,None,selected_view
    view_id,region,name,span=chosen[0]
    if selected_view and selected_view!=view_id:
        raise ValueError('The query analysis region is not in the selected camera view')
    text=text[:span[0]]+' '+text[span[1]:]
    return text,{'view_id':view_id,'region_id':region.get('id'),'name':name,
        'kind':region.get('kind'),'polygon':region.get('polygon')},selected_view or view_id


def _scope_timezone(scope_snapshot):
    schedule = (scope_snapshot or {}).get('schedule') or {}
    zone_name = schedule.get('timezone')
    if not zone_name:
        raise ValueError('Select a saved run with a recorded IANA time zone before using local date or time filters')
    try:
        return ZoneInfo(zone_name)
    except ZoneInfoNotFoundError:
        raise ValueError('The saved run has an unavailable IANA time zone') from None


def _matches_schedule(row, requested, scope_snapshot):
    if not requested:
        return True
    if requested.get('empty_time_range'):
        return False
    zone = _scope_timezone(scope_snapshot)
    raw = row.get('capture_utc')
    if not raw:
        return False
    try:
        captured = datetime.fromisoformat(str(raw).replace('Z', '+00:00'))
        if captured.tzinfo is None:
            return False
        local = captured.astimezone(zone)
    except (ValueError, OverflowError):
        return False
    day = local.date()
    start_text, end_text = requested.get('start_time'), requested.get('end_time')
    if start_text and end_text:
        start, end = _parse_time(start_text), _parse_time(end_text)
        clock = local.timetz().replace(tzinfo=None)
        if start < end:
            if not start <= clock < end:
                return False
        else:
            if not (clock >= start or clock < end):
                return False
            if clock < end:
                day -= timedelta(days=1)
    if requested.get('weekdays') and day.weekday() not in requested['weekdays']:
        return False
    if requested.get('start_date') and day < date.fromisoformat(requested['start_date']):
        return False
    if requested.get('end_date') and day > date.fromisoformat(requested['end_date']):
        return False
    return True



def interpret(query: str, *, mission_id: str | None = None, view_id: str | None = None,
              run_id: str | None = None, scope_snapshot: dict | None = None) -> dict:
    """Parse a small, explicit grammar; never executes generated code or calls a model."""
    if not isinstance(query, str) or not 1 <= len(query.strip()) <= 500:
        raise ValueError('Describe an object in 1–500 characters')
    query_without_time, schedule_filter = _extract_temporal_filter(query)
    query_without_scope, requested_view = _extract_view_filter(query_without_time, scope_snapshot, view_id)
    query_without_scope, requested_region, requested_view = _extract_region_filter(
        query_without_scope, scope_snapshot, requested_view)
    if schedule_filter:
        _scope_timezone(scope_snapshot)
    view_id = requested_view
    words = re.findall(r"[\w-]+", query_without_scope.casefold(), flags=re.UNICODE)
    if not words:
        raise ValueError('Describe at least one vehicle category')

    positives: set[str] = set()
    negatives: set[str] = set()
    colors: set[str] = set()
    subtypes: set[str] = set()
    unknown: list[str] = []
    negated = False
    category_words = {word: category for category, (_, forms) in _CATEGORIES.items() for word in forms}
    for word in words:
        if word in {'not', 'without', 'except', 'exclude', 'excluding'}:
            negated = True
            continue
        recognized = False
        if word in category_words:
            (negatives if negated else positives).add(category_words[word])
            recognized = True
        if word in _COLOR_ALIASES:
            colors.add(('!' if negated else '') + _COLOR_ALIASES[word])
            recognized = True
        if word in _SUBTYPE_ALIASES:
            subtypes.add(('!' if negated else '') + _SUBTYPE_ALIASES[word])
            recognized = True
        if recognized:
            negated = False
        elif word in _FILLER:
            continue
        else:
            unknown.append(word)
            negated = False

    if unknown:
        raise ValueError('Unsupported search terms: ' + ', '.join(sorted(set(unknown))))
    if not positives:
        raise ValueError('Name a supported vehicle category: car, truck, bus, motorcycle, or bicycle')

    required = []
    excluded_attributes = []
    for value in sorted(colors):
        (excluded_attributes if value.startswith('!') else required).append({'attribute': 'color', 'value': value.lstrip('!')})
    for value in sorted(subtypes):
        (excluded_attributes if value.startswith('!') else required).append({'attribute': 'subtype', 'value': value.lstrip('!')})
    # Categories such as pickup/SUV are not detector classes. The broad class is a searchable candidate,
    # while subtype remains explicitly unobservable unless a future reviewed attribute source adds it.
    spec = {
        'version': 1,
        'query': query.strip(),
        'categories': sorted(positives),
        'excluded_categories': sorted(negatives),
        'required_attributes': required,
        'excluded_attributes': excluded_attributes,
        'mission_id': mission_id,
        'view_id': view_id,
        'region_filter': requested_region,
        'run_id': run_id,
        'schedule_filter': schedule_filter,
        'scope_snapshot': json.loads(json.dumps(scope_snapshot, ensure_ascii=False)) if scope_snapshot is not None else None,
        'scope': {
            'mission': 'current mission only',
            'run': run_id or 'no saved inference run exists',
            'run_scope': ('the run freezes its schedule, view assignments, and ROI input decisions'
                          if run_id else 'no saved run schedule or region scope is available'),
            'region': (requested_region['name'] if requested_region else
                       ('selected view, if provided; otherwise all views in the saved run'
                        if run_id else 'no saved run views are available')),
        },
        'attribute_limits': ['color', 'fine vehicle subtype'],
        'operator_confirmation_required': True,
    }
    canonical = json.dumps(spec, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    spec['spec_hash'] = hashlib.sha256(canonical.encode('utf-8')).hexdigest()
    spec['interpretation'] = {
        'target_categories': spec['categories'],
        'excluded_categories': spec['excluded_categories'],
        'required_attributes': required,
        'excluded_attributes': excluded_attributes,
        'unsupported_or_unobservable': ['color', 'fine vehicle subtype'] if required or excluded_attributes else [],
        'requested_view': view_id or 'all views in selected run',
        'requested_region': requested_region['name'] if requested_region else 'all saved run regions',
        'requested_local_schedule': schedule_filter or 'all captures in selected run',
        'execution': 'saved evidence search only; never an object-presence proof',
        'attribute_source_note': ('Color and fine subtype are not inferred by the detector. Search filters operator-recorded values; missing values remain uncertain.'),
    }
    return spec


def execute(library, mission: str, spec: dict, *, confirmed: bool, run: str | None = None,
            scope_snapshot: dict | None = None) -> dict:
    if confirmed is not True:
        raise ValueError('Review and confirm the interpreted search before execution')
    if not isinstance(spec, dict) or not isinstance(spec.get('query'), str):
        raise ValueError('Search specification is invalid')
    fresh = interpret(spec['query'], mission_id=mission, view_id=spec.get('view_id'),
                      run_id=spec.get('run_id'), scope_snapshot=scope_snapshot)
    if spec != fresh:
        raise ValueError('Search specification changed; interpret it again')
    if run and spec.get('run_id') and run != spec['run_id']:
        raise ValueError('Selected inference run changed; interpret the query again')
    selected_run = spec.get('run_id') or run
    rows = library.images(mission, selected_run)
    selected_run = selected_run or next((row.get('prediction_run') for row in rows if row.get('prediction_run')), None)
    if selected_run and not any(row.get('prediction_run') == selected_run for row in rows):
        raise ValueError('Inference run is not part of this mission')
    matches = []
    uncertain_count = 0
    filtered_images = 0
    from .regions import point_in_polygon
    for row in rows:
        if spec.get('view_id') and row.get('view_id') != spec['view_id']:
            filtered_images += 1
            continue
        if not _matches_schedule(row, spec.get('schedule_filter') or {}, scope_snapshot):
            filtered_images += 1
            continue
        requested_region=spec.get('region_filter')
        row_regions=((scope_snapshot or {}).get('regions_by_view') or {}).get(row.get('view_id')) or []
        if requested_region:
            matching=next((region for region in row_regions if region.get('id')==requested_region.get('region_id')),None)
            if not matching or matching.get('polygon')!=requested_region.get('polygon'):
                filtered_images+=1
                continue
        exclusions=[region.get('polygon') for region in row_regions if region.get('kind')=='exclude']
        # A review only replaces predictions when it is explicitly bound to this exact run.
        if row.get('revision') and ((selected_run and row.get('review_prediction_run') == selected_run) or
                                     (not selected_run and not row.get('prediction_run') and not row.get('review_prediction_run'))):
            items = json.loads(row['annotations'])
            source = 'review'
        else:
            items = row.get('predictions', [])
            source = 'model'
        found = []
        for number, item in enumerate(items):
            if item.get('rejected') or item.get('kind') not in {'vehicle', 'candidate'}:
                continue
            box=item.get('box')
            if requested_region:
                if not isinstance(box,(list,tuple)) or len(box)!=4:
                    continue
                center=(float(box[0])+float(box[2])/2,float(box[1])+float(box[3])/2)
                if (not point_in_polygon(center,requested_region['polygon']) or
                    any(point_in_polygon(center,polygon) for polygon in exclusions)):
                    continue
            elif exclusions:
                if not isinstance(box,(list,tuple)) or len(box)!=4:
                    continue
                center=(float(box[0])+float(box[2])/2,float(box[1])+float(box[3])/2)
                if any(point_in_polygon(center,polygon) for polygon in exclusions):
                    continue
            label = str(item.get('label', '')).casefold()
            category = next((name for name, (_, forms) in _CATEGORIES.items() if label in forms), None)
            if category and category not in spec['categories']:
                continue
            if category and category in spec['excluded_categories']:
                continue
            # Detector proposals never establish color/subtype. Only attributes saved in a
            # run-bound human review can filter known values; absent observations stay uncertain.
            uncertain = []
            if not category:
                uncertain.append('vehicle category')
            known_attributes = item.get('vehicle_attributes', {}) if source == 'review' else {}
            attribute_conflict = False
            for attribute in spec['required_attributes']:
                observed = known_attributes.get(attribute['attribute'])
                if observed is None:
                    uncertain.append(attribute['attribute'])
                elif observed != attribute['value']:
                    attribute_conflict = True
            for attribute in spec['excluded_attributes']:
                observed = known_attributes.get(attribute['attribute'])
                if observed is None:
                    uncertain.append(attribute['attribute'])
                elif observed == attribute['value']:
                    attribute_conflict = True
            if attribute_conflict:
                continue
            uncertain = list(dict.fromkeys(uncertain))
            if uncertain:
                uncertain_count += 1
            found.append({
                'item_index': number,
                'label': item.get('label', 'vehicle'),
                'kind': item.get('kind'),
                'source': 'human-reviewed' if source == 'review' else 'unconfirmed model proposal',
                'box': item.get('box'),
                'uncertain_attributes': uncertain,
            })
        if found:
            matches.append({'image_id': row['id'], 'name': row['name'], 'capture_utc': row.get('capture_utc'),
                            'view_id': row.get('view_id'), 'view_name': row.get('view_name'), 'objects': found})
    return {
        'spec_hash': fresh['spec_hash'], 'mission': mission, 'run': selected_run,
        'matched_images': len(matches), 'matched_objects': sum(len(row['objects']) for row in matches),
        'uncertain_objects': uncertain_count, 'scope_filtered_images': filtered_images, 'results': matches,
        'evidence_note': 'Matches are saved proposals or human annotations, not proof of object presence. Unknown attributes were not guessed.',
    }
