"""Constrained, offline interpretation and search over saved image evidence."""
from __future__ import annotations

import hashlib
import json
import re


_CATEGORIES = {
    'car': ('car', ('car', 'cars', 'sedan', 'sedans', 'מכונית', 'מכוניות')),
    'truck': ('truck', ('truck', 'trucks', 'pickup', 'pickups', 'lorry', 'lorries', 'משאית', 'משאיות', 'טנדר', 'טנדרים')),
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
                    'sedan': 'sedan', 'sedans': 'sedan', 'suv': 'suv', 'suvs': 'suv', 'van': 'van', 'vans': 'van'}
_FILLER = {'find', 'show', 'locate', 'search', 'for', 'the', 'a', 'an', 'some', 'any', 'vehicle', 'vehicles',
           'object', 'objects', 'that', 'are', 'is', 'with', 'having', 'in', 'at', 'from', 'during', 'captured',
           'image', 'images', 'please', 'and', 'or', 'not', 'without', 'except', 'exclude', 'excluding',
           'מצא', 'מצאי', 'חפש', 'חפשי', 'הצג', 'הציגי', 'את', 'של', 'בתמונה', 'בתמונות', 'במשימה'}


def interpret(query: str, *, mission_id: str | None = None, view_id: str | None = None,
              run_id: str | None = None) -> dict:
    """Parse a small, explicit grammar; never executes generated code or calls a model."""
    if not isinstance(query, str) or not 1 <= len(query.strip()) <= 500:
        raise ValueError('Describe an object in 1–500 characters')
    words = re.findall(r"[\w-]+", query.casefold(), flags=re.UNICODE)
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
        'run_id': run_id,
        'scope': {
            'mission': 'current mission only',
            'run': run_id or 'latest saved inference run',
            'run_scope': 'the run freezes its schedule, view assignments, and ROI input decisions',
            'region': 'selected view, if provided; otherwise all views in the saved run',
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
        'execution': 'saved evidence search only; never an object-presence proof',
    }
    return spec


def execute(library, mission: str, spec: dict, *, confirmed: bool, run: str | None = None) -> dict:
    if confirmed is not True:
        raise ValueError('Review and confirm the interpreted search before execution')
    if not isinstance(spec, dict) or not isinstance(spec.get('query'), str):
        raise ValueError('Search specification is invalid')
    fresh = interpret(spec['query'], mission_id=mission, view_id=spec.get('view_id'), run_id=spec.get('run_id'))
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
    for row in rows:
        if spec.get('view_id') and row.get('view_id') != spec['view_id']:
            continue
        # A review only replaces predictions when it is explicitly bound to this exact run.
        if row.get('revision') and selected_run and row.get('review_prediction_run') == selected_run:
            items = json.loads(row['annotations'])
            source = 'review'
        else:
            items = row.get('predictions', [])
            source = 'model'
        found = []
        for number, item in enumerate(items):
            if item.get('rejected') or item.get('kind') not in {'vehicle', 'candidate'}:
                continue
            label = str(item.get('label', '')).casefold()
            category = next((name for name, (_, forms) in _CATEGORIES.items() if label in forms), None)
            if category and category not in spec['categories']:
                continue
            if category and category in spec['excluded_categories']:
                continue
            # A generic hand-drawn vehicle label cannot establish a requested fine class.
            uncertain = []
            if not category:
                uncertain.append('vehicle category')
            for attribute in spec['required_attributes'] + spec['excluded_attributes']:
                uncertain.append(attribute['attribute'])
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
        'uncertain_objects': uncertain_count, 'results': matches,
        'evidence_note': 'Matches are saved proposals or human annotations, not proof of object presence. Unknown attributes were not guessed.',
    }
