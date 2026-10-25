"""Review-only recurring-camera-view proposals from local visual geometry."""
from __future__ import annotations

from collections import defaultdict, deque
from itertools import combinations


MAX_IMAGES = 1200
MAX_DESCRIPTORS = 500
MIN_SHARED_MATCHES = 7
MIN_GEOMETRIC_INLIERS = 8
MIN_GEOMETRIC_SCORE = 0.055


def _cv_modules():
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise ValueError('Local OpenCV/NumPy runtime is required for camera-view proposals') from exc
    return cv2, np


def _features(library, mission, image_id):
    cv2, np = _cv_modules()
    path = library.image_path(mission, image_id)
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f'Could not decode image for view proposal: {path.name}')
    height, width = image.shape[:2]
    scale = min(1.0, 800 / max(height, width))
    if scale < 1.0:
        image = cv2.resize(image, (max(1, round(width * scale)), max(1, round(height * scale))),
                           interpolation=cv2.INTER_AREA)
    # Grayscale + local histogram equalization reduces exposure/illumination effects.
    normalized = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(image)
    detector = cv2.ORB_create(nfeatures=1000, scaleFactor=1.2, nlevels=8, fastThreshold=7)
    keypoints, descriptors = detector.detectAndCompute(normalized, None)
    if descriptors is None or len(keypoints) < MIN_GEOMETRIC_INLIERS:
        return {'keypoints': [], 'descriptors': None}
    if len(keypoints) > MAX_DESCRIPTORS:
        strongest = sorted(range(len(keypoints)), key=lambda i: keypoints[i].response, reverse=True)[:MAX_DESCRIPTORS]
        keypoints = [keypoints[i] for i in strongest]
        descriptors = descriptors[strongest]
    points = np.float32([point.pt for point in keypoints])
    return {'keypoints': points, 'descriptors': descriptors}


def _verified_similarity(first, second):
    cv2, np = _cv_modules()
    a, b = first['descriptors'], second['descriptors']
    if a is None or b is None:
        return None
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = matcher.knnMatch(a, b, k=2)
    good = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance]
    if len(good) < MIN_GEOMETRIC_INLIERS:
        return None
    source = np.float32([first['keypoints'][item.queryIdx] for item in good]).reshape(-1, 1, 2)
    target = np.float32([second['keypoints'][item.trainIdx] for item in good]).reshape(-1, 1, 2)
    try:
        _, mask = cv2.findHomography(source, target, cv2.RANSAC, 5.0)
    except cv2.error:
        return None
    if mask is None:
        return None
    inliers = int(mask.sum())
    score = inliers / max(1, min(len(a), len(b)))
    if inliers < MIN_GEOMETRIC_INLIERS or score < MIN_GEOMETRIC_SCORE:
        return None
    return {'inliers': inliers, 'score': round(min(1.0, score), 4)}


def propose(library, mission: str) -> dict:
    """Propose groups; this function never writes a view or changes an assignment."""
    cv2, np = _cv_modules()
    rows = library.images(mission)
    if len(rows) > MAX_IMAGES:
        raise ValueError(f'This review-only proposal pass supports up to {MAX_IMAGES} images; select a smaller mission partition')
    if len(rows) < 2:
        return {'available_images': len(rows), 'analyzed_images': len(rows), 'groups': [],
                'unmatched': [row['id'] for row in rows], 'method': 'grayscale CLAHE + ORB + RANSAC geometric matching'}

    features = {row['id']: _features(library, mission, row['id']) for row in rows}
    usable = [row for row in rows if features[row['id']]['descriptors'] is not None]
    descriptors = [features[row['id']]['descriptors'] for row in usable]
    if len(usable) < 2:
        return {'available_images': len(rows), 'analyzed_images': len(rows), 'groups': [],
                'unmatched': [row['id'] for row in rows], 'method': 'grayscale CLAHE + ORB + RANSAC geometric matching'}

    # Approximate local-feature retrieval avoids comparing every image pair in the whole mission.
    combined = np.vstack(descriptors)
    if len(combined) < 2:
        return {'available_images': len(rows), 'analyzed_images': len(rows), 'groups': [],
                'unmatched': [row['id'] for row in rows], 'method': 'grayscale CLAHE + ORB + RANSAC geometric matching'}
    owners = np.concatenate([np.full(len(item), index, dtype=np.int32) for index, item in enumerate(descriptors)])
    matcher = cv2.FlannBasedMatcher(dict(algorithm=6, table_number=12, key_size=20, multi_probe_level=2), {})
    matcher.add([combined])
    matcher.train()
    neighbors = matcher.knnMatch(combined, k=min(8, len(combined)))
    votes = defaultdict(lambda: defaultdict(float))
    for query_index, matches in enumerate(neighbors):
        source = int(owners[query_index])
        for match in matches:
            target = int(owners[match.trainIdx])
            if target == source or match.distance > 72:
                continue
            votes[source][target] += 1.0 / (1.0 + match.distance)

    top = {}
    for source, candidates in votes.items():
        top[source] = {target for target, score in sorted(candidates.items(), key=lambda pair: pair[1], reverse=True)[:10]
                       if score >= MIN_SHARED_MATCHES / 73}
    candidate_pairs = set()
    for source, targets in top.items():
        for target in targets:
            if source in top.get(target, set()):
                candidate_pairs.add((min(source, target), max(source, target)))

    edges = {}
    for first_index, second_index in sorted(candidate_pairs):
        relation = _verified_similarity(features[usable[first_index]['id']], features[usable[second_index]['id']])
        if relation:
            edges[first_index, second_index] = relation

    # Connected components make suggestions without creating persistent assignments.
    adjacency = defaultdict(set)
    for first, second in edges:
        adjacency[first].add(second)
        adjacency[second].add(first)
    components = []
    seen = set()
    for start in adjacency:
        if start in seen:
            continue
        queue = deque([start]); seen.add(start); component = []
        while queue:
            current = queue.popleft(); component.append(current)
            for neighbor in adjacency[current]:
                if neighbor not in seen:
                    seen.add(neighbor); queue.append(neighbor)
        if len(component) >= 2:
            components.append(component)

    proposals = []
    grouped_ids = set()
    for number, component in enumerate(sorted(components, key=lambda values: (-len(values), min(values))), start=1):
        members = [usable[index] for index in component]
        component_edges = [relation for (a, b), relation in edges.items() if a in component and b in component]
        centrality = {index: sum(edges.get((min(index, other), max(index, other)), {}).get('score', 0)
                                 for other in component if other != index) for index in component}
        representative = max(component, key=lambda index: (centrality[index], -index))
        group = []
        for row in sorted(members, key=lambda item: (item.get('capture_utc') or '', item['name'])):
            grouped_ids.add(row['id'])
            group.append({'id': row['id'], 'name': row['name'], 'capture_utc': row.get('capture_utc'),
                          'view_id': row.get('view_id'), 'view_name': row.get('view_name'),
                          'representative': row['id'] == usable[representative]['id']})
        proposals.append({'proposal': number, 'image_count': len(group), 'representative_image_id': usable[representative]['id'],
                          'mean_pair_score': round(sum(edge['score'] for edge in component_edges) / max(1, len(component_edges)), 4),
                          'assignments_changed': False, 'images': group})

    return {
        'available_images': len(rows), 'analyzed_images': len(rows), 'feature_images': len(usable),
        'groups': proposals, 'unmatched': [row['id'] for row in rows if row['id'] not in grouped_ids],
        'method': 'grayscale CLAHE + ORB + approximate descriptor retrieval + RANSAC geometric verification',
        'illumination_handling': 'grayscale and local contrast equalization reduce brightness sensitivity; this is a proposal heuristic, not lighting invariance proof',
        'acceptance': 'unvalidated; review representatives and every proposed member before using or assigning a group',
    }
