"""Bounded, separately stored light references for recurring camera views."""
from pathlib import Path
import uuid

import numpy as np

from .store import digest


MAX_REFERENCES_PER_VIEW = 32
MAX_PIXELS = 2_000_000
MIN_REFERENCE_IMAGES = 5
SIMILARITY_THRESHOLD = 0.8


def build_view_backgrounds(inputs, heldout, row_view_key, load_rgb, folder, stopped,
                           *, cv2_module=None, light_mask_fn=None,
                           illumination_fn=None):
    """Build one artifact per view, retaining at most 32 grayscale refs in RAM.

    Each view is prepared and written before the next one is loaded, keeping
    the peak background memory bounded independently of the number of views.
    """
    if cv2_module is None:
        import cv2 as cv2_module
    if light_mask_fn is None:
        from .detector import light_mask as light_mask_fn
    if illumination_fn is None:
        from .detector import illumination as illumination_fn
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    groups = {}
    for row in inputs:
        key = row_view_key(row)
        if key is not None:
            groups.setdefault(key, []).append(row)
    artifacts = {}
    for key, rows in groups.items():
        if stopped():
            break
        refs, masks, grays = [], [], []
        shape = None
        anchor = None
        for row in rows:
            if stopped():
                break
            if row['id'] in heldout:
                continue
            rgb = load_rgb(row)
            gray = cv2_module.cvtColor(rgb, cv2_module.COLOR_RGB2GRAY)
            if not illumination_fn(rgb)['low_light'] or gray.size > MAX_PIXELS:
                continue
            if shape is None:
                shape = gray.shape
                anchor = cv2_module.resize(gray, (160, 90)).astype(float).ravel()
            if gray.shape != shape:
                continue
            signature = cv2_module.resize(gray, (160, 90)).astype(float).ravel()
            similarity = float(np.corrcoef(anchor, signature)[0, 1])
            if not np.isfinite(similarity) or similarity < SIMILARITY_THRESHOLD:
                continue
            refs.append({'image_id': row['id'], 'sha256': row['sha256']})
            masks.append(light_mask_fn(gray))
            grays.append(gray)
            if len(refs) >= MAX_REFERENCES_PER_VIEW:
                break
        if not refs:
            continue
        stack = np.asarray(grays, dtype=np.uint8)
        counts = np.sum(masks, axis=0, dtype=np.uint16)
        median = np.median(stack, axis=0).astype(np.float32)
        mad = np.median(np.abs(stack.astype(np.float32) - median), axis=0).astype(np.float32)
        name = f'light-background-{uuid.uuid4().hex}.npz'
        path = folder / name
        np.savez_compressed(path, median=median, mad=mad, counts=counts, anchor=anchor,
            masks=np.asarray(masks, dtype=np.uint8), stack=stack,
            reference_ids=np.asarray([ref['image_id'] for ref in refs]))
        artifacts[key] = {'path': path, 'sha256': digest(path), 'references': refs,
            'view': key, 'min_reference_images': MIN_REFERENCE_IMAGES,
            'persistence_threshold': .6,
            'rule': 'leave-target-out persistence plus novelty >= max(25,3*MAD); held-out references excluded; similarity >= 0.8'}
    return artifacts


def load_background(artifact):
    """Load one view artifact into memory; callers should release it on switch."""
    with np.load(artifact['path'], allow_pickle=False) as stored:
        return {key: stored[key].copy() for key in ('median', 'mad', 'counts', 'anchor', 'masks', 'stack', 'reference_ids')}
