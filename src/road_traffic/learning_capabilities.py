"""Declared learning/refit capabilities for algorithms shipped by this project."""
from copy import deepcopy


_CAPABILITIES = [
    {
        'id': 'vehicle-detector.yolox-s-coco',
        'name': 'YOLOX-S vehicle proposal detector',
        'role': 'local CPU detector; unconfirmed vehicle proposals',
        'methods': {
            'fine_tuning': 'not_implemented',
            'full_retraining': 'not_implemented',
            'downstream_classifier_training': 'not_supported',
            'background_model_refitting': 'not_supported',
            'parameter_calibration_only': 'not_implemented',
        },
        'available_controls': ['run-scoped confidence threshold', 'run-scoped tiled inference'],
        'limitations': ['No project training adapter exists for YOLOX.',
            'Run controls are not calibrated parameters or evidence of accuracy.'],
    },
    {
        'id': 'night-light.context-heuristics',
        'name': 'Night-light context and pairing heuristics',
        'role': 'proposal generation and uncertainty rules; never confirmed vehicle counts',
        'methods': {
            'fine_tuning': 'not_supported',
            'full_retraining': 'not_supported',
            'downstream_classifier_training': 'not_supported',
            'background_model_refitting': 'not_supported',
            'parameter_calibration_only': 'not_implemented',
        },
        'available_controls': ['run-scoped light processing enable/disable'],
        'limitations': ['No parameter calibration workflow.',
            'A light or paired-headlight hypothesis is not proof of a vehicle.'],
    },
    {
        'id': 'night-light.multinomial-correction-head',
        'name': 'Night-light correction head',
        'role': 'separate light-patch candidate classifier',
        'methods': {
            'fine_tuning': 'not_supported',
            'full_retraining': 'not_supported',
            'downstream_classifier_training': 'supported',
            'background_model_refitting': 'not_supported',
            'parameter_calibration_only': 'not_supported',
        },
        'available_controls': ['manual candidate fitting from eligible reviewed training labels',
            'separate validation and one-time held-out test evaluation',
            'explicit per-run candidate selection'],
        'limitations': ['Trains a separate multinomial head; YOLOX remains unchanged.',
            'Held-out test labels are excluded from fitting and validation.',
            'Candidate evaluation does not promote it or establish traffic-count accuracy.'],
    },
    {
        'id': 'night-background.per-view-reference',
        'name': 'Per-view night-light background reference',
        'role': 'run-scoped background/context evidence',
        'methods': {
            'fine_tuning': 'not_supported',
            'full_retraining': 'not_supported',
            'downstream_classifier_training': 'not_supported',
            'background_model_refitting': 'supported',
            'parameter_calibration_only': 'not_supported',
        },
        'available_controls': ['bounded per-run reference build from compatible non-held-out images'],
        'limitations': ['Builds context artifacts, not a learned detector checkpoint.',
            'Real-camera robustness and nighttime accuracy remain unvalidated.'],
    },
    {
        'id': 'plate-region.yolox-tiny',
        'name': 'Plate-region crop detector',
        'role': 'plate-box localization proposals for human approval',
        'methods': {
            'fine_tuning': 'not_implemented',
            'full_retraining': 'not_implemented',
            'downstream_classifier_training': 'not_supported',
            'background_model_refitting': 'not_supported',
            'parameter_calibration_only': 'not_implemented',
        },
        'available_controls': ['run-scoped detection threshold'],
        'limitations': ['No project training adapter or country-specific accuracy evidence.',
            'Local OCR text is an unverified suggestion; human review is recorded separately.',
            'Identity lookup is excluded.'],
    },
    {
        'id': 'video-association.policy-tracker',
        'name': 'Video same-recording association tracker',
        'role': 'explicit-policy track association; ambiguous matches stay separate',
        'methods': {
            'fine_tuning': 'not_supported',
            'full_retraining': 'not_supported',
            'downstream_classifier_training': 'not_supported',
            'background_model_refitting': 'not_supported',
            'parameter_calibration_only': 'not_implemented',
        },
        'available_controls': ['operator-entered tracker policy frozen per run'],
        'limitations': ['No automatic policy tuner or representative-footage acceptance.',
            'No cross-camera or cross-recording identity matching.'],
    },
    {
        'id': 'camera-view.geometry-proposer',
        'name': 'Camera-view geometry proposer',
        'role': 'review-only grayscale/ORB/RANSAC similarity suggestions',
        'methods': {
            'fine_tuning': 'not_supported',
            'full_retraining': 'not_supported',
            'downstream_classifier_training': 'not_supported',
            'background_model_refitting': 'not_supported',
            'parameter_calibration_only': 'not_implemented',
        },
        'available_controls': ['none; assignments are made only through the revisioned manual workflow'],
        'limitations': ['Proposal scores are uncalibrated and do not assign or merge views.'],
    },
    {
        'id': 'object-search.constrained-grammar',
        'name': 'Offline object-search grammar',
        'role': 'typed query interpretation over saved image evidence',
        'methods': {
            'fine_tuning': 'not_supported',
            'full_retraining': 'not_supported',
            'downstream_classifier_training': 'not_supported',
            'background_model_refitting': 'not_supported',
            'parameter_calibration_only': 'not_implemented',
        },
        'available_controls': ['none; unsupported terms are rejected and visual uncertainty is retained'],
        'limitations': ['No language model or generated-code execution is used.'],
    },
]


def inventory():
    """Return a detached, JSON-safe capability inventory for APIs and the UI."""
    return deepcopy(_CAPABILITIES)
