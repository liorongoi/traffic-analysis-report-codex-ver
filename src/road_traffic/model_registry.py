"""Mission-scoped registry for immutable, operator-selected YOLOX ONNX packs."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import uuid

from .store import now


BASE_MODEL_ID = 'base'


def _candidate_root(library):
    return library.root / 'models' / 'yolox-candidates'


def _model_metadata(pack):
    from .detector import SUPPORTED_LABELS, model_pack
    root = Path(pack).resolve(strict=True)
    model, digest = model_pack(root)
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    classes = manifest.get('classes')
    if classes is not None and (not isinstance(classes, list) or not classes or
            len(set(classes)) != len(classes) or any(label not in SUPPORTED_LABELS for label in classes)):
        raise ValueError('Candidate manifest must list unique supported vehicle classes')
    return root, model, digest, manifest


def list_models(library, base_pack, mission=None):
    from .detector import model_pack
    _, base_sha = model_pack(base_pack)
    models = [{'id': BASE_MODEL_ID, 'name': 'YOLOX-S COCO baseline', 'sha256': base_sha,
        'classes': ['bicycle', 'car', 'motorcycle', 'bus', 'truck'], 'base': True}]
    root = _candidate_root(library)
    if root.is_dir():
        for folder in sorted(root.iterdir()):
            if not folder.is_dir() or folder.name.startswith('.'):
                continue
            try:
                _, _, digest, manifest = _model_metadata(folder)
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
                continue
            models.append({'id': folder.name,
                'name': str(manifest.get('name') or folder.name)[:120], 'sha256': digest,
                'classes': manifest.get('classes') or ['bicycle', 'car', 'motorcycle', 'bus', 'truck'],
                'base': False})
    selected = BASE_MODEL_ID
    if mission:
        selected = library.settings(mission).get('vehicle_model_id') or BASE_MODEL_ID
        if not any(item['id'] == selected for item in models):
            selected = BASE_MODEL_ID
    return {'models': models, 'selected_id': selected}


def resolve_model(library, base_pack, model_id):
    if model_id == BASE_MODEL_ID:
        return Path(base_pack).resolve(strict=True)
    if not isinstance(model_id, str) or not re.fullmatch(r'candidate-[a-f0-9]{12}', model_id):
        raise ValueError('Invalid vehicle model ID')
    candidate = (_candidate_root(library) / model_id).resolve(strict=True)
    if not candidate.is_relative_to(_candidate_root(library).resolve()):
        raise ValueError('Vehicle model path is outside the local candidate registry')
    _model_metadata(candidate)
    return candidate


def import_candidate(library, source):
    source, model, digest, manifest = _model_metadata(source)
    classes = manifest.get('classes')
    if not classes:
        raise ValueError('Candidate must declare the trained vehicle class list in manifest.json')
    candidate_id = 'candidate-' + digest[:12]
    root = _candidate_root(library)
    root.mkdir(parents=True, exist_ok=True)
    target = root / candidate_id
    if target.exists():
        _, _, existing_digest, _ = _model_metadata(target)
        if existing_digest != digest:
            raise ValueError('Candidate ID collision; model registry was left unchanged')
        return {'id': candidate_id, 'sha256': digest, 'existing': True}
    temporary = root / ('.import-' + uuid.uuid4().hex)
    temporary.mkdir()
    try:
        shutil.copyfile(model, temporary / 'model.onnx')
        shutil.copyfile(source / 'manifest.json', temporary / 'manifest.json')
        (temporary / 'source.json').write_text(json.dumps({'imported_utc': now(),
            'source_name': source.name, 'model_sha256': digest}, indent=2), encoding='utf-8')
        _, _, copied_digest, _ = _model_metadata(temporary)
        if copied_digest != digest:
            raise ValueError('Copied model checksum changed during import')
        temporary.replace(target)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {'id': candidate_id, 'sha256': digest, 'existing': False}


def select_model(library, base_pack, mission, model_id, operator, reason):
    if not isinstance(operator, str) or not operator.strip() or len(operator.strip()) > 120:
        raise ValueError('Enter the operator name for this model choice')
    if not isinstance(reason, str) or not reason.strip() or len(reason.strip()) > 1000:
        raise ValueError('Enter a reason for selecting this model')
    path = resolve_model(library, base_pack, model_id)
    _, _, digest, _ = _model_metadata(path)
    previous = library.settings(mission).get('vehicle_model_id') or BASE_MODEL_ID
    with library.connect(mission) as conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS vehicle_model_decisions(
            id TEXT PRIMARY KEY,previous_model_id TEXT NOT NULL,model_id TEXT NOT NULL,
            model_sha256 TEXT NOT NULL,operator TEXT NOT NULL,reason TEXT NOT NULL,created TEXT NOT NULL)''')
        if previous != model_id:
            conn.execute('UPDATE mission_settings SET vehicle_model_id=? WHERE mission_id=?', (model_id, mission))
            conn.execute('INSERT INTO vehicle_model_decisions VALUES (?,?,?,?,?,?,?)',
                (str(uuid.uuid4()), previous, model_id, digest, operator.strip(), reason.strip(), now()))
    return {'mission': mission, 'selected_id': model_id, 'sha256': digest,
        'changed': previous != model_id}


def decision_history(library, mission):
    with library.connect(mission) as conn:
        exists = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='vehicle_model_decisions'").fetchone()
        if not exists:
            return []
        return [dict(row) for row in conn.execute(
            'SELECT * FROM vehicle_model_decisions ORDER BY created DESC,id DESC LIMIT 100')]
