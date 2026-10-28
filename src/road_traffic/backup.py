"""Verified, non-destructive mission backup and restore packages."""
from __future__ import annotations
from contextlib import nullcontext

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sqlite3
import tempfile
import uuid
import zipfile
from datetime import datetime, timezone


FORMAT = "roadloom-mission-backup"
VERSION = 1
MAX_MANIFEST_BYTES = 8 * 1024 * 1024


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_destination(library, destination):
    path = Path(destination).expanduser()
    if not path.is_absolute():
        raise ValueError("Backup destination must be an absolute path")
    parent = path.parent.resolve(strict=True)
    if not parent.is_dir() or not os.access(parent, os.W_OK | os.X_OK):
        raise ValueError("Backup destination folder is not writable")
    path = parent / path.name
    workspace = library.root.resolve()
    if path == workspace or path.is_relative_to(workspace) or workspace.is_relative_to(path):
        raise ValueError("Keep mission backups outside the active workspace")
    if path.exists():
        raise ValueError("Backup destination already exists; choose a new filename")
    return path


def _safe_relative(name):
    if not isinstance(name, str):
        raise ValueError("Backup contains a non-text path")
    relative = PurePosixPath(name)
    if (not name or relative.is_absolute() or "\\" in name or
            ":" in name or any(ord(char) < 32 for char in name) or
            any(part in ("", ".", "..") or part.endswith((".", " ")) for part in relative.parts)):
        raise ValueError("Backup contains an unsafe path")
    return relative


def _is_link(path):
    return path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction())


def _copy_file(source: Path, destination: Path, archive_name: str, entries: list):
    before = source.stat()
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    after = source.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"File changed during backup: {archive_name}")
    entries.append({"path": archive_name, "size": destination.stat().st_size,
        "sha256": _hash(destination)})


def _snapshot_database(source_path: Path, destination: Path, archive_name: str, entries: list):
    source = sqlite3.connect(source_path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        with sqlite3.connect(destination) as snapshot:
            source.backup(snapshot)
    finally:
        source.close()
    entries.append({"path": archive_name, "size": destination.stat().st_size,
        "sha256": _hash(destination)})


def _registered_outputs(database, output_root, mission):
    paths = set()
    with sqlite3.connect(database) as conn:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        queries = []
        if "exports" in tables:
            queries.append(("exports", "output_path", "SELECT output_path FROM exports WHERE mission_id=?"))
        if "video_clip_exports" in tables:
            queries.append(("video_clip_exports", "output_path", "SELECT output_path FROM video_clip_exports WHERE mission_id=?"))
        if "video_event_table_exports" in tables:
            queries.append(("video_event_table_exports", "output_path", "SELECT output_path FROM video_event_table_exports WHERE mission_id=?"))
            queries.append(("video_event_table_exports", "temp_path", "SELECT temp_path FROM video_event_table_exports WHERE mission_id=?"))
        for table, column, query in queries:
            paths.update((table, column, row[0]) for row in conn.execute(query, (mission,)) if row[0])
    mapped = []
    for table, column, raw_path in sorted(paths):
        path = Path(raw_path).expanduser()
        resolved = path.resolve(strict=False)
        try:
            relative = resolved.relative_to(output_root)
        except ValueError as exc:
            raise ValueError(f"Registered output is outside the mission output folder: {raw_path}") from exc
        if not relative.parts or any(part in (".", "..") for part in relative.parts):
            raise ValueError(f"Registered output has an unsafe relative path: {raw_path}")
        current = output_root
        for part in relative.parts:
            current = current / part
            if _is_link(current):
                raise ValueError(f"Registered output contains a linked path: {raw_path}")
        mapped.append({"table": table, "column": column, "original_path": str(path),
            "relative_path": relative.as_posix(), "exists": resolved.is_file()})
    return mapped


def create_backup(library, mission: str, destination):
    """Create a consistent DB copy plus immutable mission and plate-chip files."""
    target = _safe_destination(library, destination)
    mission_root = library.location(mission)
    if _is_link(mission_root / "mission.sqlite3"):
        raise ValueError("Mission database cannot be linked")
    if not (mission_root / "mission.sqlite3").is_file():
        raise ValueError("Mission not found")
    settings = library.settings(mission)
    entries = []
    with tempfile.TemporaryDirectory(prefix="roadloom-backup-") as temp_name:
        staging = Path(temp_name)
        staged_mission = staging / "mission"
        staged_mission.mkdir()
        # SQLite's backup API captures committed pages consistently, including
        # data which may still be in the source WAL file.
        _snapshot_database(mission_root / "mission.sqlite3", staged_mission / "mission.sqlite3",
            "mission/mission.sqlite3", entries)
        for source_path in sorted(mission_root.rglob("*")):
            relative = source_path.relative_to(mission_root)
            if relative.as_posix() == "mission.sqlite3":
                continue
            if source_path.is_symlink() or (hasattr(source_path, "is_junction") and source_path.is_junction()):
                raise ValueError(f"Mission contains a linked path and cannot be backed up safely: {relative}")
            if source_path.is_file():
                _copy_file(source_path, staged_mission / relative, "mission/" + relative.as_posix(), entries)

        output_paths = []
        missing_outputs = []
        output_root = None
        if settings["output_root"]:
            output_root = Path(settings["output_root"]).expanduser().resolve(strict=True)
            if library.root.is_relative_to(output_root) or output_root.is_relative_to(library.root):
                raise ValueError("Mission and crop output folders must be separate")
            if _is_link(output_root / "plate-crops"):
                raise ValueError("Plate crop backup cannot follow linked folders")
            output_paths = _registered_outputs(staged_mission / "mission.sqlite3", output_root, mission)
            copied_outputs = {item["relative_path"] for item in output_paths if item["exists"]}
            chip_root = output_root / "plate-crops" / mission
            if chip_root.exists():
                if _is_link(chip_root):
                    raise ValueError("Plate crop backup cannot follow linked folders")
                for source_path in sorted(chip_root.rglob("*")):
                    relative = source_path.relative_to(chip_root)
                    if _is_link(source_path):
                        raise ValueError(f"Plate crop data contains a linked path: {relative}")
                    if source_path.is_file():
                        if source_path.name in ("review.sqlite3-wal", "review.sqlite3-shm"):
                            continue
                        copied_outputs.add((Path("plate-crops") / mission / relative).as_posix())
            for relative_name in sorted(copied_outputs):
                source_path = output_root / Path(relative_name)
                target_path = staging / "output" / Path(relative_name)
                archive_name = "output/" + relative_name
                if source_path.name == "review.sqlite3":
                    _snapshot_database(source_path, target_path, archive_name, entries)
                else:
                    _copy_file(source_path, target_path, archive_name, entries)
            missing_outputs = [item["original_path"] for item in output_paths if not item["exists"]]

        with sqlite3.connect(staged_mission / "mission.sqlite3") as conn:
            mission_row = conn.execute("SELECT name FROM mission WHERE id=?", (mission,)).fetchone()
        if mission_row is None:
            raise ValueError("Mission not found")
        manifest = {"format": FORMAT, "version": VERSION, "mission_id": mission,
            "mission_name": mission_row[0],
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "files": sorted(entries, key=lambda item: item["path"]),
            "output_paths": output_paths, "missing_outputs": missing_outputs,
            "included": ["mission database and files"] + (["configured outputs and plate crop review data"]
                if any(item["path"].startswith("output/") for item in entries) else [])}
        manifest_path = staging / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        descriptor, temp_name = tempfile.mkstemp(prefix="." + target.name + ".", suffix=".partial", dir=target.parent)
        os.close(descriptor)
        temp_archive = Path(temp_name)
        try:
            with zipfile.ZipFile(temp_archive, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.write(manifest_path, "manifest.json")
                for entry in manifest["files"]:
                    relative = _safe_relative(entry["path"])
                    compression = (zipfile.ZIP_DEFLATED if relative.suffix.lower() in (".sqlite3", ".db")
                        else zipfile.ZIP_STORED)
                    archive.write(staging.joinpath(*relative.parts), entry["path"], compress_type=compression)
            # Verify the bytes as actually stored before publishing the archive.
            verify_backup(temp_archive)
            os.link(temp_archive, target)
            temp_archive.unlink()
        except BaseException:
            temp_archive.unlink(missing_ok=True)
            raise
    return {"path": str(target), "mission_id": mission, "file_count": len(entries),
        "size": target.stat().st_size, "sha256": _hash(target), "missing_outputs": missing_outputs}


def _read_verified(archive_path):
    try:
        archive = zipfile.ZipFile(archive_path, "r")
    except (OSError, zipfile.BadZipFile) as exc:
        raise ValueError("File is not a valid mission backup") from exc
    with archive:
        infos = archive.infolist()
        if len(infos) > 200000:
            raise ValueError("Backup has too many files")
        names = [info.filename for info in infos]
        if len(names) != len(set(names)) or "manifest.json" not in names:
            raise ValueError("Backup has duplicate paths or no manifest")
        if archive.getinfo("manifest.json").file_size > MAX_MANIFEST_BYTES:
            raise ValueError("Backup manifest exceeds the supported size")
        try:
            manifest = json.loads(archive.read("manifest.json"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("Backup manifest is unreadable") from exc
        if manifest.get("format") != FORMAT or manifest.get("version") != VERSION:
            raise ValueError("Unsupported mission backup format")
        try:
            mission = str(uuid.UUID(manifest["mission_id"]))
        except (KeyError, ValueError, TypeError, AttributeError) as exc:
            raise ValueError("Backup has an invalid mission ID") from exc
        if mission != manifest["mission_id"]:
            raise ValueError("Backup mission ID is not canonical")
        expected = {"manifest.json"}
        seen = set()
        file_entries = manifest.get("files")
        if not isinstance(file_entries, list):
            raise ValueError("Backup file list is invalid")
        for entry in file_entries:
            if not isinstance(entry, dict):
                raise ValueError("Backup file entry is invalid")
            name = entry.get("path")
            relative = _safe_relative(name)
            if name in seen or name == "manifest.json":
                raise ValueError("Backup manifest contains duplicate file entries")
            if relative.parts[0] not in ("mission", "output"):
                raise ValueError("Backup contains an unsupported top-level path")
            if len(relative.parts) < 2:
                raise ValueError("Backup file path is incomplete")
            size = entry.get("size")
            digest_value = entry.get("sha256")
            if type(size) is not int or size < 0 or not isinstance(digest_value, str) or len(digest_value) != 64:
                raise ValueError("Backup file metadata is invalid")
            seen.add(name)
            expected.add(name)
            info = archive.getinfo(name)
            if info.file_size != size:
                raise ValueError(f"Backup size check failed: {name}")
            digest = hashlib.sha256()
            with archive.open(info) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != digest_value:
                raise ValueError(f"Backup checksum failed: {name}")
        if set(names) != expected:
            raise ValueError("Backup contains unlisted files")
        if "mission/mission.sqlite3" not in seen:
            raise ValueError("Backup has no mission database")
        output_paths = manifest.get("output_paths", [])
        if not isinstance(output_paths, list):
            raise ValueError("Backup output references are invalid")
        for item in output_paths:
            if not isinstance(item, dict) or item.get("table") not in (
                    "exports", "video_clip_exports", "video_event_table_exports"):
                raise ValueError("Backup output reference is invalid")
            columns = {"exports": {"output_path"}, "video_clip_exports": {"output_path"},
                "video_event_table_exports": {"output_path", "temp_path"}}
            if item.get("column") not in columns[item["table"]]:
                raise ValueError("Backup output reference is invalid")
            _safe_relative(item.get("relative_path"))
            if not isinstance(item.get("original_path"), str) or type(item.get("exists")) is not bool:
                raise ValueError("Backup output reference is invalid")
            if item["exists"] and "output/" + item["relative_path"] not in seen:
                raise ValueError("Backup is missing a referenced output file")
        total_size = sum(info.file_size for info in infos)
        if total_size > 2**63 - 1:
            raise ValueError("Backup reports an unsupported expanded size")
        return manifest


def verify_backup(archive_path):
    """Verify archive structure, paths, sizes and SHA-256 checksums."""
    manifest = _read_verified(archive_path)
    with zipfile.ZipFile(archive_path, "r") as archive, tempfile.TemporaryDirectory(prefix="roadloom-verify-") as td:
        database = Path(td) / "mission.sqlite3"
        database.write_bytes(archive.read("mission/mission.sqlite3"))
        try:
            with sqlite3.connect(database) as conn:
                if conn.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                    raise ValueError("Backup mission database failed SQLite integrity check")
                row = conn.execute("SELECT id,name FROM mission").fetchone()
        except sqlite3.Error as exc:
            raise ValueError("Backup mission database is unreadable") from exc
        if not row or row[0] != manifest["mission_id"]:
            raise ValueError("Backup mission identity does not match its manifest")
    return {"valid": True, "mission_id": manifest["mission_id"],
        "mission_name": manifest.get("mission_name", ""), "file_count": len(manifest["files"]),
        "missing_outputs": manifest.get("missing_outputs", [])}


def restore_backup(library, archive_path, *, plate_output_root=None):
    """Restore only into a workspace without that mission; never overwrite work."""
    manifest = _read_verified(archive_path)
    mission = manifest["mission_id"]
    target = library.location(mission)
    if target.exists() or _is_link(target):
        raise ValueError("This workspace already contains the mission; restore will not overwrite it")
    entries = {entry["path"]: entry for entry in manifest["files"]}
    required = sum(entry["size"] for entry in entries.values())
    if required > shutil.disk_usage(library.root).free:
        raise ValueError("Not enough free workspace space to restore this mission")
    output_entries = [name for name in entries if name.startswith("output/")]
    output_paths = manifest.get("output_paths", [])
    restored_output = None
    output_parent = None
    if output_entries or output_paths:
        output_root = plate_output_root
        if not output_root:
            with zipfile.ZipFile(archive_path) as archive, tempfile.TemporaryDirectory(prefix="roadloom-restore-db-") as td:
                db = Path(td) / "mission.sqlite3"
                db.write_bytes(archive.read("mission/mission.sqlite3"))
                with sqlite3.connect(db) as conn:
                    row = conn.execute("SELECT output_root FROM mission_settings WHERE mission_id=?", (mission,)).fetchone()
                    output_root = row[0] if row else ""
        output_root = Path(output_root).expanduser() if output_root else None
        if output_root is None or not output_root.is_absolute() or not output_root.is_dir():
            raise ValueError("This backup includes output files; choose an existing output folder")
        output_root = output_root.resolve(strict=True)
        if library.root.is_relative_to(output_root) or output_root.is_relative_to(library.root):
            raise ValueError("Output and mission workspace must remain separate")
        output_parent = output_root / "roadloom-restored"
        if _is_link(output_parent):
            raise ValueError("Restore cannot follow a linked output folder")
        restored_output = output_parent / mission
        if restored_output.exists() or _is_link(restored_output):
            raise ValueError("Restored output destination already exists; restore will not overwrite it")
        output_required = sum(entries[name]["size"] for name in output_entries)
        if output_required > shutil.disk_usage(output_root).free:
            raise ValueError("Not enough free output space to restore mission exports and plate chips")

    if restored_output:
        output_parent.mkdir(parents=True, exist_ok=True)
    external_temp = (tempfile.TemporaryDirectory(prefix=f".restore-{mission}-", dir=output_parent)
        if restored_output else nullcontext(None))
    with tempfile.TemporaryDirectory(prefix=f".restore-{mission}-", dir=library.root) as td, external_temp as external_td:
        staging = Path(td)
        staged_mission = staging / "mission"
        with zipfile.ZipFile(archive_path, "r") as archive:
            for name in entries:
                relative = _safe_relative(name)
                if relative.parts[0] != "mission":
                    continue
                destination = staged_mission.joinpath(*relative.parts[1:])
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(name) as source, destination.open("xb") as output:
                    shutil.copyfileobj(source, output, 1024 * 1024)

        output_stage = None
        if restored_output:
            output_stage = Path(external_td) / "output"
            output_stage.mkdir()
            with zipfile.ZipFile(archive_path, "r") as archive:
                for name in output_entries:
                    relative = _safe_relative(name)
                    destination = output_stage.joinpath(*relative.parts[1:])
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(name) as source, destination.open("xb") as output:
                        shutil.copyfileobj(source, output, 1024 * 1024)

        db_path = staged_mission / "mission.sqlite3"
        with sqlite3.connect(db_path) as conn:
            check = conn.execute("PRAGMA quick_check").fetchone()[0]
            identity = conn.execute("SELECT id FROM mission").fetchone()
            if check != "ok" or not identity or identity[0] != mission:
                raise ValueError("Backup database integrity or mission identity check failed")
            if restored_output:
                conn.execute("UPDATE mission_settings SET output_root=? WHERE mission_id=?", (str(restored_output), mission))
                allowed = {"exports": {"output_path"}, "video_clip_exports": {"output_path"},
                    "video_event_table_exports": {"output_path", "temp_path"}}
                for item in output_paths:
                    table, column = item.get("table"), item.get("column")
                    if table not in allowed or column not in allowed[table]:
                        raise ValueError("Backup contains an unsupported output reference")
                    relative = _safe_relative(item.get("relative_path"))
                    if len(relative.parts) < 1:
                        raise ValueError("Backup contains an incomplete output reference")
                    resolved_path = str(restored_output.joinpath(*relative.parts))
                    conn.execute(f"UPDATE {table} SET {column}=? WHERE mission_id=? AND {column}=?",
                        (resolved_path, mission, item.get("original_path")))
        if restored_output:
            os.replace(output_stage, restored_output)
        try:
            os.replace(staged_mission, target)
        except BaseException:
            if restored_output and restored_output.exists():
                shutil.rmtree(restored_output)
            raise
    return {"mission_id": mission, "mission_name": manifest.get("mission_name", ""),
        "restored_to": str(target), "output_restored_to": str(restored_output) if restored_output else None,
        "plate_crops_restored": any(name.startswith("output/plate-crops/") for name in output_entries),
        "missing_outputs": manifest.get("missing_outputs", [])}
