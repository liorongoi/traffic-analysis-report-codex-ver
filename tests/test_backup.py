import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
import zipfile
from PIL import Image
from road_traffic.backup import create_backup, restore_backup, verify_backup
from road_traffic.store import Library


class MissionBackupTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.workspace = self.root / "workspace"
        self.library = Library(self.workspace)
        self.source = self.root / "input"
        self.source.mkdir()
        Image.new("RGB", (16, 12), (40, 80, 120)).save(self.source / "17052026-1604-camera.jpg")
        self.mission = self.library.create("Backup sample", "Lebanon", "Asia/Beirut")
        self.library.import_folder(self.mission, self.source, "camera")
        self.archive = self.root / "mission-backup.zip"

    def tearDown(self):
        self.tmp.cleanup()

    def test_backup_restores_mission_database_and_source_snapshot(self):
        result = create_backup(self.library, self.mission, self.archive)
        self.assertEqual(result["mission_id"], self.mission)
        self.assertTrue(verify_backup(self.archive)["valid"])
        restored = Library(self.root / "restored-workspace")
        result = restore_backup(restored, self.archive)
        self.assertEqual(result["mission_id"], self.mission)
        self.assertEqual(restored.missions()[0]["name"], "Backup sample")
        self.assertEqual(restored.image_path(self.mission, restored.images(self.mission)[0]["id"]).read_bytes(),
            (self.source / "17052026-1604-camera.jpg").read_bytes())

    def test_backup_never_overwrites_existing_destination_or_mission(self):
        create_backup(self.library, self.mission, self.archive)
        with self.assertRaisesRegex(ValueError, "already exists"):
            create_backup(self.library, self.mission, self.archive)
        with self.assertRaisesRegex(ValueError, "will not overwrite"):
            restore_backup(self.library, self.archive)
        self.assertEqual(len(self.library.missions()), 1)

    def test_checksum_tampering_is_rejected_before_restore(self):
        create_backup(self.library, self.mission, self.archive)
        altered = self.root / "altered.zip"
        with zipfile.ZipFile(self.archive) as source, zipfile.ZipFile(altered, "w") as target:
            for item in source.infolist():
                data = source.read(item.filename)
                if item.filename.startswith("mission/sources/"):
                    data = data + b"tampered"
                target.writestr(item.filename, data)
        with self.assertRaisesRegex(ValueError, "size check failed|checksum failed"):
            verify_backup(altered)
        self.assertFalse((self.root / "bad-restore" / self.mission).exists())

    def test_unsafe_archive_path_is_rejected(self):
        create_backup(self.library, self.mission, self.archive)
        altered = self.root / "unsafe.zip"
        with zipfile.ZipFile(self.archive) as source, zipfile.ZipFile(altered, "w") as target:
            manifest = json.loads(source.read("manifest.json"))
            manifest["files"].append({"path": "../escape", "size": 0, "sha256": ""})
            target.writestr("manifest.json", json.dumps(manifest))
            for item in source.infolist():
                if item.filename != "manifest.json":
                    target.writestr(item.filename, source.read(item.filename))
        with self.assertRaisesRegex(ValueError, "unsafe path"):
            verify_backup(altered)

    def test_backup_carries_external_plate_chips_and_review_database(self):
        output = self.root / "plate-output"
        output.mkdir()
        mission = self.library.create("Plate backup", "Lebanon", "Asia/Beirut", {
            "analysis_mode": "license_plate", "output_root": str(output)})
        source = self.root / "plate-input"
        source.mkdir()
        Image.new("RGB", (16, 12), (10, 20, 30)).save(source / "17052026-1605-plate.jpg")
        self.library.import_folder(mission, source, "camera")
        report = output / "traffic-report.csv"
        report.write_text("mission,images\n", encoding="utf-8")
        with self.library.connect(mission) as conn:
            conn.execute("INSERT INTO exports(id,mission_id,format,output_path,row_count,status,created) VALUES (?,?,?,?,?,?,?)",
                ("backup-export", mission, "csv", str(report), 0, "completed", "2026-10-29T00:00:00+00:00"))
        chips = output / "plate-crops" / mission
        chips.mkdir(parents=True)
        (chips / "chip-1.jpg").write_bytes(b"saved plate chip")
        review_db = sqlite3.connect(chips / "review.sqlite3")
        review_db.execute("PRAGMA journal_mode=WAL")
        review_db.execute("CREATE TABLE reviewed_chip (id TEXT PRIMARY KEY, decision TEXT)")
        review_db.execute("INSERT INTO reviewed_chip VALUES ('chip-1','approved')")
        review_db.commit()
        review_db.close()
        archive = self.root / "plate-mission.zip"
        create_backup(self.library, mission, archive)

        restored_output = self.root / "restored-plate-output"
        restored_output.mkdir()
        restored = Library(self.root / "plate-restored-workspace")
        result = restore_backup(restored, archive, plate_output_root=restored_output)
        self.assertTrue(result["plate_crops_restored"])
        restored_root = restored_output / "roadloom-restored" / mission
        self.assertEqual((restored_root / "plate-crops" / mission / "chip-1.jpg").read_bytes(), b"saved plate chip")
        self.assertEqual((restored_root / "traffic-report.csv").read_text(encoding="utf-8"), "mission,images\n")
        self.assertEqual(restored.settings(mission)["output_root"], str(restored_root.resolve()))
        with restored.connect(mission) as conn:
            self.assertEqual(conn.execute("SELECT output_path FROM exports WHERE id='backup-export'").fetchone()[0],
                str(restored_root.resolve() / "traffic-report.csv"))
        with sqlite3.connect(restored_root / "plate-crops" / mission / "review.sqlite3") as conn:
            self.assertEqual(conn.execute("SELECT decision FROM reviewed_chip WHERE id='chip-1'").fetchone()[0], "approved")


if __name__ == "__main__":
    unittest.main()
