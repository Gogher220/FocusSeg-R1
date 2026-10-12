import json
from pathlib import Path
import tempfile
import unittest

from local_scripts.seg_lora_storage import (
    atomic_checkpoint, compatible_provenance, keep_latest_best, recover_best,
)


class StorageTests(unittest.TestCase):
    def test_failed_write_preserves_old_checkpoint_and_removes_new_partial(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "checkpoint_00016000.pt"
            target.write_bytes(b"old-complete-checkpoint")
            def fail(value, stream):
                stream.write(b"incomplete")
                raise OSError("quota exceeded")
            with self.assertRaises(OSError):
                atomic_checkpoint(target, {}, fail)
            self.assertEqual(target.read_bytes(), b"old-complete-checkpoint")
            self.assertEqual(list(Path(folder).glob("*.partial")), [])

    def test_retention_keeps_latest_and_best_and_preserves_reports(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for step in (0, 10000, 14346, 15500, 16000):
                (root / f"checkpoint_{step:08d}.pt").write_bytes(b"checkpoint")
            (root / "validation_00010000.json").write_text("{}")
            (root / "bridge.pt").write_bytes(b"other")
            deleted = keep_latest_best(root, "checkpoint_00016000.pt", "checkpoint_00014346.pt")
            self.assertEqual(len(deleted), 3)
            self.assertEqual(len(list(root.glob("checkpoint_*.pt"))), 2)
            self.assertTrue((root / "validation_00010000.json").exists())
            self.assertTrue((root / "bridge.pt").exists())
            keep_latest_best(root, "checkpoint_00016000.pt", "checkpoint_00016000.pt")
            self.assertEqual(len(list(root.glob("checkpoint_*.pt"))), 1)

    def test_missing_best_prevents_pruning(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "checkpoint_00016000.pt"
            target.write_bytes(b"complete")
            with self.assertRaises(ValueError):
                keep_latest_best(folder, target.name, "checkpoint_00014346.pt")
            self.assertTrue(target.exists())

    def test_storage_migration_rejects_training_changes(self):
        old = {"config_digest": "config", "index_sha256": "index", "code": {
            "local_scripts/seg_lora_run.py": "old", "seg_lora/model.py": "model"}}
        new = {**old, "code": {**old["code"], "local_scripts/seg_lora_run.py": "new",
                              "local_scripts/seg_lora_storage.py": "storage"}}
        self.assertTrue(compatible_provenance(old, new, old))
        self.assertFalse(compatible_provenance(old, new))
        changed = {**new, "code": {**new["code"], "seg_lora/model.py": "different"}}
        self.assertFalse(compatible_provenance(old, changed, old))
        self.assertFalse(compatible_provenance(old, {**new, "index_sha256": "different"}, old))
        self.assertFalse(compatible_provenance(old, {**new, "config_digest": "different"}, old))

    def test_recover_newer_best_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            best = {"file": "checkpoint_00016000.pt", "step": 16000, "score": .6}
            (root / best["file"]).write_bytes(b"complete")
            (root / "best.json").write_text(json.dumps(best))
            self.assertEqual(recover_best(root, {"score": .58}), best)


if __name__ == "__main__":
    unittest.main()
