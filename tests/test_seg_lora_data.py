import json
from pathlib import Path
import tempfile
import unittest
from seg_lora.data import annotation_ids, CompleteIndexBuilder, read_config
from focusseg_r1.full_runtime import epoch_plan
from focusseg_r1.full_data import FullDataStore, SCHEMA


class FreshDataTests(unittest.TestCase):
    def test_explicit_empty_is_kept(self):
        self.assertEqual(annotation_ids({"anno_id": []}), [])

    def test_missing_or_null_not_negative(self):
        for item in ({}, {"anno_id": None}, {"anno_id": [None]}):
            with self.assertRaises(ValueError):
                annotation_ids(item)

    def test_union_and_synthetic_ids(self):
        self.assertEqual(annotation_ids({"anno_id": [1, 2]}), ["1", "2"])
        self.assertEqual(annotation_ids({}, synthetic=12), ["12"])

    def test_no_private_holdout(self):
        with tempfile.TemporaryDirectory() as directory:
            builder = CompleteIndexBuilder({"data_root": directory}, Path(directory) / "new.sqlite")
            builder.sample("revos:v:0", "revos", "video:v", "v", "missing target", {"kind": "sequence", "ids": []})
            row = builder.db.execute("SELECT split,mask_spec FROM samples").fetchone()
            self.assertEqual(row[0], "train")
            self.assertEqual(json.loads(row[1])["ids"], [])
            builder.db.close()

    def test_three_epochs_four_rank_coverage(self):
        for epoch in range(3):
            plan, padding = epoch_plan(list(range(67)), 20261002, epoch, 32)
            self.assertEqual(len(plan), 96)
            self.assertEqual(padding, 29)
            self.assertEqual(set(plan), set(range(67)))
            self.assertEqual(sum(len(plan[r::4]) for r in range(4)), 96)

    def test_explicit_negative_decodes_all_zero(self):
        try:
            from PIL import Image
            import numpy as np
        except ImportError:
            self.skipTest("Pillow/numpy required for decode fixture")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for frame in ("00000", "00001"):
                Image.new("RGB", (12, 8)).save(root / f"{frame}.jpg")
            builder = CompleteIndexBuilder({"data_root": directory}, root / "new.sqlite")
            builder.media("v", {"kind": "frames", "directory": directory, "frames": ["00000", "00001"]})
            builder.sample("revos:v:0", "revos", "video:v", "v", "absent target", {"kind": "sequence", "ids": []})
            builder.db.execute("INSERT INTO meta VALUES ('report',?)", (json.dumps({"schema": SCHEMA, "ready": True}),))
            builder.db.commit()
            builder.db.close()
            store = FullDataStore(root / "new.sqlite")
            sample = store.load(1, 123, 5)
            self.assertEqual(sample["masks"].shape, (5, 8, 12))
            self.assertFalse(np.any(sample["masks"]))
            store.close()

    def test_configuration(self):
        config = read_config(Path(__file__).resolve().parents[1] / "configs/seg_lora_fresh_3epoch.json")
        self.assertEqual(config["query_module"], "absent")
        self.assertEqual(config["validation"], "official_refsav_full_1945")
        self.assertEqual(config["world_size"] * config["gradient_accumulation"], 32)


if __name__ == "__main__":
    unittest.main()
