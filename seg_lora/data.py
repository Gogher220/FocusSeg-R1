"""Complete seven-source training index; no private holdout or silent filtering."""
import json
from pathlib import Path

from focusseg_r1.full_data import (
    IndexBuilder, FullDataStore, SOURCES, SCHEMA, config_digest, video_group,
    _trusted_refs, load_mask_dictionary,
)

REVOS_SHA = "3917dfba8725d080c8658dc43ad4a84cf41ce51438acfaee70bef5aa0310417a"


def annotation_ids(item, *, synthetic=None):
    if synthetic is not None:
        return [str(synthetic)]
    # An explicit [] is a negative target. Missing/null is NOT equivalent.
    if "anno_id" not in item or item["anno_id"] is None:
        raise ValueError("Missing/null anno_id; do not invent a negative target")
    value = item["anno_id"]
    if not isinstance(value, list):
        value = [value]
    if any(v is None or isinstance(v, (list, dict)) for v in value):
        raise ValueError("Invalid annotation ID")
    return [str(v) for v in value]


def read_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("experiment") != "seg_lora_fresh_v1":
        raise ValueError("Not a fresh SEG LoRA configuration")
    if set(config["sources"]) != SOURCES or config["epochs"] != 3 or config["world_size"] != 4:
        raise ValueError("Expected seven sources, three epochs, four GPUs")
    if config["initialization"] != "original_weights_only" or config["frames"] != 5:
        raise ValueError("Old query checkpoints and changed frame budgets are not supported")
    return config


class CompleteIndexBuilder(IndexBuilder):
    def sample(self, uid, source, group, media, text, mask_spec):
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Missing target text: {uid}")
        self.db.execute("INSERT INTO samples(uid,source,group_id,media_id,text,mask_spec,split) VALUES (?,?,?,?,?,?,?)",
                        (uid, source, group, media, text.strip(), json.dumps(mask_spec), "train"))
        self.stats[f"{source}:train"] += 1

    def add_frames(self, source, spec, protected):
        meta_path = self.input_file(self.root / spec["meta"])
        if source == "revos" and self.inputs[str(meta_path)]["sha256"] != REVOS_SHA:
            raise ValueError("ReVOS training metadata differs from the verified official release")
        videos = json.loads(meta_path.read_text(encoding="utf-8"))["videos"]
        masks = load_mask_dictionary(self.input_file(self.root / spec["masks"]))
        seen_masks = set()
        running = 0
        for video_id, video in videos.items():
            frames = sorted(video["frames"])
            if not frames or len(frames) != len(set(frames)):
                raise ValueError(f"Invalid frame list: {source}/{video_id}")
            directory = self.root / spec["images"] / video_id
            for frame in frames:
                if not (directory / f"{frame}.jpg").is_file():
                    raise FileNotFoundError(directory / f"{frame}.jpg")
            media_id = f"{source}:{video_id}"
            self.media(media_id, {"kind": "frames", "directory": str(directory), "frames": frames})
            for exp_id in sorted(video["expressions"]):
                item = video["expressions"][exp_id]
                raw_ids = annotation_ids(item, synthetic=running if spec.get("synthetic_anno_ids") else None)
                running += 1
                ids = []
                for raw_id in raw_ids:
                    sequence = masks[raw_id]
                    if len(sequence) != len(frames):
                        raise ValueError(f"Frame/mask mismatch: {source}/{raw_id}")
                    key = f"{source}:{raw_id}"
                    if key not in seen_masks:
                        self.mask(key, sequence)
                        seen_masks.add(key)
                    ids.append(key)
                if not ids:
                    self.stats[f"{source}:explicit_empty_targets"] += 1
                self.sample(f"{source}:{video_id}:{exp_id}", source, video_group(video_id),
                            media_id, item["exp"], {"kind": "sequence", "ids": ids})

    def build(self):
        try:
            for source, spec in self.config["sources"].items():
                print(f"INDEX_SOURCE_START {source}", flush=True)
                if spec["kind"] == "coco":
                    refs = _trusted_refs(self.input_file(self.root / spec["directory"] / spec["refs"]))
                    self.add_coco(source, spec, refs, set())
                elif spec["kind"] == "frames":
                    self.add_frames(source, spec, set())
                else:
                    self.add_sav(source, spec, set())
                self.db.commit()
                print("INDEX_COUNTS " + json.dumps(dict(self.stats)), flush=True)
            if self.stats["revos:explicit_empty_targets"] != 114 or self.stats["revos:train"] != 29252:
                raise ValueError("Unexpected verified ReVOS release counts")
            if self.stats["refsav:train"] != 72509:
                raise ValueError("Full Ref-SAV training set must contain 72509 expressions")
            if any(self.stats[f"{s}:train"] <= 0 for s in SOURCES):
                raise ValueError("An agreed training source is empty")
            # Only this official validation set is used for checkpoint selection.
            # Other datasets' validation overlap is not silently removed from training.
            meta = self.input_file(self.root / "ref_sav_eval/meta_expressions_valid.json")
            payload = json.loads(meta.read_text(encoding="utf-8"))["videos"]
            n_eval = sum(len(v["expressions"]) for v in payload.values())
            if n_eval != 1945:
                raise ValueError(f"Unexpected Ref-SAV validation expression count: {n_eval}")
            trained = {r[0] for r in self.db.execute("SELECT DISTINCT group_id FROM samples")}
            overlap = sorted(trained & {video_group(v) for v in payload})
            if overlap:
                raise ValueError(f"Official Ref-SAV validation overlaps training: {overlap[:10]}")
            self.input_file(self.root / "ref_sav_eval/mask_dict.json")
            report = {"schema": SCHEMA, "experiment": self.config["experiment"], "ready": True,
                      "config": self.config, "config_digest": config_digest(self.config),
                      "stats": dict(self.stats), "inputs": self.inputs,
                      "private_holdout": False, "official_refsav_val_expressions": n_eval,
                      "official_refsav_train_overlap": overlap,
                      "pixel_decode_validation": "sampled preflight only, not all media",
                      "alignment": "complete seven segmentation training sources; not full official multitask recipe"}
            self.db.execute("INSERT INTO meta VALUES ('report',?)", (json.dumps(report),))
            self.db.commit()
            return report
        finally:
            self.db.close()


def open_store(path, config):
    store = FullDataStore(path, config)
    if store.report.get("experiment") != "seg_lora_fresh_v1":
        store.close()
        raise ValueError("Historical query index rejected")
    return store
