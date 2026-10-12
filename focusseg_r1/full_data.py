"""Seven-source segmentation index and lazy mask loading.

Build once on a compute node. Every rank reads one immutable SQLite database;
large JSON/pickle annotation dictionaries are never loaded by training ranks.
Only trusted, explicitly configured pickle files are accepted by the builder.
"""
from __future__ import annotations

import hashlib
import json
import math
import pickle
import random
import sqlite3
from collections import Counter
from functools import lru_cache
from pathlib import Path

from .refvos_data import read_refvos_index, load_mask_dictionary, sample_frame_indices

SCHEMA = "focusseg_full_v1"
SOURCES = {"refcoco", "refcoco_plus", "refcocog", "rvos", "mevis", "revos", "refsav"}


def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256(":".join(map(str, parts)).encode()).digest()[:8], "big")


def config_digest(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def file_digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("schema") != SCHEMA or set(config.get("sources", {})) != SOURCES:
        raise ValueError("Config must explicitly contain all seven agreed sources")
    if not 0 < config["dev_fraction"] < 0.2 or config["frames"] != 5:
        raise ValueError("Expected a small group-disjoint dev fraction and five video frames")
    if config["projection_mode"] != "freeze" or config["loss"] != "full_pixel_bce2_dice0.5":
        raise ValueError("This entry implements frozen projection and full-pixel BCE/Dice")
    return config


def group_split(group, config):
    return "dev" if stable_seed(config["seed"], "split", group) / 2**64 < config["dev_fraction"] else "train"


def video_group(video_id):
    # Shared IDs across sources share a split. Renamed copies require an external
    # content-deduplication audit; we do not claim content-level decontamination.
    return "video:" + str(video_id).replace("\\", "/").rstrip("/").split("/")[-1]


def _json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def json_mask_value(value):
    """Trusted RVOS pickles may hold byte-string COCO RLE counts."""
    if isinstance(value, bytes):
        return value.decode("ascii")
    if isinstance(value, dict):
        return {str(k): json_mask_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_mask_value(v) for v in value]
    if hasattr(value, "tolist"):
        return json_mask_value(value.tolist())
    return value


def _trusted_refs(path):
    with Path(path).open("rb") as handle:
        result = pickle.load(handle)
    if not isinstance(result, list):
        raise ValueError(f"Expected trusted REFER refs list: {path}")
    return result


def resolve_coco_image(directory, file_name):
    name = Path(file_name).name
    candidates = [directory / "coco2014" / name,
                  directory / "coco2014" / "train2014" / name,
                  directory / "coco2014" / "val2014" / name,
                  directory / "coco2014" / "images" / "train2014" / name,
                  directory / "coco2014" / "images" / "val2014" / name,
                  directory / "coco2014" / "images" / name]
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(f"COCO image {name}; checked: {candidates}")


class IndexBuilder:
    def __init__(self, config, output):
        self.config = config
        self.root = Path(config["data_root"])
        self.output = Path(output)
        if self.output.exists():
            raise FileExistsError(self.output)
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.output)
        self.db.executescript("""
            CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE media(id TEXT PRIMARY KEY, payload TEXT NOT NULL);
            CREATE TABLE masks(id TEXT PRIMARY KEY, payload TEXT NOT NULL);
            CREATE TABLE samples(id INTEGER PRIMARY KEY, uid TEXT UNIQUE NOT NULL,
              source TEXT NOT NULL, group_id TEXT NOT NULL, media_id TEXT NOT NULL,
              text TEXT NOT NULL, mask_spec TEXT NOT NULL, split TEXT NOT NULL);
            CREATE INDEX sample_split ON samples(split, source);
        """)
        self.stats = Counter()
        self.inputs = {}

    def input_file(self, path):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        self.inputs[str(path)] = {"bytes": path.stat().st_size, "sha256": file_digest(path)}
        return path

    def media(self, key, payload):
        self.db.execute("INSERT INTO media VALUES (?,?)", (key, json.dumps(payload)))

    def mask(self, key, payload):
        self.db.execute("INSERT INTO masks VALUES (?,?)", (key, json.dumps(json_mask_value(payload))))

    def sample(self, uid, source, group, media, text, mask_spec):
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Missing target text: {uid}")
        split = group_split(group, self.config)
        self.db.execute("INSERT INTO samples(uid,source,group_id,media_id,text,mask_spec,split) VALUES (?,?,?,?,?,?,?)",
                        (uid, source, group, media, text.strip(), json.dumps(mask_spec), split))
        self.stats[f"{source}:{split}"] += 1

    def build(self):
        try:
            coco_refs = {}
            protected_images = set()
            for source, spec in self.config["sources"].items():
                if spec["kind"] == "coco":
                    refs = _trusted_refs(self.input_file(self.root / spec["directory"] / spec["refs"]))
                    coco_refs[source] = refs
                    protected_images.update(int(ref["image_id"]) for ref in refs if ref["split"] != "train")
            protected_videos = set()
            missing_protected_metadata = []
            for relative in self.config["protected_video_metadata"]:
                path = self.root / relative
                if not path.is_file():
                    missing_protected_metadata.append(str(path))
                    continue
                payload = _json(self.input_file(path))
                videos = payload.get("videos", payload)
                protected_videos.update(video_group(v) for v in videos)
            # Ref-SAV evaluation isolation is a required known check.
            if not (self.root / "ref_sav_eval/meta_expressions_valid.json").is_file():
                raise FileNotFoundError("Required Ref-SAV evaluation index for leakage exclusion")
            for source, spec in self.config["sources"].items():
                print(f"INDEX_SOURCE_START {source}", flush=True)
                if spec["kind"] == "coco":
                    self.add_coco(source, spec, coco_refs.pop(source), protected_images)
                elif spec["kind"] == "frames":
                    self.add_frames(source, spec, protected_videos)
                elif spec["kind"] == "sav":
                    self.add_sav(source, spec, protected_videos)
                else:
                    raise ValueError(spec)
                self.db.commit()
                print("INDEX_SOURCE_DONE " + json.dumps(dict(self.stats)), flush=True)
            for source in SOURCES:
                if self.stats[f"{source}:train"] == 0 or self.stats[f"{source}:dev"] == 0:
                    raise ValueError(f"Empty train/dev pool for required source {source}")
            report = {
                "schema": SCHEMA, "config_digest": config_digest(self.config),
                "config": self.config, "stats": dict(self.stats), "inputs": self.inputs,
                "missing_optional_protected_metadata": missing_protected_metadata,
                "all_indexed_media_paths_exist": True,
                "pixel_decode_validation": "performed on access; full dataset pixel decode not yet run",
                "deduplication": "COCO image IDs across three sources; shared video basename IDs; no content hashes",
                "ready": True,
            }
            self.db.execute("INSERT INTO meta VALUES ('report',?)", (json.dumps(report),))
            self.db.commit()
            return report
        finally:
            self.db.close()

    def add_coco(self, source, spec, refs, protected):
        directory = self.root / spec["directory"]
        instances = _json(self.input_file(directory / "instances.json"))
        images = {int(item["id"]): item for item in instances["images"]}
        annotations = {int(item["id"]): item for item in instances["annotations"]}
        seen_media, seen_masks = set(), set()
        for ref in refs:
            if ref["split"] != "train":
                continue
            image_id = int(ref["image_id"])
            if image_id in protected:
                self.stats[f"{source}:excluded_eval_image_refs"] += 1
                continue
            image = images[image_id]
            annotation = annotations[int(ref["ann_id"])]
            if int(annotation["image_id"]) != image_id:
                raise ValueError(f"Reference/annotation image mismatch: {source}/{ref['ref_id']}")
            media_id = f"{source}:image:{image_id}"
            mask_id = f"{source}:ann:{annotation['id']}"
            if media_id not in seen_media:
                path = resolve_coco_image(directory, image["file_name"])
                self.media(media_id, {"kind": "image", "path": str(path),
                                      "height": int(image["height"]), "width": int(image["width"])})
                seen_media.add(media_id)
            if mask_id not in seen_masks:
                if not annotation.get("segmentation"):
                    raise ValueError(f"Missing polygon/RLE: {mask_id}")
                self.mask(mask_id, annotation["segmentation"])
                seen_masks.add(mask_id)
            if not ref["sentences"]:
                raise ValueError(f"No referring sentences: {source}/{ref['ref_id']}")
            for number, sentence in enumerate(ref["sentences"]):
                self.sample(f"{source}:{ref['ref_id']}:{sentence.get('sent_id', number)}", source,
                            f"coco2014:{image_id}", media_id, sentence.get("sent", sentence.get("raw")),
                            {"kind": "coco", "ids": [mask_id]})

    def add_frames(self, source, spec, protected):
        excluded_missing = []
        index = read_refvos_index(self.input_file(self.root / spec["meta"]), source=source,
                                 synthetic_anno_ids=spec.get("synthetic_anno_ids", False),
                                 skip_missing_mask_annotation_id=(source == "revos"),
                                 excluded_missing_anno_ids=excluded_missing)
        if excluded_missing:
            self.stats[f"{source}:excluded_missing_anno_id"] = len(excluded_missing)
            print("INDEX_EXCLUDED_MISSING_ANNO_ID " + json.dumps({
                "source": source, "count": len(excluded_missing),
                "first_examples": excluded_missing[:5],
            }), flush=True)
        masks = load_mask_dictionary(self.input_file(self.root / spec["masks"]))
        seen_media, seen_masks = set(), set()
        for expression in index:
            group = video_group(expression.video_id)
            if group in protected:
                self.stats[f"{source}:excluded_eval_expressions"] += 1
                continue
            media_id = f"{source}:{expression.video_id}"
            if media_id not in seen_media:
                directory = self.root / spec["images"] / expression.video_id
                for frame in expression.frame_names:
                    if not (directory / f"{frame}.jpg").is_file():
                        raise FileNotFoundError(directory / f"{frame}.jpg")
                self.media(media_id, {"kind": "frames", "directory": str(directory), "frames": expression.frame_names})
                seen_media.add(media_id)
            ids = []
            for raw_id in expression.mask_anno_ids:
                sequence = masks[raw_id]
                if len(sequence) != len(expression.frame_names):
                    raise ValueError(f"Mask/frame count mismatch: {source}/{raw_id}")
                mask_id = f"{source}:{raw_id}"
                if mask_id not in seen_masks:
                    self.mask(mask_id, sequence)
                    seen_masks.add(mask_id)
                ids.append(mask_id)
            self.sample(f"{source}:{expression.video_id}:{expression.expression_id}", source, group,
                        media_id, expression.text, {"kind": "sequence", "ids": ids})

    def add_sav(self, source, spec, protected):
        metadata = _json(self.input_file(self.root / spec["meta"]))
        directory = self.root / spec["videos"]
        videos = {}
        for path in directory.glob("*/*.mp4"):
            if path.stem in videos:
                raise ValueError(f"Duplicate SA-V video ID: {path.stem}")
            videos[path.stem] = path
        for number, (video_id, item) in enumerate(metadata.items()):
            if item.get("video_id", video_id) != video_id:
                raise ValueError(f"SA-V video ID mismatch: {video_id}")
            if video_group(video_id) in protected:
                self.stats[f"{source}:excluded_eval_expressions"] += len(item["objects"])
                continue
            path = videos.get(video_id)
            if path is None:
                raise FileNotFoundError(f"SA-V {video_id} under {directory}")
            anno = path.with_name(f"{video_id}_manual.json")
            payload = _json(anno)
            if payload["video_id"] != video_id or payload["video_split"] != "train":
                raise ValueError(f"Unexpected SA-V split/ID: {anno}")
            ids = [int(i) for i in payload["masklet_id"]]
            length = len(payload["masklet"])
            if len(ids) != len(set(ids)) or not length or any(len(frame) != len(ids) for frame in payload["masklet"]):
                raise ValueError(f"Malformed masklet: {anno}")
            frame_count = int(payload["video_frame_count"])
            if math.ceil(frame_count / 4) != length:
                raise ValueError(f"SA-V [::4] frame/masklet mismatch: {anno}")
            media_id = f"refsav:{video_id}"
            self.media(media_id, {"kind": "sav", "path": str(path), "annotation": str(anno),
                                  "frames": length, "raw_frames": frame_count,
                                  "height": int(payload["video_height"]), "width": int(payload["video_width"])})
            for object_key, obj in item["objects"].items():
                object_id = int(obj.get("obj_id", object_key))
                if object_id not in ids:
                    raise ValueError(f"Missing SA-V masklet object {object_id}: {anno}")
                self.sample(f"refsav:{video_id}:{object_key}", source, video_group(video_id), media_id,
                            obj.get("formated"), {"kind": "sav", "object_id": object_id})
            if number % 500 == 0:
                self.db.commit()
                print(f"INDEX_REFSAV_VIDEO {number}/{len(metadata)}", flush=True)


def decode_segmentation(segmentation, height, width):
    import numpy as np
    from pycocotools import mask as masks
    if segmentation is None:
        return np.zeros((height, width), dtype=bool)
    if isinstance(segmentation, list):
        rle = masks.merge(masks.frPyObjects(segmentation, height, width))
    elif isinstance(segmentation.get("counts"), list):
        rle = masks.frPyObjects(segmentation, height, width)
    else:
        rle = dict(segmentation)
        if isinstance(rle.get("counts"), str):
            rle["counts"] = rle["counts"].encode("ascii")
    decoded = masks.decode(rle)
    if decoded.ndim == 3:
        decoded = decoded.any(axis=2)
    if decoded.shape != (height, width):
        raise ValueError(f"Mask/image shape mismatch {decoded.shape} != {(height, width)}")
    return decoded.astype(bool)


def read_sav_frames(path, raw_indices, expected_count):
    """Sequential grab keeps exact decoded frame ordinals; no approximate seeking."""
    import cv2
    from PIL import Image
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    result = {}
    try:
        actual_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if actual_count != expected_count:
            raise ValueError(f"Video metadata frame count {actual_count} != annotation {expected_count}: {path}")
        wanted = set(raw_indices)
        for index in range(max(wanted) + 1):
            if not capture.grab():
                raise RuntimeError(f"Video decode ends at frame {index}: {path}")
            if index in wanted:
                ok, frame = capture.retrieve()
                if not ok:
                    raise RuntimeError(f"Cannot decode frame {index}: {path}")
                result[index] = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    return [result[index].copy() for index in raw_indices]


class FullDataStore:
    def __init__(self, path, config=None):
        self.path = Path(path).resolve()
        self.db = sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True)
        self.db.row_factory = sqlite3.Row
        try:
            row = self.db.execute("SELECT value FROM meta WHERE key='report'").fetchone()
            if row is None:
                raise ValueError("Index build did not finish; refusing partial database")
            self.report = json.loads(row[0])
            if not self.report.get("ready") or self.report["schema"] != SCHEMA:
                raise ValueError("Unready/incompatible data index")
            if config is not None and self.report["config_digest"] != config_digest(config):
                raise ValueError("Index configuration differs from requested training config")
        except Exception:
            self.db.close()
            raise

    def ids(self, split):
        return [(row["id"], row["uid"], row["source"]) for row in self.db.execute(
            "SELECT id,uid,source FROM samples WHERE split=? ORDER BY id", (split,))]

    def fixed_ids(self, split, per_source, seed):
        result = []
        all_rows = self.ids(split)
        for source in sorted(SOURCES):
            rows = [row for row in all_rows if row[2] == source]
            rows.sort(key=lambda row: stable_seed(seed, "fixed", row[1]))
            if len(rows) < per_source:
                raise ValueError(f"Only {len(rows)} {source}/{split} expressions, need {per_source}")
            result.extend(row[0] for row in rows[:per_source])
        return result

    @lru_cache(maxsize=32)
    def mask(self, mask_id):
        row = self.db.execute("SELECT payload FROM masks WHERE id=?", (mask_id,)).fetchone()
        if row is None:
            raise KeyError(mask_id)
        return json.loads(row[0])

    @lru_cache(maxsize=8)
    def sav_annotation(self, path):
        return _json(path)

    def load(self, sample_id, seed, frame_count=5):
        import numpy as np
        from PIL import Image
        row = self.db.execute("SELECT s.*,m.payload FROM samples s JOIN media m ON s.media_id=m.id WHERE s.id=?",
                              (int(sample_id),)).fetchone()
        if row is None:
            raise KeyError(sample_id)
        sample = dict(row)
        media, mask_spec = json.loads(sample.pop("payload")), json.loads(sample.pop("mask_spec"))
        kind = media["kind"]
        if kind == "image":
            indices = (0,)
            with Image.open(media["path"]) as image:
                images = [image.convert("RGB")]
        else:
            length = len(media["frames"]) if kind == "frames" else media["frames"]
            indices = sample_frame_indices(length, frame_count, random.Random(seed))
            if kind == "frames":
                images = []
                for index in indices:
                    with Image.open(Path(media["directory"]) / (media["frames"][index] + ".jpg")) as image:
                        images.append(image.convert("RGB"))
            elif kind == "sav":
                images = read_sav_frames(media["path"], [i * 4 for i in indices], media["raw_frames"])
            else:
                raise ValueError(kind)
        width, height = images[0].size
        if any(image.size != (width, height) for image in images):
            raise ValueError(f"Frame size mismatch: {sample['uid']}")
        if "height" in media and (height, width) != (media["height"], media["width"]):
            raise ValueError(f"Media/annotation dimensions mismatch: {sample['uid']}")
        masks = []
        if kind == "sav":
            annotation = self.sav_annotation(media["annotation"])
            object_index = [int(i) for i in annotation["masklet_id"]].index(mask_spec["object_id"])
            masks = [decode_segmentation(annotation["masklet"][i][object_index], height, width) for i in indices]
        else:
            for index in indices:
                union = np.zeros((height, width), dtype=bool)
                for mask_id in mask_spec["ids"]:
                    payload = self.mask(mask_id)
                    if kind == "frames":
                        if len(payload) != len(media["frames"]):
                            raise ValueError(f"Sequence length mismatch: {sample['uid']}")
                        payload = payload[index]
                    union |= decode_segmentation(payload, height, width)
                masks.append(union)
        return {**sample, "images": images, "masks": np.stack(masks), "frame_indices": indices}

    def close(self):
        self.mask.cache_clear()
        self.sav_annotation.cache_clear()
        self.db.close()
