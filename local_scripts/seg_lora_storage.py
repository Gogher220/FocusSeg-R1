"""CPU-only checkpoint safety and the reviewed storage-policy migration."""
import json
import os
from pathlib import Path
import re
import tempfile

CHECKPOINT = re.compile(r"checkpoint_([0-9]{8})\.pt\Z")
RUNNER = "local_scripts/seg_lora_run.py"
STORAGE = "local_scripts/seg_lora_storage.py"


def compatible_provenance(saved, current, reference=None):
    if saved == current:
        return True
    if reference is None or saved != reference:
        return False
    if {k: v for k, v in saved.items() if k != "code"} != {
        k: v for k, v in current.items() if k != "code"
    }:
        return False
    old, new = saved.get("code", {}), current.get("code", {})
    if RUNNER not in old or RUNNER not in new or STORAGE not in new:
        return False
    # Only the runner's storage logic and this new helper are allowed to change.
    return {k: v for k, v in old.items() if k not in (RUNNER, STORAGE)} == {
        k: v for k, v in new.items() if k not in (RUNNER, STORAGE)
    }


def atomic_checkpoint(path, payload, writer):
    """Never truncate the old checkpoint; remove only our failed temporary file."""
    path = Path(path)
    fd, name = tempfile.mkstemp(prefix=path.stem + ".", suffix=".partial", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as handle:
            writer(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def keep_latest_best(directory, latest_name, best_name=None):
    """Prune exact checkpoint filenames only, after both retained files exist."""
    directory = Path(directory).resolve()
    keep = {latest_name}
    if best_name:
        keep.add(best_name)
    for name in keep:
        path = directory / name
        if not CHECKPOINT.fullmatch(name) or path.is_symlink() or not path.is_file():
            raise ValueError(f"Missing/unsafe retained checkpoint: {name}")
    candidates = []
    for path in directory.iterdir():
        if CHECKPOINT.fullmatch(path.name) and path.name not in keep:
            if path.is_symlink() or not path.is_file() or path.resolve().parent != directory:
                raise ValueError(f"Unsafe checkpoint deletion target: {path}")
            candidates.append(path)
    # Validate every target before deleting the first file.
    for path in candidates:
        path.unlink()
    return [p.name for p in candidates]


def recover_best(directory, embedded):
    """Best metadata can be newer than a checkpoint saved before validation."""
    path = Path(directory) / "best.json"
    if not path.exists():
        return embedded
    external = json.loads(path.read_text(encoding="utf-8"))
    if not CHECKPOINT.fullmatch(external["file"]) or not (Path(directory) / external["file"]).is_file():
        raise ValueError("best.json points to a missing/unsafe checkpoint")
    if embedded is None or external["score"] > embedded["score"]:
        return external
    return embedded
