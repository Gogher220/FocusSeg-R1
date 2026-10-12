"""CPU-only, testable scheduling and checkpoint selection for the full run."""
from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path


def epoch_plan(sample_ids, seed, epoch, global_batch):
    if not sample_ids or global_batch < 1:
        raise ValueError("Empty data or invalid global batch")
    plan = list(sample_ids)
    random.Random(seed + epoch).shuffle(plan)
    padding = (-len(plan)) % global_batch
    original = list(plan)
    plan.extend(original[i % len(original)] for i in range(padding))
    return plan, padding


def rank_microbatches(plan, offset, world_size, rank, accumulation):
    width = world_size * accumulation
    if not 0 <= rank < world_size or offset % width or offset + width > len(plan):
        raise ValueError("Invalid distributed sampling cursor")
    return [(offset + i * world_size + rank, plan[offset + i * world_size + rank])
            for i in range(accumulation)]


def lr_multiplier(step, total, warmup_ratio):
    warmup = max(1, math.ceil(total * warmup_ratio))
    if step < warmup:
        return (step + 1) / warmup
    return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, total - warmup))))


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def update_stopping(state, score, train_loss, progress, config):
    """Stop sustained degradation; plateau is a review flag, not proof of failure."""
    if not math.isfinite(score) or not math.isfinite(train_loss):
        raise FloatingPointError("Nonfinite fixed evaluation metric")
    state = dict(state or {})
    delta = config["early_stop_min_delta"]
    best = state.get("best", score)
    improved = score > best + delta
    state["best"] = max(best, score)
    eligible = progress >= max(config["early_stop_min_epoch_fraction"], config["warmup_ratio"])
    state["bad"] = (state.get("bad", 0) + 1 if eligible and score < best - delta else 0)
    state["flat"] = (state.get("flat", 0) + 1 if eligible and not improved else 0)
    initial_loss = state.setdefault("initial_train_loss", train_loss)
    # Fixed probes are diagnostic; do not automatically call a stable low loss unfitted.
    no_fit = train_loss >= initial_loss * 0.99
    state["no_fit"] = state.get("no_fit", 0) + 1 if eligible and no_fit else 0
    patience = config["early_stop_patience"]
    state["stop"] = state["bad"] >= patience or state["no_fit"] >= patience
    state["reason"] = ("sustained_dev_regression" if state["bad"] >= patience else
                       "fixed_train_probe_not_improving" if state["no_fit"] >= patience else "continue")
    state["plateau_review"] = state["flat"] >= patience
    return state


def retained_checkpoints(entries, latest, best):
    """Return only manifest-listed basenames; initial/latest/best are retained."""
    if not entries:
        return set()
    ordered = sorted(entries, key=lambda e: e["step"])
    top = sorted(entries, key=lambda e: (e["score"], e["step"]), reverse=True)[:best]
    return {e["file"] for e in ordered[-latest:] + top + [ordered[0]]}
