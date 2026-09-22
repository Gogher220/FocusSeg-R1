from __future__ import annotations


def parse_trainable_patterns(patterns: str | list[str] | tuple[str, ...] | None) -> list[str]:
    if patterns is None:
        return []
    if isinstance(patterns, str):
        return [item.strip() for item in patterns.split(",") if item.strip()]
    return [str(item).strip() for item in patterns if str(item).strip()]


def set_trainable_by_patterns(
    model,
    patterns: str | list[str] | tuple[str, ...] | None,
) -> list[str]:
    parsed = parse_trainable_patterns(patterns)
    for name, param in model.named_parameters():
        param.requires_grad = any(pattern in name for pattern in parsed)
    return parsed


def prepare_model_for_checkpointing(model) -> None:
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
        return
    base_model = getattr(model, "base_model", None)
    if base_model is not None and hasattr(base_model, "enable_input_require_grads"):
        base_model.enable_input_require_grads()


def print_trainable_parameters(model, prefix: str = "Trainable parameters") -> tuple[int, int, float]:
    trainable = 0
    total = 0
    for _, param in model.named_parameters():
        numel = param.numel()
        total += numel
        if param.requires_grad:
            trainable += numel
    pct = 100 * trainable / total if total else 0.0
    print(f"{prefix}: {trainable:,} / {total:,} ({pct:.4f}%)", flush=True)
    return trainable, total, pct
