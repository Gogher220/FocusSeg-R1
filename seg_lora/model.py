"""SEG-token supervised LoRA. No old query module and no teacher-forced eval."""
from contextlib import contextmanager
import re
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoConfig
from peft import LoraConfig, get_peft_model
from qwen_vl_utils import process_vision_info
from configuration_sa2va_pro import Sa2VAProConfig
from modeling_sa2va_pro import Sa2VAProModel
from sa2va_pro_utils import build_processor
from sa2va_checkpoint_utils import load_official_sa2va_segmentation_components


class SegEmbedding(nn.Module):
    """Task-local single token row; original embedding matrix stays untouched."""
    def __init__(self, original, token_id, row):
        super().__init__()
        self.original = original
        self.token_id = token_id
        self.seg_row = nn.Parameter(row.float().clone())
        self.active = True

    @property
    def weight(self):
        return self.original.weight

    def forward(self, ids):
        value = self.original(ids)
        if self.active:
            value = torch.where((ids == self.token_id)[..., None], self.seg_row.to(value.dtype), value)
        return value


class SegHead(SegEmbedding):
    def forward(self, hidden):
        logits = self.original(hidden)
        if self.active:
            logits = logits.clone()
            logits[..., self.token_id] = F.linear(hidden, self.seg_row[None].to(hidden.dtype)).squeeze(-1)
        return logits


def point_losses(pred, target, config):
    """PointRend-style uncertainty sampling; targets and predictions share coordinates."""
    n = pred.shape[0]
    count = config["mask_points"]
    def sample(tensor, coords):
        return F.grid_sample(tensor.float(), coords[:, :, None] * 2 - 1,
                             align_corners=False).squeeze(1).squeeze(-1)
    with torch.no_grad():
        candidates = torch.rand(n, int(count * config["oversample_ratio"]), 2, device=pred.device)
        uncertainty = -sample(pred, candidates).abs()
        chosen = uncertainty.topk(int(count * config["importance_sample_ratio"]), dim=1).indices
        coords = candidates.gather(1, chosen[..., None].expand(-1, -1, 2))
        coords = torch.cat([coords, torch.rand(n, count - coords.shape[1], 2, device=pred.device)], dim=1)
        truth = sample(target, coords)
    values = sample(pred, coords)
    bce = F.binary_cross_entropy_with_logits(values, truth)
    prob = values.sigmoid()
    dice = (1 - (2 * (prob * truth).sum(1) + 1) / (prob.sum(1) + truth.sum(1) + 1)).mean()
    return bce, dice


class SegLoRA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.recipe = config
        raw = AutoConfig.from_pretrained(config["model"], trust_remote_code=True)
        cfg = Sa2VAProConfig.from_dict(raw.to_dict())
        cfg.base_model_name_or_path = config["model"]
        self.processor, _ = build_processor(config["model"], seg_token="[SEG]", trust_remote_code=True)
        self.seg_id = self.processor.tokenizer.convert_tokens_to_ids("[SEG]")
        if self.processor.tokenizer.encode("[SEG]", add_special_tokens=False) != [self.seg_id]:
            raise ValueError("SEG must be exactly one token")
        self.base = Sa2VAProModel(cfg)
        if len(self.processor.tokenizer) > self.base.get_input_embeddings().weight.shape[0]:
            self.base.resize_token_embeddings(len(self.processor.tokenizer))
        self.base.set_seg_token_id(self.seg_id)
        # Save original rows BEFORE the official segmentation loader replaces them.
        original_in = self.base.get_input_embeddings().weight[self.seg_id].detach().clone()
        original_out = self.base.get_output_embeddings().weight[self.seg_id].detach().clone()
        load_official_sa2va_segmentation_components(self.base, config["seg_init"], self.seg_id)
        seg_in = self.base.get_input_embeddings().weight[self.seg_id].detach().clone()
        seg_out = self.base.get_output_embeddings().weight[self.seg_id].detach().clone()
        with torch.no_grad():
            self.base.get_input_embeddings().weight[self.seg_id].copy_(original_in)
            self.base.get_output_embeddings().weight[self.seg_id].copy_(original_out)
        self.base.requires_grad_(False)
        names = [name for name, module in self.base.model.named_modules()
                 if isinstance(module, nn.Linear) and "visual" not in name and ".layers." in name
                 and name.rsplit(".", 1)[-1] in {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}]
        if not names:
            raise ValueError("No language-only LoRA targets found")
        self.lora_targets = names
        adapter = LoraConfig(r=config["lora_rank"], lora_alpha=config["lora_alpha"],
                             lora_dropout=config["lora_dropout"], bias="none", task_type="CAUSAL_LM",
                             target_modules="^(?:" + "|".join(re.escape(n) for n in names) + ")$")
        self.base.model = get_peft_model(self.base.model, adapter)
        underlying = self.base.model.get_base_model()
        underlying.set_input_embeddings(SegEmbedding(underlying.get_input_embeddings(), self.seg_id, seg_in))
        underlying.set_output_embeddings(SegHead(underlying.get_output_embeddings(), self.seg_id, seg_out))
        self.base.text_hidden_fcs.float().requires_grad_(True)
        # SDPA and checkpointing reduce LoRA activation memory. No hidden-state cache.
        if hasattr(underlying, "set_attn_implementation"):
            underlying.set_attn_implementation("sdpa")
        underlying.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        underlying.config.use_cache = False
        self.assert_contract()

    def assert_contract(self):
        for name, param in self.named_parameters():
            intended = "lora_" in name or name.endswith(".seg_row") or name.startswith("base.text_hidden_fcs.")
            if param.requires_grad != intended:
                raise RuntimeError(f"Trainability contract violated: {name}")

    def train(self, mode=True):
        super().train(mode)
        self.base.grounding_encoder.eval()
        return self

    @property
    def device(self):
        return next(self.parameters()).device

    @contextmanager
    def qa_route(self):
        """Use the ORIGINAL Video-R1 processor/messages for QA; never seg templates."""
        underlying = self.base.model.get_base_model()
        layers = [underlying.get_input_embeddings(), underlying.get_output_embeddings()]
        old = [layer.active for layer in layers]
        was_training = self.training
        self.eval()
        try:
            for layer in layers:
                layer.active = False
            with self.base.model.disable_adapter():
                yield self.base.model
        finally:
            for layer, active in zip(layers, old):
                layer.active = active
            self.train(was_training)

    def inputs(self, images, text, supervised):
        if not images or "[SEG]" in text:
            raise ValueError("Invalid frames or reserved token inside referring expression")
        messages = [{"role": "user", "content":
                    [{"type": "image", "image": im} for im in images[:5]] +
                    [{"type": "text", "text": f"Please segment {text.strip()} in this image."}]}]
        prefix = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt = prefix
        if supervised:
            prompt = self.processor.apply_chat_template(messages + [{"role": "assistant", "content": "It is [SEG]."}],
                                                        tokenize=False, add_generation_prompt=False)
            if not prompt.startswith(prefix):
                raise ValueError("Assistant boundary is not a prefix; refuse incorrect CE masking")
        image_inputs, video_inputs = process_vision_info(messages)
        def encode(value):
            return self.processor(text=[value], images=image_inputs, videos=video_inputs, padding=True,
                                  return_tensors="pt", min_pixels=self.recipe["min_pixels"],
                                  max_pixels=self.recipe["max_pixels"]).to(self.device)
        data = encode(prompt)
        if data.input_ids.shape[1] > self.recipe["max_length"]:
            raise ValueError("Sequence exceeds max_length; no silent truncation or sample skipping")
        if supervised:
            start = encode(prefix).input_ids
            if not torch.equal(data.input_ids[:, :start.shape[1]], start):
                raise ValueError("Tokenized assistant boundary mismatch")
            labels = data.input_ids.clone()
            labels[:, :start.shape[1]] = -100
            data["labels"] = labels
            if int(((labels == self.seg_id)).sum()) != 1:
                raise ValueError("Exactly one supervised SEG token required")
        return data

    def sam_state(self, images):
        pixels = []
        for image in images:
            arr = self.base.extra_image_processor.apply_image(np.asarray(image.convert("RGB")))
            pixels.append(self.base.grounding_encoder.preprocess_image(torch.from_numpy(arr).permute(2, 0, 1).contiguous()))
        tensor = torch.stack(pixels).to(self.device, dtype=next(self.base.grounding_encoder.parameters()).dtype)
        with torch.no_grad():
            return self.base.grounding_encoder.get_sam2_embeddings(tensor)

    def project(self, hidden):
        return self.base.text_hidden_fcs(hidden.float()).to(next(self.base.grounding_encoder.parameters()).dtype)

    def forward(self, images, text, masks):
        data = self.inputs(images, text, True)
        output = self.base.model(**data, output_hidden_states=True, return_dict=True, use_cache=False)
        hidden = output.hidden_states[-1][data["labels"] == self.seg_id]
        prompt = self.project(hidden)
        pred = self.base.grounding_encoder.inject_language_embd(self.sam_state(images), [prompt for _ in images])
        if pred.shape[:2] != (len(images), 1) or not pred.requires_grad:
            raise RuntimeError("Mask count mismatch or detached segmentation path")
        gt = torch.as_tensor(masks.copy(), device=self.device, dtype=torch.float32)[:, None]
        gt = F.interpolate(gt, size=pred.shape[-2:], mode="nearest")
        bce, dice = point_losses(pred, gt, self.recipe)
        loss = output.loss + 2 * bce + .5 * dice
        return {"loss": loss, "mask_loss": 2 * bce + .5 * dice,
                "ce": output.loss.detach(), "bce": bce.detach(), "dice": dice.detach()}

    @torch.no_grad()
    def predict(self, images, text):
        data = self.inputs(images, text, False)
        sequences = self.base.model.generate(**data, do_sample=False, use_cache=True,
                                            max_new_tokens=self.recipe["max_new_tokens"])
        start = data.input_ids.shape[1]
        positions = (sequences[0, start:] == self.seg_id).nonzero().flatten()
        if positions.numel() != 1:
            # No forced SEG fallback: failed generation counts as empty prediction.
            return None, {"generated_seg_count": int(positions.numel()), "generation_failed": True}
        data["input_ids"] = sequences
        data["attention_mask"] = torch.ones_like(sequences)
        # Re-forward includes the generated SEG itself, avoiding generation hidden-state offset ambiguity.
        underlying = self.base.model.get_base_model()
        if hasattr(underlying, "rope_deltas"):
            underlying.rope_deltas = None
        if hasattr(underlying.model, "rope_deltas"):
            underlying.model.rope_deltas = None
        output = self.base.model(**data, output_hidden_states=True, return_dict=True, use_cache=False)
        prompt = self.project(output.hidden_states[-1][:, start + int(positions[0])])
        masks = self.base.grounding_encoder.language_embd_inference(
            self.sam_state(images), [prompt for _ in range(min(5, len(images)))])
        if masks.shape[:2] != (len(images), 1):
            raise ValueError("Inference mask count differs from input video")
        return masks, {"generated_seg_count": 1, "generation_failed": False}

    def trainable_state(self):
        return {n: p.detach().cpu().clone() for n, p in self.named_parameters() if p.requires_grad}

    def load_trainable(self, state):
        parameters = dict(self.named_parameters())
        expected = {n for n, p in parameters.items() if p.requires_grad}
        if set(state) != expected:
            raise ValueError("Not a matching SEG LoRA state; historical query weights rejected")
        with torch.no_grad():
            for name, value in state.items():
                parameters[name].copy_(value)
