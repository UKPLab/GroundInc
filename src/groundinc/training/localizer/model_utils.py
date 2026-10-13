from __future__ import annotations

import logging

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


logger = logging.getLogger(__name__)


def resolve_torch_dtype(dtype_name: str):
    normalized = dtype_name.lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    if normalized == "auto":
        return "auto"
    raise ValueError("torch_dtype must be one of: auto, bfloat16, float16, float32")


def load_model_and_processor(model_args):
    model_kwargs = {
        "cache_dir": model_args.cache_dir,
        "dtype": resolve_torch_dtype(model_args.torch_dtype),
        "device_map": None,
    }
    if model_args.attn_implementation:
        model_kwargs["attn_implementation"] = model_args.attn_implementation

    logger.info("Loading Qwen3-VL model: %s", model_args.model_name_or_path)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        **model_kwargs,
    )
    model.config.use_cache = False
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    return model, processor


def maybe_freeze_vision_tower(model, enabled: bool):
    if not enabled:
        return
    candidates = [
        getattr(getattr(model, "model", None), "visual", None),
        getattr(getattr(model, "model", None), "vision_tower", None),
        getattr(getattr(model, "model", None), "vision_encoder", None),
        getattr(model, "vision_tower", None),
        getattr(model, "vision_encoder", None),
    ]
    vision_tower = next((candidate for candidate in candidates if candidate is not None), None)
    if vision_tower is None:
        logger.warning("Could not find a vision tower; vision parameters remain trainable")
        return
    for param in vision_tower.parameters():
        param.requires_grad = False
    logger.info("Vision tower frozen")


def maybe_apply_lora(model, model_args):
    if not model_args.lora_enable:
        logger.info("LoRA disabled")
        return model

    target_modules = [
        module.strip()
        for module in model_args.lora_target_modules.split(",")
        if module.strip()
    ]
    lora_config = LoraConfig(
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=model_args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        use_rslora=getattr(model_args, "lora_use_rslora", True),
    )
    model = get_peft_model(model, lora_config)
    logger.info(
        "LoRA config: r=%d alpha=%d rslora=%s (scale=%.2f)",
        model_args.lora_r, model_args.lora_alpha,
        getattr(model_args, "lora_use_rslora", True),
        (model_args.lora_alpha / (model_args.lora_r ** 0.5))
        if getattr(model_args, "lora_use_rslora", True)
        else (model_args.lora_alpha / model_args.lora_r),
    )
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    logger.info("LoRA enabled; trainable parameters: %.1fM", trainable / 1e6)
    return model
