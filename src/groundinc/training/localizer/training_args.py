from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from transformers import HfArgumentParser, TrainingArguments

from groundinc.training.prompts import prompt_names


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Hugging Face model id or local checkpoint path."}
    )
    model_family: str = field(
        default="auto",
        metadata={"help": "auto, qwen3vl, or gemma4."},
    )
    cache_dir: Optional[str] = None
    torch_dtype: str = "bfloat16"
    attn_implementation: Optional[str] = "flash_attention_2"
    freeze_vision_tower: bool = False
    lora_enable: bool = True
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    # rsLoRA rescales the adapter by alpha/sqrt(r) instead of alpha/r. With
    # r=32, alpha=64 that is ~11.3x vs 2.0x — a much more aggressive adapter.
    lora_use_rslora: bool = True
    lora_target_modules: str = "q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj"
    wandb_entity: str = ""
    wandb_project: str = "groundinc-localizer"


@dataclass
class DataArguments:
    train_data_path: str = field(metadata={"help": "Path to training JSONL file."})
    whisperx_dir: str = field(metadata={"help": "Directory containing *_asr.json files."})
    eval_data_path: Optional[str] = None
    video_mode: str = "id"
    video_dir: Optional[str] = None
    broken_videos_txt: Optional[str] = None
    max_length: int = 8192
    fps: float = 0.5
    max_pixels: int = 50176
    max_video_frames: Optional[int] = None
    debug_dataset: bool = False
    prompt_name: str = field(
        default="veracity",
        metadata={"help": f"Named prompt. Available: {prompt_names()}"},
    )
    task_mode: str = field(
        default="auto",
        metadata={
            "help": "auto, veracity_only, no_evidence, full, inconsistency, or inconsistency_no_grounding."
        },
    )
    prompt_template: Optional[str] = field(
        default=None,
        metadata={"help": "Inline custom prompt with {fps}, {asr_content}, {caption}."},
    )
    prompt_template_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a custom prompt text file."},
    )


def parse_args():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith((".yaml", ".yml")):
        config_path = Path(sys.argv[1]).resolve()
        model_args, data_args, train_args = parser.parse_yaml_file(str(config_path))
        if data_args.prompt_template_path:
            prompt_path = Path(data_args.prompt_template_path).expanduser()
            if not prompt_path.is_absolute():
                data_args.prompt_template_path = str(config_path.parent / prompt_path)
        return model_args, data_args, train_args
    return parser.parse_args_into_dataclasses()


def normalize_report_to(report_to):
    if report_to is None:
        return []
    if isinstance(report_to, str):
        return [report_to]
    return list(report_to)
