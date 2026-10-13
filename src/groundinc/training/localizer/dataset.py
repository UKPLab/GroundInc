"""Paired localizer dataset + collator.

Each dataset item is one faithful/manipulated *pair*. Both captions share the
same video, the same deterministic candidate bank, and the same gold target
candidate; only the caption text differs. Each side is encoded as a sequence that
ends exactly at the fixed assistant prefix ``Evidence interval:`` so that the
next-token logits at the final real position define a categorical distribution
over candidate IDs.

The collator flattens pairs into a batch of ``2B`` sequences (row ``2b`` = faithful
"plus", row ``2b+1`` = manipulated "minus") plus the per-pair metadata the loss
needs (gold index, candidate token ids, validity mask, prediction positions).

Reuses ``training/final_train`` for video handling so behaviour matches the SFT
pipeline. Video decoding is done per side (both sides share the video, so this
duplicates vision compute; acceptable for the MVP, flagged for optimization).
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset


from .data_io import (  # noqa: E402
    extract_frames_decord,
    extract_frames_ffmpeg,
    load_asr_text,
    resolve_video_path,
)

from .id_tokens import DEFAULT_PREFIX, IdVocabulary, build_id_vocabulary  # noqa: E402
from .interval_bank import build_bank  # noqa: E402
from .prompt import build_localizer_prompt  # noqa: E402
from .pairing import Pair  # noqa: E402

logger = logging.getLogger(__name__)


class SequenceTooLong(Exception):
    """Raised when an encoded sample exceeds max_length; the item is skipped."""


class PairedLocalizerDataset(Dataset):
    def __init__(
        self,
        *,
        pairs: List[Pair],
        processor,
        vocab: Optional[IdVocabulary] = None,
        whisperx_dir: Optional[str] = None,
        video_dir: Optional[str] = None,
        video_mode: str = "id",
        fps: float = 0.5,
        max_pixels: int = 50176,
        max_video_frames: Optional[int] = None,
        max_length: int = 6144,
        # Bank configuration (must be identical across variants).
        bank_widths=(8.0, 16.0, 32.0, 64.0),
        bank_stride_frac: float = 0.5,
        max_candidates: int = 128,
        bank_min_len: float = 1.0,
        iou_mode: str = "hull",
        enable_video: bool = True,
        model_family: str = "qwen3vl",
    ):
        self.pairs = pairs
        self.processor = processor
        self.model_family = model_family
        self.whisperx_dir = whisperx_dir
        self.video_dir = video_dir
        self.video_mode = video_mode
        self.fps = fps
        self.max_pixels = max_pixels
        self.max_video_frames = max_video_frames
        self.max_length = max_length
        self.bank_kwargs = dict(
            widths=bank_widths,
            stride_frac=bank_stride_frac,
            max_candidates=max_candidates,
            min_len=bank_min_len,
        )
        self.iou_mode = iou_mode
        self.enable_video = enable_video

        # A single global ID vocabulary sized to the largest possible bank.
        if vocab is None:
            vocab = build_id_vocabulary(
                processor.tokenizer if hasattr(processor, "tokenizer") else processor,
                num_needed=max_candidates,
                prefix=DEFAULT_PREFIX,
            )
        self.vocab = vocab

        # Diagnostics.
        self.best_ious: List[float] = []
        self.skipped = 0

    def __len__(self) -> int:
        return len(self.pairs)

    # -- video / text construction -------------------------------------------
    def _tokenizer(self):
        return self.processor.tokenizer if hasattr(self.processor, "tokenizer") else self.processor

    def _build_video_content(self, pair: Pair, video_path: str) -> dict:
        duration = float(pair.video_duration_seconds or 0)
        nframes = max(1, int(duration * self.fps)) if duration > 0 else None
        if self.max_video_frames is not None:
            nframes = min(nframes, self.max_video_frames) if nframes else self.max_video_frames
        content = {"type": "video", "video": video_path, "max_pixels": self.max_pixels, "min_pixels": 3136}
        if nframes is not None:
            content["nframes"] = nframes
        else:
            content["fps"] = self.fps
        return content

    def _encode_side(self, pair: Pair, caption: str, bank, asr_text: str, video_path: Optional[str]) -> Dict[str, torch.Tensor]:
        if self.model_family == "gemma4":
            return self._encode_side_gemma(pair, caption, bank, asr_text, video_path)
        prompt = build_localizer_prompt(
            caption=caption,
            asr_content=asr_text,
            fps=self.fps,
            bank=bank,
            vocab=self.vocab,
        )

        if self.enable_video:
            from qwen_vl_utils import process_vision_info

            user_turn = {
                "role": "user",
                "content": [self._build_video_content(pair, video_path), {"type": "text", "text": prompt}],
            }
            try:
                image_inputs, video_inputs = process_vision_info([user_turn])
                if video_inputs is not None and video_inputs[0].shape[0] == 0:
                    raise ValueError("Decoded zero video frames")
            except Exception as exc:
                logger.warning("Video backend failed for %s (%s); ffmpeg fallback", pair.video_id, exc)
                user_turn["content"][0] = {"type": "video", "video": extract_frames_ffmpeg(video_path, self.fps, max_frames=self.max_video_frames)}
                image_inputs, video_inputs = process_vision_info([user_turn])

            prefix_text = self.processor.apply_chat_template(
                [user_turn], tokenize=False, add_generation_prompt=True
            ) + DEFAULT_PREFIX
            inputs = self.processor(
                text=[prefix_text],
                images=image_inputs,
                videos=video_inputs,
                padding=False,
                return_tensors="pt",
            )
        else:
            # Text-only path (unit tests / ablation without video).
            user_turn = {"role": "user", "content": prompt}
            prefix_text = self.processor.apply_chat_template(
                [user_turn], tokenize=False, add_generation_prompt=True
            ) + DEFAULT_PREFIX
            inputs = self.processor(text=[prefix_text], padding=False, return_tensors="pt")

        input_ids = inputs["input_ids"][0]
        if input_ids.shape[0] > self.max_length:
            # Skip (deterministically) rather than crash the DataLoader. Both
            # variants skip identically, so the controlled comparison is preserved.
            raise SequenceTooLong(
                f"Sequence for {pair.pair_id} exceeds max_length "
                f"({input_ids.shape[0]} > {self.max_length})"
            )
        attention_mask = torch.ones_like(input_ids)
        # The ID is predicted at the final real token (sequence ends at the prefix).
        pred_position = input_ids.shape[0] - 1

        side = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pred_position": torch.tensor(pred_position, dtype=torch.long),
        }
        # Qwen3-VL M-RoPE needs mm_token_type_ids alongside input_ids (required by
        # transformers >=5.x when video/image grids are present).
        if "mm_token_type_ids" in inputs:
            side["mm_token_type_ids"] = inputs["mm_token_type_ids"][0][: input_ids.shape[0]]
        for key in ("pixel_values_videos", "video_grid_thw", "pixel_values", "image_grid_thw"):
            if key in inputs:
                value = inputs[key]
                if key.endswith("_grid_thw") and value.dim() > 0 and value.shape[0] == 1:
                    value = value.squeeze(0)
                if key == "pixel_values_videos" and value.dim() == 3:
                    value = value.squeeze(0)
                side[key] = value
        return side

    def _encode_side_gemma(self, pair: Pair, caption: str, bank, asr_text: str, video_path: Optional[str]) -> Dict[str, torch.Tensor]:
        """Gemma-4 encode path. Mirrors training/final_train Gemma video handling
        (decord frames + `video_metadata`, no max_pixels — it is a no-op on Gemma),
        then appends the fixed ``Evidence interval:`` prefix so the next-token logits
        at the final position define the categorical distribution over candidate IDs
        (same contract as the Qwen path; single-token IDs verified for the Gemma
        tokenizer)."""
        prompt = build_localizer_prompt(
            caption=caption, asr_content=asr_text, fps=self.fps, bank=bank, vocab=self.vocab
        )
        tok = self._tokenizer()
        if not self.enable_video:
            user_turn = {"role": "user", "content": prompt}
            enc = self.processor.apply_chat_template(
                [user_turn], tokenize=True, return_dict=True, return_tensors="pt",
                add_generation_prompt=True,
            )
        else:
            try:
                video, video_metadata = extract_frames_decord(video_path, self.fps, max_frames=self.max_video_frames)
            except Exception as exc:
                logger.warning("Decord failed for %s (%s); ffmpeg fallback", pair.video_id, exc)
                video = extract_frames_ffmpeg(video_path, self.fps, max_frames=self.max_video_frames)
                video_metadata = {"total_num_frames": len(video), "fps": self.fps,
                                  "video_backend": "ffmpeg", "frames_indices": list(range(len(video)))}
            user_turn = {"role": "user", "content": [
                {"type": "video", "video": video, "video_metadata": video_metadata},
                {"type": "text", "text": prompt},
            ]}
            enc = self.processor.apply_chat_template(
                [user_turn], tokenize=True, return_dict=True, return_tensors="pt",
                add_generation_prompt=True,
                processor_kwargs={"video_metadata": [video_metadata], "do_sample_frames": False},
            )

        input_ids = enc["input_ids"][0]
        # Append the fixed output prefix; the ID is predicted right after it.
        prefix_ids = torch.tensor(tok(DEFAULT_PREFIX, add_special_tokens=False)["input_ids"], dtype=input_ids.dtype)
        input_ids = torch.cat([input_ids, prefix_ids])
        if input_ids.shape[0] > self.max_length:
            raise SequenceTooLong(
                f"Sequence for {pair.pair_id} exceeds max_length ({input_ids.shape[0]} > {self.max_length})"
            )
        n_prefix = int(prefix_ids.shape[0])
        attention_mask = torch.ones_like(input_ids)
        pred_position = input_ids.shape[0] - 1

        side = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pred_position": torch.tensor(pred_position, dtype=torch.long),
        }
        # Gemma-4-E4B emits mm_token_type_ids (per-token, marks image tokens); extend
        # with 0 (text) for the appended prefix so lengths line up.
        if "mm_token_type_ids" in enc:
            tt = enc["mm_token_type_ids"][0]
            side["mm_token_type_ids"] = torch.cat([tt, torch.zeros(n_prefix, dtype=tt.dtype)])
        # Copy vision tensors generically (pixel_values_videos, video_position_ids;
        # no video_grid_thw). KEEP the leading batch dim of 1 — the Gemma-4 patch
        # embedder expects 4D [batch, frames, patches, ...], so the collator cats the
        # per-side [1, frames, ...] tensors into [B, frames, ...] (NOT a 3D [B*frames,
        # ...] stack, which crashes `_position_embeddings`' permute(0,2,1,3)).
        for key, value in enc.items():
            if key in ("input_ids", "attention_mask", "mm_token_type_ids"):
                continue
            if torch.is_tensor(value):
                side[key] = value
        return side

    def __getitem__(self, index: int) -> Dict:
        # A sample can overflow max_length (long ASR + candidate list). Skip
        # deterministically to the next pair instead of crashing the worker; the
        # skip is a pure function of the data, so both variants drop the same pairs.
        n = len(self.pairs)
        last_exc = None
        for offset in range(min(64, n)):
            idx = (index + offset) % n
            try:
                return self._build_item(idx)
            except SequenceTooLong as exc:
                self.skipped += 1
                last_exc = exc
                if self.skipped <= 20 or self.skipped % 100 == 0:
                    logger.warning("Skipping oversized pair (%d skipped total): %s", self.skipped, exc)
            except Exception as exc:  # pragma: no cover - defensive
                last_exc = exc
                logger.error("Failed pair at index %d: %s", idx, exc)
        raise RuntimeError(f"Exceeded consecutive skip/failure budget; last: {last_exc}")

    def _build_item(self, index: int) -> Dict:
        pair = self.pairs[index]
        bank = build_bank(pair.video_duration_seconds, **self.bank_kwargs)
        gold_index, best_iou = bank.best_candidate(pair.gold_span, iou_mode=self.iou_mode)
        if gold_index < 0:
            raise ValueError(f"Pair {pair.pair_id} has no valid gold target")

        asr_text = load_asr_text(self.whisperx_dir, pair.video_id) if self.whisperx_dir else ""
        video_path = None
        if self.enable_video:
            record_like = {"id": pair.video_id, "video_path": pair.video_path}
            video_path = resolve_video_path(record_like, self.video_mode, self.video_dir)

        plus = self._encode_side(pair, pair.faithful_caption, bank, asr_text, video_path)
        minus = self._encode_side(pair, pair.manipulated_caption, bank, asr_text, video_path)

        self.best_ious.append(best_iou)  # record only successfully built items
        k = len(bank)
        cand_token_ids = torch.tensor(self.vocab.token_ids[:k], dtype=torch.long)
        return {
            "plus": plus,
            "minus": minus,
            "gold_index": torch.tensor(gold_index, dtype=torch.long),
            "cand_token_ids": cand_token_ids,
            "num_candidates": k,
            "meta": {
                "pair_id": pair.pair_id,
                "video_id": pair.video_id,
                "modality": pair.caption_modality,
                "manipulated_type": pair.manipulated_type,
                "best_iou": best_iou,
                "bank_hash": bank.version_hash,
                "gold_span": pair.gold_span,
            },
        }


@dataclass
class PairedLocalizerCollator:
    """Flatten pairs into a ``2B``-row model batch plus per-pair loss metadata."""

    pad_token_id: int

    def __call__(self, instances: List[Dict]) -> Dict:
        sides = []  # flattened: [plus_0, minus_0, plus_1, minus_1, ...]
        for inst in instances:
            sides.append(inst["plus"])
            sides.append(inst["minus"])

        max_len = max(s["input_ids"].shape[0] for s in sides)
        n = len(sides)
        batch_input_ids = torch.full((n, max_len), self.pad_token_id, dtype=torch.long)
        batch_attention = torch.zeros((n, max_len), dtype=torch.long)
        pred_positions = torch.zeros(n, dtype=torch.long)
        has_mm_types = all("mm_token_type_ids" in s for s in sides)
        batch_mm_types = torch.zeros((n, max_len), dtype=torch.long) if has_mm_types else None

        for i, side in enumerate(sides):
            length = side["input_ids"].shape[0]
            batch_input_ids[i, :length] = side["input_ids"]
            batch_attention[i, :length] = side["attention_mask"]
            pred_positions[i] = side["pred_position"]
            if has_mm_types:
                batch_mm_types[i, :length] = side["mm_token_type_ids"][:length]

        batch = {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention,
            "pred_positions": pred_positions,
        }
        if has_mm_types:
            batch["mm_token_type_ids"] = batch_mm_types

        # Merge vision tensors in row order (concatenated grids, per Qwen contract).
        self._merge(sides, batch, "pixel_values_videos", cat=True)
        self._merge(sides, batch, "video_grid_thw", cat=False)
        self._merge(sides, batch, "video_position_ids", cat=True)  # Gemma-4-E4B
        self._merge(sides, batch, "pixel_values", cat=True)
        self._merge(sides, batch, "image_grid_thw", cat=False)

        # Per-pair loss metadata.
        b = len(instances)
        gold_index = torch.stack([inst["gold_index"] for inst in instances])
        kmax = max(inst["num_candidates"] for inst in instances)
        cand_token_ids = torch.zeros((b, kmax), dtype=torch.long)
        cand_mask = torch.zeros((b, kmax), dtype=torch.bool)
        for j, inst in enumerate(instances):
            k = inst["num_candidates"]
            cand_token_ids[j, :k] = inst["cand_token_ids"]
            cand_mask[j, :k] = True

        batch["gold_index"] = gold_index
        batch["cand_token_ids"] = cand_token_ids
        batch["cand_mask"] = cand_mask
        batch["meta"] = [inst["meta"] for inst in instances]
        return batch

    def _merge(self, sides, batch, key: str, cat: bool):
        if key not in sides[0]:
            return
        if not all(key in s for s in sides):
            return
        values = [s[key] for s in sides]
        batch[key] = torch.cat(values, dim=0) if cat else torch.stack(values, dim=0)
