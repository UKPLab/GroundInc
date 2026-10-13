# Data

GroundInc ships annotations only. Videos are **not** redistributed: `id` is an
11-character YouTube identifier, and you must obtain the media yourself.

| File | Rows | Caption tasks | Role |
| --- | --- | --- | --- |
| `data/groundinc_news.jsonl` | 466 | 15,496 | Main GroundInc test set (News & Politics) |
| `data/groundinc_other.jsonl` | 406 | 13,418 | Cross-domain set (Sports, Science & Technology, Education, Nonprofits & Activism) |

Both are JSON Lines, one record per annotated video segment. `build_caption_tasks`
expands a record into ~33 evaluation items.

## Record fields

### Required

| Field | Type | Description |
| --- | --- | --- |
| `id` | string | Video identifier. Resolves `{video_dir}/{id}.mp4` and `{asr_dir}/{id}_asr.json`. |
| `select_seg` | int | One-indexed annotated segment. |
| `seg_time` | list | **Every** segment's `[start, end]` as `H:MM:SS.sss`. `seg_time[select_seg-1]` is the gold span. |
| `video_captions` | list of strings | Visual caption per segment, aligned with `seg_time`. |
| `audio_captions` | list of strings | Audio caption per segment, aligned with `seg_time`. |
| `duration` | int | Video length in seconds. Used for the SFT frame budget and for clamping predicted spans. |
| `forensic_mentalization` | object | Generated annotations; see below. |

### Metadata

| Field | Description |
| --- | --- |
| `category` | Video domain, e.g. `"News & Politics"`. Distinct from the `category` on a *result*, which is the modality (`visual`/`audio`/`combined`). |

### Optional, accepted but not shipped

| Field | Used by |
| --- | --- |
| `video_path` | `--video_mode clip` and `auto`. |
| `relative_seg_gt` | Clip-relative gold span, required with `video_path`. |
| `audio_path` | The video+audio evaluator. |
| `asr_text` / `asr` / `transcript` | Inline transcript. Set this to run without an `--asr_dir` at all. |

## `forensic_mentalization`

| Field | Description |
| --- | --- |
| `true_visual`, `true_audio`, `true_combined` | Faithful rephrasings of the annotated segment's captions. |
| `visual_only_manipulations` | False visual captions. |
| `audio_only_manipulations` | False audio captions. |
| `combined_manipulations` | Cross-modal items: one modality manipulated, the other left true. Carries `type` = `manipulated_visual_true_audio` or `true_visual_manipulated_audio`. |
| `visual_planning`, `audio_planning` | Raw stage-1 output: the rephrasing plus the manipulation recipes. Not read by any code; retained so each false caption's derivation is inspectable. |

Every manipulation carries `method`, `entity_type` (`internal`/`external`/`n/a`),
`false_caption`, and `refutation`.

Three taxonomy labels — Quantifier and Scope Distortions, Epistemic Inflation,
Entity Disambiguation / Homonyms — are dropped at load time by
`EXCLUDED_LABELS` and never appear as tasks.

## Rebuilding the release files

`scripts/prepare_release_data.py` produces these files from the working
dataset, dropping the twelve multi-turn QA fields (`v2a_question` … `t2a_answer`)
and the `index` / `segment_num` bookkeeping fields, none of which any code reads.

```bash
python scripts/prepare_release_data.py \
  --input news_data_mt_final.jsonl --output data/groundinc_news.jsonl
```

## What you must supply

Neither media nor transcripts ship, and no extraction scripts are included.

- **Video** — `{video_dir}/{id}.mp4`, fetched by YouTube id.
- **ASR** — `{asr_dir}/{id}_asr.json` holding a `"prompt"` string (the paper used
  WhisperX) or `"segments"` with `start`/`end`/`text`. Segment timings are
  required for `--video_mode predicted_snippet` with `--asr_align span`.
  Alternatively, put the transcript inline as `asr_text` on each record.
- **Audio** — `{audio_dir}/{id}.wav|mp3` for the video+audio evaluator.

Generation reads the video but not the transcripts.

---

## Compound subset (multi-span / multi-label)

Compound records carry manipulations under `data["multi_inconsistency"]` in three
subsets with **list-valued** gold spans and methods (a different shape from the
single-span records, so they use a dedicated task builder,
`groundinc.data.compound.build_multi_caption_tasks`):

| Subset key | Structure |
| --- | --- |
| `multi_span` | ≥2 evidence regions, 1 manipulation label |
| `multi_label` | 1 region, ≥2 labels |
| `multi_span_label` | ≥2 regions, ≥2 labels |

Each item pairs a controlled true/false caption with gold `ground_truth_spans`
(list) and `methods` (list). Build tasks per subset:

```python
from groundinc.data.compound import build_multi_caption_tasks, SUBSETS
for subset in SUBSETS:
    tasks = build_multi_caption_tasks(record, subset)
```

The compound annotations ship alongside the single-span files; verification and
multi-span grounding over them reuse the same runners, with a multi-span prompt
style and parser registered in `groundinc.data.prompts*`.
