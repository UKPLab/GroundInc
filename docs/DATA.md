# Data

GroundInc ships annotations only. Videos are not included. The `id` field is a
YouTube identifier, and you need to obtain the media yourself.

| File | Role |
| --- | --- |
| `data/groundinc_news.jsonl` | GroundInc test set (News & Politics) |
| `data/groundinc_other.jsonl` | Cross-domain set (Sports, Science & Technology, Education, Nonprofits & Activism) |

Both are JSON Lines, one record per annotated video segment.

## Record fields

| Field | Type | Description |
| --- | --- | --- |
| `id` | string | Video identifier. Resolves `{video_dir}/{id}.mp4` and `{asr_dir}/{id}_asr.json`. |
| `select_seg` | int | One-indexed annotated segment. |
| `seg_time` | list | Each segment's `[start, end]` as `H:MM:SS.sss`. `seg_time[select_seg-1]` is the gold span. |
| `video_captions` | list | Visual caption per segment. |
| `audio_captions` | list | Audio caption per segment. |
| `duration` | int | Video length in seconds. |
| `category` | string | Video domain, e.g. `"News & Politics"`. |
| `groundinc_annotations` | object | Faithful captions, inconsistent captions, labels, and refutations, described below. |

## groundinc_annotations

| Field | Description |
| --- | --- |
| `true_visual`, `true_audio`, `true_combined` | Faithful captions of the segment. |
| `visual_only_manipulations` | False visual captions. |
| `audio_only_manipulations` | False audio captions. |
| `combined_manipulations` | Captions with one modality manipulated. |

Each manipulation has `method`, `entity_type`, `false_caption`, and `refutation`.

## What you need to supply

- Video: `{video_dir}/{id}.mp4`.
- ASR: `{asr_dir}/{id}_asr.json` with a `"prompt"` string, or `"segments"` with
  `start`/`end`/`text`. You can also put the transcript inline as `asr_text` on
  each record.
- Audio: `{audio_dir}/{id}.wav` for the video and audio runner.
