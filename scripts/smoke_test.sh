#!/usr/bin/env bash
set -euo pipefail

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PYTHONPATH:-}:src"
export PYTHONPYCACHEPREFIX="${TMPDIR:-/tmp}/groundinc_pycache_$$"
trap 'rm -rf "$PYTHONPYCACHEPREFIX"' EXIT

python -m compileall -q src
python -m groundinc.evaluation.vllm_asr --list_prompt_styles >/dev/null
python -m groundinc.evaluation.vllm_audio --list_prompt_styles >/dev/null

# predicted_snippet must be an accepted video mode, and its grounding pass must
# resolve to a registered prompt style.
python -m groundinc.evaluation.vllm_asr --help | grep -q "predicted_snippet"
python -m groundinc.evaluation.vllm_asr --list_prompt_styles | grep -q "^grounding:"

# Clock-formatted spans must parse to seconds: predicted_snippet cuts video with
# this result, so a regression here changes the evidence, not just the score.
python - <<'PY'
from groundinc.data.prompts_asr import parse_timestamps
assert parse_timestamps("happens in 00:24 - 01:07") == [24.0, 67.0]
assert parse_timestamps("47.43 - 52.84") == [47.43, 52.84]
assert parse_timestamps("") == []
PY

# Training preparation imports and exposes every prompt/target pairing.
python - <<'PY'
from groundinc.training import PROMPTS, PROMPT_NAME_TO_INFERENCE_STYLE, resolve_prompt
from groundinc.training.targets import build_ground_truth_answer
assert set(PROMPTS) == set(PROMPT_NAME_TO_INFERENCE_STYLE), "prompt/style mapping incomplete"
task = {"type": "rephrased_true", "manipulated_modality": "none",
        "manipulation_method_gt": "No Manipulation", "ground_truth_span": [1.0, 2.0]}
for name in PROMPTS:
    _, mode = resolve_prompt(prompt_name=name, task_mode="auto",
                             prompt_template=None, prompt_template_path=None)
    assert build_ground_truth_answer(task, visual_evidence="v", audio_evidence="a",
                                     task_mode=mode)
PY
python -m groundinc.training.prepare_sft --help >/dev/null

# The task builder must exist exactly once so training and inference cannot drift.
test "$(grep -rc 'def build_caption_tasks' src --include='*.py' | grep -v ':0$' | wc -l)" -eq 1

echo "Smoke test passed."
