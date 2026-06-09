#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash dream/scripts/run_benchmarks.sh --model-type base --task gsm8k
  bash dream/scripts/run_benchmarks.sh --model-type instruct --task all

Options:
  --model-type base|instruct   Dream checkpoint family to evaluate.
  --task gsm8k|math|humaneval|mbpp|all

Environment overrides:
  MODEL_PATH             Local or HuggingFace model path.
  OUTPUT_ROOT            Result directory. Default: <repo>/runs/dream
  CUDA_VISIBLE_DEVICES   GPU selection passed through to accelerate.
  DEVICE                 lm-eval device argument. Default: cuda
  MAIN_PROCESS_PORT      Accelerate main process port. Default: 29501
  LIMIT                  lm-eval --limit value for smoke tests, e.g. LIMIT=1
USAGE
}

MODEL_TYPE=""
TASK=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-type)
      MODEL_TYPE="${2:-}"
      shift 2
      ;;
    --task)
      TASK="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ "${MODEL_TYPE}" != "base" && "${MODEL_TYPE}" != "instruct" ]]; then
  echo "--model-type must be either 'base' or 'instruct'." >&2
  usage >&2
  exit 1
fi

case "${TASK}" in
  gsm8k|math|humaneval|mbpp|all) ;;
  *)
    echo "--task must be one of: gsm8k, math, humaneval, mbpp, all." >&2
    usage >&2
    exit 1
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DREAM_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${DREAM_DIR}/.." && pwd)"

export HF_ALLOW_CODE_EVAL="${HF_ALLOW_CODE_EVAL:-1}"
export HF_DATASETS_TRUST_REMOTE_CODE="${HF_DATASETS_TRUST_REMOTE_CODE:-true}"

MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29501}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/runs/dream}"
DEVICE="${DEVICE:-cuda}"

if [[ -z "${MODEL_PATH:-}" ]]; then
  if [[ "${MODEL_TYPE}" == "base" ]]; then
    MODEL_PATH="Dream-org/Dream-v0-Base-7B"
  else
    MODEL_PATH="Dream-org/Dream-v0-Instruct-7B"
  fi
fi

run_eval() {
  local task_key="$1"
  local lm_task="$2"
  local fewshot="$3"
  local gen_len="$4"
  local i_win="$5"
  local o_win="$6"
  local refresh_cycle="$7"
  local slide_window="$8"
  local early_stop="$9"
  local temperature="${10}"
  local top_p="${11}"

  local output_path="${OUTPUT_ROOT}/${MODEL_TYPE}/${task_key}"
  local model_args
  model_args="pretrained=${MODEL_PATH},max_new_tokens=${gen_len},diffusion_steps=${gen_len},add_bos_token=true,alg=entropy,o_win_size=${o_win},i_win_size=${i_win},refresh_cycle=${refresh_cycle},slide_window=${slide_window},early_stop=${early_stop},temperature=${temperature},top_p=${top_p},attn_implementation=eager"

  local cmd=(
    accelerate launch --main_process_port "${MAIN_PROCESS_PORT}"
    "${DREAM_DIR}/eval.py"
    --model dream_window_diffusion
    --model_args "${model_args}"
    --tasks "${lm_task}"
    --num_fewshot "${fewshot}"
    --batch_size 1
    --device "${DEVICE}"
    --log_samples
    --output_path "${output_path}"
    --confirm_run_unsafe_code
  )

  if [[ -n "${LIMIT:-}" ]]; then
    cmd+=(--limit "${LIMIT}")
  fi

  if [[ "${MODEL_TYPE}" == "instruct" ]]; then
    cmd+=(--apply_chat_template)
  fi

  echo "Running ${MODEL_TYPE} ${task_key} (${lm_task})"
  echo "Output: ${output_path}"
  "${cmd[@]}"
}

run_task() {
  local task_key="$1"
  if [[ "${MODEL_TYPE}" == "base" ]]; then
    case "${task_key}" in
      gsm8k)
        run_eval gsm8k gsm8k_cot 8 256 16 128 32 true false 0.0 0.99
        ;;
      math)
        run_eval math minerva_math 4 512 16 128 32 true false 0.0 0.99
        ;;
      humaneval)
        run_eval humaneval humaneval 0 512 16 128 40 true false 0.0 0.95
        ;;
      mbpp)
        run_eval mbpp mbpp 3 512 16 128 32 true false 0.0 0.99
        ;;
    esac
  else
    case "${task_key}" in
      gsm8k)
        run_eval gsm8k gsm8k_cot 0 256 16 128 32 true true 0.1 0.9
        ;;
      math)
        run_eval math minerva_math 0 512 16 128 32 true false 0.0 0.99
        ;;
      humaneval)
        run_eval humaneval humaneval 0 768 16 128 32 true true 0.1 0.9
        ;;
      mbpp)
        run_eval mbpp mbpp_instruct 0 1024 16 128 32 true false 0.1 0.9
        ;;
    esac
  fi
}

mkdir -p "${OUTPUT_ROOT}/${MODEL_TYPE}"

if [[ "${TASK}" == "all" ]]; then
  for task_key in gsm8k math humaneval mbpp; do
    run_task "${task_key}"
  done
else
  run_task "${TASK}"
fi
