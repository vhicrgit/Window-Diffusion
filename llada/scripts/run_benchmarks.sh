#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash llada/scripts/run_benchmarks.sh --task gsm8k
  bash llada/scripts/run_benchmarks.sh --task all

Options:
  --task gsm8k|math|humaneval|mbpp|all

Environment overrides:
  MODEL_PATH             Local or HuggingFace model path. Default: GSAI-ML/LLaDA-8B-Base
  OUTPUT_ROOT            Result directory. Default: <repo>/runs/llada
  CUDA_VISIBLE_DEVICES   GPU selection passed through to accelerate.
  DEVICE                 lm-eval device argument. Default: cuda
  BATCH_SIZE             Must remain 1 for the current LLaDA implementation. Default: 1
  MAIN_PROCESS_PORT      Accelerate main process port. Default: 29511
  LIMIT                  lm-eval --limit value for smoke tests, e.g. LIMIT=1
  WINDOW_TOKENS          External window length. Default: 64
  ACTIVE_TOKENS          Internal window length. Default: 16
  REFRESH_CYCLE          KV refresh cycle / phase length. Default: 32
USAGE
}

TASK=""

while [[ $# -gt 0 ]]; do
  case "$1" in
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

case "${TASK}" in
  gsm8k|math|humaneval|mbpp|all) ;;
  *)
    echo "--task must be one of: gsm8k, math, humaneval, mbpp, all." >&2
    usage >&2
    exit 1
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLADA_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${LLADA_DIR}/.." && pwd)"

export HF_ALLOW_CODE_EVAL="${HF_ALLOW_CODE_EVAL:-1}"
export HF_DATASETS_TRUST_REMOTE_CODE="${HF_DATASETS_TRUST_REMOTE_CODE:-true}"

MODEL_PATH="${MODEL_PATH:-GSAI-ML/LLaDA-8B-Base}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/runs/llada}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29511}"
WINDOW_TOKENS="${WINDOW_TOKENS:-64}"
ACTIVE_TOKENS="${ACTIVE_TOKENS:-16}"
REFRESH_CYCLE="${REFRESH_CYCLE:-32}"

if [[ "${BATCH_SIZE}" != "1" ]]; then
  echo "The current LLaDA Window-Diffusion implementation only supports BATCH_SIZE=1." >&2
  exit 1
fi

run_eval() {
  local task_key="$1"
  local lm_task="$2"
  local fewshot="$3"
  local gen_len="$4"

  local output_path="${OUTPUT_ROOT}/${task_key}"
  local model_args
  model_args="pretrained=${MODEL_PATH},gen_length=${gen_len},steps=${gen_len},window_tokens=${WINDOW_TOKENS},active_tokens=${ACTIVE_TOKENS},refresh_cycle=${REFRESH_CYCLE},temperature=0.0,cfg_scale=0.0,remasking=low_confidence,attn_implementation=eager"

  local cmd=(
    accelerate launch --main_process_port "${MAIN_PROCESS_PORT}"
    "${LLADA_DIR}/eval.py"
    --model llada_window_diffusion
    --model_args "${model_args}"
    --tasks "${lm_task}"
    --num_fewshot "${fewshot}"
    --batch_size "${BATCH_SIZE}"
    --device "${DEVICE}"
    --log_samples
    --output_path "${output_path}"
  )

  if [[ -n "${LIMIT:-}" ]]; then
    cmd+=(--limit "${LIMIT}")
  fi

  if [[ "${task_key}" == "humaneval" || "${task_key}" == "mbpp" ]]; then
    cmd+=(--confirm_run_unsafe_code)
  fi

  echo "Running ${task_key} (${lm_task})"
  echo "Output: ${output_path}"
  "${cmd[@]}"
}

run_task() {
  local task_key="$1"
  case "${task_key}" in
    gsm8k)
      run_eval gsm8k gsm8k 4 256
      ;;
    math)
      run_eval math minerva_math 4 256
      ;;
    humaneval)
      run_eval humaneval humaneval 0 512
      ;;
    mbpp)
      run_eval mbpp mbpp 3 512
      ;;
  esac
}

mkdir -p "${OUTPUT_ROOT}"

if [[ "${TASK}" == "all" ]]; then
  for task_key in gsm8k math humaneval mbpp; do
    run_task "${task_key}"
  done
else
  run_task "${TASK}"
fi
