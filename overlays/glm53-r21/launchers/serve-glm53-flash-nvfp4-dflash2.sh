#!/usr/bin/env bash
set -euo pipefail

# DFlash K7 verifies eight target rows per request, while MTP3 verifies four.
# These capture sizes keep C1/C8/C12/C24 target batches on exact full CUDA
# graphs. Other request counts use vLLM's next captured padded size.
readonly base_launcher=/usr/local/libexec/serve-glm53-flash-nvfp4-dflash2.sh
capture_sizes=${CUDAGRAPH_CAPTURE_SIZES:-'1 2 4 8 16 32 40 48 64 96 128 192 256'}

readonly configured_max_cudagraph_capture_size=${MAX_CUDAGRAPH_CAPTURE_SIZE:-256}

fail() {
  printf '%s\n' "$1" >&2
  exit 2
}

normalize_nonnegative_integer() {
  local name=$1
  local value=$2
  local output_name=$3
  local normalized
  [[ ${value} =~ ^[0-9]+$ ]] ||
    fail "${name} must be a non-negative integer; got ${value}"
  [[ ${value} =~ ^0*([0-9]{1,19})$ ]] ||
    fail "${name} exceeds Bash signed integer range; got ${value}"
  normalized=${BASH_REMATCH[1]}
  if ((${#normalized} == 19)) &&
    [[ ${normalized} > 9223372036854775807 ]]; then
    fail "${name} exceeds Bash signed integer range; got ${value}"
  fi
  printf -v "${output_name}" '%s' "${normalized}"
}
normalize_nonnegative_integer MAX_CUDAGRAPH_CAPTURE_SIZE \
  "${configured_max_cudagraph_capture_size}" max_cudagraph_capture_size
readonly max_cudagraph_capture_size
export MAX_CUDAGRAPH_CAPTURE_SIZE=${max_cudagraph_capture_size}


require_open_unit_interval() {
  local name=$1
  local value=$2
  [[ ${value} =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$ ]] ||
    fail "${name} must be numeric; got ${value}"
  awk -v value="${value}" 'BEGIN { exit !(value > 0.0 && value < 1.0) }' ||
    fail "${name} must be greater than zero and less than one; got ${value}"
}

has_cli_option() {
  local expected=$1
  shift
  local argument
  for argument in "$@"; do
    if [[ ${argument} == "${expected}" || ${argument} == "${expected}="* ]]; then
      return 0
    fi
  done
  return 1
}

runtime_args=()

# Compute-share fairness charges measured model-execution time to prefill and
# decode. CLI fairness options are one complete, authoritative override; when
# either is present, the environment-owned pair is not appended.
fairness_engine=${FAIRNESS_ENGINE:-none}
prefill_compute_share=${PREFILL_COMPUTE_SHARE:-}
case "${fairness_engine}" in
  none) ;;
  compute_share)
    [[ -n ${prefill_compute_share} ]] ||
      fail 'PREFILL_COMPUTE_SHARE is required for FAIRNESS_ENGINE=compute_share'
    require_open_unit_interval PREFILL_COMPUTE_SHARE "${prefill_compute_share}"
    ;;
  micro_slicing) ;;
  *)
    fail "FAIRNESS_ENGINE must be none, compute_share, or micro_slicing; got ${fairness_engine}"
    ;;
esac

if [[ ${fairness_engine} != none && ${PREFILL_SCHEDULE_INTERVAL:-1} != 1 ]]; then
  fail 'PREFILL_SCHEDULE_INTERVAL must be 1 when FAIRNESS_ENGINE is enabled'
fi

if ! has_cli_option --fairness-engine "$@" &&
  ! has_cli_option --prefill-compute-share "$@"; then
  case "${fairness_engine}" in
    compute_share)
      runtime_args+=(
        --fairness-engine compute_share
        --prefill-compute-share "${prefill_compute_share}"
      )
      ;;
    micro_slicing)
      runtime_args+=(--fairness-engine micro_slicing)
      ;;
  esac
fi

# A caller-supplied CLI value is authoritative and must not be duplicated.
if [[ ${capture_sizes} != none ]] &&
  ! has_cli_option --cudagraph-capture-sizes "$@"; then
  read -r -a capture_size_args <<<"${capture_sizes}"
  if ((${#capture_size_args[@]} == 0)); then
    fail 'CUDAGRAPH_CAPTURE_SIZES must contain positive integers or "none"'
  fi

  previous=0
  for size in "${capture_size_args[@]}"; do
    if [[ ! ${size} =~ ^[1-9][0-9]*$ ]]; then
      fail "Invalid CUDA graph capture size: ${size}"
    fi
    normalize_nonnegative_integer "CUDA graph capture size" "${size}" size
    if ((size <= previous)); then
      fail "CUDAGRAPH_CAPTURE_SIZES must be strictly increasing; got ${size} after ${previous}"
    fi
    if ((size > max_cudagraph_capture_size)); then
      fail "CUDA graph capture size ${size} exceeds MAX_CUDAGRAPH_CAPTURE_SIZE=${configured_max_cudagraph_capture_size}"
    fi
    previous=${size}
  done
  runtime_args+=(--cudagraph-capture-sizes "${capture_size_args[@]}")
fi

# Mixed decode/prefill controls are disabled by default. Nonzero values opt in
# to bounded prefill quanta, partial-prefill admission, and decode-burst QoS.
max_num_prefill_tokens_per_step=${MAX_NUM_PREFILL_TOKENS_PER_STEP:-0}
max_num_partial_prefills=${MAX_NUM_PARTIAL_PREFILLS:-0}
decode_prefill_min_decode_steps=${DECODE_PREFILL_MIN_DECODE_STEPS:-0}
decode_prefill_max_wait_ms=${DECODE_PREFILL_MAX_WAIT_MS:-0}

normalize_nonnegative_integer MAX_NUM_PREFILL_TOKENS_PER_STEP \
  "${max_num_prefill_tokens_per_step}" max_num_prefill_tokens_per_step
normalize_nonnegative_integer MAX_NUM_PARTIAL_PREFILLS \
  "${max_num_partial_prefills}" max_num_partial_prefills
normalize_nonnegative_integer DECODE_PREFILL_MIN_DECODE_STEPS \
  "${decode_prefill_min_decode_steps}" decode_prefill_min_decode_steps
normalize_nonnegative_integer DECODE_PREFILL_MAX_WAIT_MS \
  "${decode_prefill_max_wait_ms}" decode_prefill_max_wait_ms

if ((max_num_prefill_tokens_per_step > 0)) &&
  ! has_cli_option --max-num-prefill-tokens-per-step "$@"; then
  runtime_args+=(
    --max-num-prefill-tokens-per-step "${max_num_prefill_tokens_per_step}"
  )
fi
if ((max_num_partial_prefills > 0)) &&
  ! has_cli_option --max-num-partial-prefills "$@"; then
  runtime_args+=(--max-num-partial-prefills "${max_num_partial_prefills}")
fi
if ((decode_prefill_min_decode_steps > 0)) &&
  ! has_cli_option --decode-prefill-min-decode-steps "$@"; then
  runtime_args+=(
    --decode-prefill-min-decode-steps "${decode_prefill_min_decode_steps}"
  )
fi
if ((decode_prefill_max_wait_ms > 0)) &&
  ! has_cli_option --decode-prefill-max-wait-ms "$@"; then
  runtime_args+=(--decode-prefill-max-wait-ms "${decode_prefill_max_wait_ms}")
fi

exec "${base_launcher}" "$@" "${runtime_args[@]}"
