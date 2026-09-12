#!/usr/bin/env bash
set -euo pipefail

readonly cache_launcher=/usr/local/libexec/serve-glm53-flash-lmcache-cache-complete.sh

if (($# == 1)) && [[ $1 == --help || $1 == -h ]]; then
  exec /usr/local/bin/serve-glm53-flash-nvfp4-dflash2.sh "$@"
fi

# Explicit cache policy or geometry also selects the cache-mode contract. With
# no cache-related setting, preserve the image's ordinary serving command.
explicit_checkpoint_policy=0
for arg in "$@"; do
  case "${arg}" in
    --recurrent-checkpoint-policy | --recurrent-checkpoint-policy=*)
      explicit_checkpoint_policy=1
      ;;
  esac
done
if [[ -z ${CACHE_MODE+x} && -z ${LMCACHE_ENABLED+x} && \
  -z ${GLM53_TARGET_BLOCK_SIZE+x} && -z ${GLM53_MAMBA_BLOCK_SIZE+x} && \
  ${explicit_checkpoint_policy} == 0 && \
  ${CACHE_CONFIG_DRY_RUN:-0} != 1 ]]; then
  exec "${cache_launcher}" "$@"
fi

fail() {
  printf '%s\n' "$1" >&2
  exit 2
}

require_positive_integer() {
  local name=$1
  local value=$2
  [[ ${value} =~ ^[1-9][0-9]*$ ]] ||
    fail "${name} must be a positive integer; got ${value}"
}

require_nonnegative_integer() {
  local name=$1
  local value=$2
  [[ ${value} =~ ^[0-9]+$ ]] ||
    fail "${name} must be a non-negative integer; got ${value}"
}

sanitize_namespace_component() {
  local value=$1
  value=${value//[^A-Za-z0-9._-]/_}
  [[ -n ${value} ]] || value=unset
  printf '%s' "${value}"
}

# Reject command-line values for settings owned by this launcher's environment
# contract. Emitting a setting twice lets argument order silently select a cache
# layout that does not match the persistent-object namespace.
for arg in "$@"; do
  case "${arg}" in
    --kv-cache-dtype | --kv-cache-dtype=* | \
      --kv-offloading-backend | --kv-offloading-backend=* | \
      --kv-offloading-size | --kv-offloading-size=* | \
      --kv-transfer-config | --kv-transfer-config=* | \
      --enable-cumem-allocator)
      fail "${arg%%=*} is managed by CACHE_MODE and the cache environment variables"
      ;;
  esac
done

cache_mode=${CACHE_MODE:-}
if [[ -z ${cache_mode} ]]; then
  if [[ ${LMCACHE_ENABLED:-0} == 1 ]]; then
    cache_mode=lmcache
  else
    cache_mode=vram
  fi
fi
case "${cache_mode}" in
  vram | native | lmcache) ;;
  *) fail "CACHE_MODE must be vram, native, or lmcache; got ${cache_mode}" ;;
esac

# One policy selects both recurrent retention and its compatible external
# connector. A second launcher-specific policy could disagree with vLLM.
checkpoint_policy=auto
policy_value_pending=0
policy_seen=0
forwarded_args=()
for arg in "$@"; do
  if ((policy_value_pending)); then
    checkpoint_policy=${arg}
    policy_value_pending=0
    continue
  fi
  case "${arg}" in
    --recurrent-checkpoint-policy | --recurrent-checkpoint-policy=*)
      ((policy_seen == 0)) || fail 'Specify --recurrent-checkpoint-policy only once'
      policy_seen=1
      if [[ ${arg} == *=* ]]; then
        checkpoint_policy=${arg#*=}
      else
        policy_value_pending=1
      fi
      ;;
    *) forwarded_args+=("${arg}") ;;
  esac
done
((policy_value_pending == 0)) || fail '--recurrent-checkpoint-policy requires a value'
case "${checkpoint_policy}" in
  auto | aligned | request_boundaries) ;;
  *) fail "Unsupported recurrent checkpoint policy: ${checkpoint_policy}" ;;
esac
set -- "${forwarded_args[@]}"
checkpoint_model=${MODEL:-local-inference-lab/GLM-5.3-Flash-NVFP4}
if (($# > 0)) && [[ $1 != -* ]]; then
  checkpoint_model=$1
fi

# External-cache object geometry owns retention boundaries. GPU-local caching
# has no transfer-object constraint and accepts vLLM's native retention option.
if [[ ${cache_mode} != vram ]]; then
  for arg in "$@"; do
    case "${arg}" in
      --prefix-cache-retention-interval | --prefix-cache-retention-interval=*)
        fail "${arg%%=*} is managed by CACHE_MODE and the cache environment variables"
        ;;
    esac
  done
fi

tp=${TP:-4}
dcp=${DCP:-1}
require_positive_integer TP "${tp}"
require_positive_integer DCP "${dcp}"
case "${tp}" in
  2 | 4 | 8) ;;
  *) fail "The cache-complete launcher supports TP=2, TP=4 or TP=8; got ${tp}" ;;
esac
if ((tp % dcp != 0)); then
  fail "DCP must divide TP; got TP=${tp} DCP=${dcp}"
fi

interleave=${CP_KV_CACHE_INTERLEAVE_SIZE:-4}
require_positive_integer CP_KV_CACHE_INTERLEAVE_SIZE "${interleave}"
gather_selector=${DCP_CKV_GATHER:-auto}
case "${gather_selector}" in
  auto)
    if ((dcp > 1)); then
      resolved_gather=1
    else
      resolved_gather=0
    fi
    ;;
  0 | 1) resolved_gather=${DCP_CKV_GATHER} ;;
  *) fail "DCP_CKV_GATHER must be auto, 0, or 1; got ${gather_selector}" ;;
esac

kv_cache_quant=${KV_CACHE_QUANT:-}
if [[ -z ${kv_cache_quant} ]]; then
  configured_kv_cache_dtype=${KV_CACHE_DTYPE:-fp8}
  case "${configured_kv_cache_dtype}" in
    fp8 | fp8_e4m3 | fp8_ds_mla) kv_cache_quant=fp8_ds_mla ;;
    nvfp4_ds_mla) kv_cache_quant=nvfp4_ds_mla ;;
    *)
      fail "KV_CACHE_DTYPE is not representable by KV_CACHE_QUANT: ${configured_kv_cache_dtype}"
      ;;
  esac
fi
case "${kv_cache_quant}" in
  fp8_ds_mla)
    vllm_kv_cache_dtype=fp8
    lmcache_kv_cache_dtype=fp8_ds_mla
    ;;
  nvfp4_ds_mla)
    vllm_kv_cache_dtype=nvfp4_ds_mla
    lmcache_kv_cache_dtype=nvfp4_ds_mla
    ;;
  *)
    fail "KV_CACHE_QUANT must be fp8_ds_mla or nvfp4_ds_mla; got ${kv_cache_quant}"
    ;;
esac

# LMCache objects are global sequence ranges. DCP shards each target cache
# block across ranks, so automatic geometry resolves one complete per-rank
# page inside the configured retention interval. GPU-only and native-offload
# modes use the capacity-efficient 2,048-token page unless overridden.
if [[ -n ${GLM53_TARGET_BLOCK_SIZE:-} ]]; then
  target_block_size=${GLM53_TARGET_BLOCK_SIZE}
elif [[ ${cache_mode} == lmcache ]]; then
  target_block_size=auto
else
  target_block_size=2048
fi
mamba_block_size=${GLM53_MAMBA_BLOCK_SIZE:-auto}
if [[ ${target_block_size} != auto ]]; then
  require_positive_integer GLM53_TARGET_BLOCK_SIZE "${target_block_size}"
  ((target_block_size % 64 == 0)) ||
    fail "GLM53_TARGET_BLOCK_SIZE must be a multiple of 64; got ${target_block_size}"
fi
if [[ ${mamba_block_size} != auto ]]; then
  require_positive_integer GLM53_MAMBA_BLOCK_SIZE "${mamba_block_size}"
  ((mamba_block_size % 64 == 0)) ||
    fail "GLM53_MAMBA_BLOCK_SIZE must be a multiple of 64; got ${mamba_block_size}"
  if [[ ${target_block_size} != auto ]] &&
    ((mamba_block_size % target_block_size != 0 &&
      target_block_size % mamba_block_size != 0)); then
    fail "GLM53_MAMBA_BLOCK_SIZE must be a multiple or divisor of GLM53_TARGET_BLOCK_SIZE"
  fi
fi

speculator=${SPECULATOR:-mtp}
case "${speculator}" in
  mtp)
    spec_depth=${MTP_DEPTH:-${MTP:-${NUM_SPECULATIVE_TOKENS:-0}}}
    require_nonnegative_integer MTP_DEPTH "${spec_depth}"
    ((spec_depth <= 5)) || fail "MTP_DEPTH must be between 0 and 5; got ${spec_depth}"
    ;;
  dflash | dflash2)
    speculator=dflash2
    spec_depth=${DFLASH_DEPTH:-${NUM_SPECULATIVE_TOKENS:-7}}
    require_positive_integer DFLASH_DEPTH "${spec_depth}"
    ((spec_depth <= 7)) ||
      fail "DFLASH_DEPTH must be between 1 and 7; got ${spec_depth}"
    ;;
  *) fail "SPECULATOR must be mtp or dflash2; got ${speculator}" ;;
esac

export TP=${tp}
export DCP=${dcp}
export SPECULATOR=${speculator}
export NUM_SPECULATIVE_TOKENS=${spec_depth}
export KV_CACHE_DTYPE=${vllm_kv_cache_dtype}
export LMCACHE_VLLM_KV_CACHE_DTYPE=${vllm_kv_cache_dtype}
export LMCACHE_KV_CACHE_DTYPE=${lmcache_kv_cache_dtype}
export VLLM_GLM53_SPLIT_TARGET_BLOCK_SIZE=${target_block_size}
export VLLM_GLM53_SPLIT_MAMBA_BLOCK_SIZE=${mamba_block_size}

case "${cache_mode}" in
  vram)
    export LMCACHE_ENABLED=0
    if [[ ${checkpoint_policy} == aligned ]]; then
      retention_supplied=0
      for arg in "$@"; do
        case "${arg}" in
          --prefix-cache-retention-interval | --prefix-cache-retention-interval=*)
            retention_supplied=1
            ;;
        esac
      done
      # Packed native exports retain interior recurrent states without making
      # each checkpoint a separate scheduler forward. Explicit retention limits
      # remain available to deployments that prefer fewer cached states.
      if ((retention_supplied == 0)); then
        set -- "$@" --prefix-cache-retention-interval None
      fi
    fi
    ;;
  native)
    export LMCACHE_ENABLED=0
    native_offload_size=${NATIVE_KV_OFFLOADING_SIZE_GB:-64}
    [[ ${native_offload_size} =~ ^[0-9]+([.][0-9]+)?$ ]] ||
      fail "NATIVE_KV_OFFLOADING_SIZE_GB must be a positive number; got ${native_offload_size}"
    [[ ! ${native_offload_size} =~ ^0+([.]0+)?$ ]] ||
      fail "NATIVE_KV_OFFLOADING_SIZE_GB must be greater than zero"
    set -- "$@" \
      --kv-offloading-backend native \
      --kv-offloading-size "${native_offload_size}" \
      --enable-cumem-allocator
    ;;
  lmcache)
    export LMCACHE_ENABLED=1
    lmcache_transfer_mode=${LMCACHE_TRANSFER_MODE:-engine_driven}
    case "${lmcache_transfer_mode}" in
      engine_driven | lmcache_driven | auto) ;;
      *)
        fail "LMCACHE_TRANSFER_MODE must be engine_driven, lmcache_driven, or auto; got ${lmcache_transfer_mode}"
        ;;
    esac
    export LMCACHE_TRANSFER_MODE=${lmcache_transfer_mode}
    if [[ ${checkpoint_policy} == auto ]]; then
      if [[ ${lmcache_transfer_mode} == engine_driven ]]; then
        checkpoint_policy=request_boundaries
      else
        checkpoint_policy=aligned
      fi
    fi
    if [[ ${checkpoint_policy} == aligned && -n ${LMCACHE_CHECKPOINT_IDENTITY:-} ]]; then
      fail 'An immutable semantic checkpoint identity is incompatible with aligned transfers'
    fi
    if [[ ${checkpoint_policy} == request_boundaries ]]; then
      [[ ${lmcache_transfer_mode} == engine_driven ]] ||
        fail 'Request-boundary LMCache requires engine_driven transfer'
    fi
    # Both external formats need immutable weight identities. Transfer policy
    # changes the connector, not whether filesystem payloads can be reused.
    if [[ ${checkpoint_policy} == request_boundaries || ${LMCACHE_L2_ENABLED:-1} == 1 ]]; then
      for arg in "$@"; do
        case "${arg}" in
          --revision | --revision=* | --speculative-config | --speculative-config=*)
            fail "${arg%%=*} would override the authenticated checkpoint; use MODEL_REVISION and the speculative-mode environment settings"
            ;;
        esac
      done
      resolved_checkpoint_identity=${LMCACHE_CHECKPOINT_IDENTITY:-}
      if [[ -z ${resolved_checkpoint_identity} && ${CACHE_CONFIG_DRY_RUN:-0} != 1 ]]; then
        identity_speculation=none
        if ((spec_depth > 0)); then
          identity_speculation=mtp
          [[ ${speculator} != dflash2 ]] || identity_speculation=dflash
        fi
        identity_result=$(/opt/venv/bin/python \
          /usr/local/libexec/glm53_checkpoint_identity.py \
          --model "${checkpoint_model}" --revision "${MODEL_REVISION:-}" \
          --draft-model "${DFLASH_MODEL:-local-inference-lab/GLM-5.3-Flash-DFlash2}" \
          --draft-revision "${DFLASH_MODEL_REVISION:-}" \
          --speculation "${identity_speculation}")
        resolved_checkpoint_identity=$(jq -cer '.checkpoint_identity' <<< "${identity_result}")
        MODEL_REVISION=$(jq -er '.model_revision' <<< "${identity_result}")
        DFLASH_MODEL_REVISION=$(jq -er '.draft_model_revision' <<< "${identity_result}")
        export MODEL_REVISION DFLASH_MODEL_REVISION
      fi
      if [[ -n ${resolved_checkpoint_identity} ]]; then
        jq -e --argjson depth "${spec_depth}" '
          type == "object" and
          (.target_revision | type == "string" and test("^([0-9a-f]{40}|[0-9a-f]{64})$")) and
          (.source_revision | type == "string" and test("^([0-9a-f]{40}|[0-9a-f]{64})$")) and
          (.draft_revision | type == "string" and
            (test("^([0-9a-f]{40}|[0-9a-f]{64})$") or ($depth == 0 and . == "")))
        ' <<< "${resolved_checkpoint_identity}" >/dev/null ||
          fail 'Persistent checkpoint identity requires immutable target, source and active draft revisions'
        LMCACHE_MODEL_REVISION_ID=$(jq -er '.target_revision' <<< "${resolved_checkpoint_identity}")
        LMCACHE_DFLASH_REVISION_ID=$(jq -er '.draft_revision' <<< "${resolved_checkpoint_identity}")
        export LMCACHE_MODEL_REVISION_ID LMCACHE_DFLASH_REVISION_ID
        if [[ ${checkpoint_policy} == request_boundaries ]]; then
          export LMCACHE_CHECKPOINT_IDENTITY=${resolved_checkpoint_identity}
        fi
      fi
    fi
    if [[ ${lmcache_transfer_mode} == engine_driven && \
      -z ${GPU_MEMORY_UTILIZATION+x} ]]; then
      export GPU_MEMORY_UTILIZATION=0.950
    fi
    lmcache_chunk_size=${LMCACHE_CHUNK_SIZE:-4096}
    target_token_budget=${LMCACHE_TARGET_TOKEN_BUDGET:-4096}
    require_positive_integer LMCACHE_CHUNK_SIZE "${lmcache_chunk_size}"
    require_positive_integer LMCACHE_TARGET_TOKEN_BUDGET "${target_token_budget}"
    # Semantic bundles carry an exact recurrent endpoint independently of the
    # storage-object size. Aligned transfers require one complete object per
    # target budget to retain their recurrent-boundary contract.
    if [[ ${checkpoint_policy} != request_boundaries &&
      ${lmcache_chunk_size} != "${target_token_budget}" ]]; then
      fail "Aligned LMCache requires matching LMCACHE_CHUNK_SIZE and LMCACHE_TARGET_TOKEN_BUDGET; got ${lmcache_chunk_size} and ${target_token_budget}"
    fi
    if [[ ${target_block_size} != auto ]] &&
      ((lmcache_chunk_size % (target_block_size * dcp) != 0)); then
      fail "LMCACHE_CHUNK_SIZE must be a multiple of GLM53_TARGET_BLOCK_SIZE times DCP; got chunk ${lmcache_chunk_size}, block ${target_block_size}, and DCP ${dcp}"
    fi
    export LMCACHE_CHUNK_SIZE=${lmcache_chunk_size}
    export LMCACHE_TARGET_TOKEN_BUDGET=${target_token_budget}

    case "${LMCACHE_L2_ENABLED:-1}" in
      0 | 1) ;;
      *) fail "LMCACHE_L2_ENABLED must be 0 or 1; got ${LMCACHE_L2_ENABLED}" ;;
    esac
    if [[ ${LMCACHE_L2_ENABLED:-1} == 1 ]]; then
      l2_root=${LMCACHE_L2_ROOT:-${LMCACHE_L2_PATH:-/lmcache-l2}}
      [[ ${l2_root} == /* && ${l2_root} != *\"* && \
        ${l2_root} != *\\* && ${l2_root} != *\'* ]] ||
        fail "LMCACHE_L2_ROOT must be an absolute path without quotes or backslashes"

      schema=$(sanitize_namespace_component "${LMCACHE_SCHEMA_REVISION:-glm53-r18-cache-v1}")
      model_id=$(sanitize_namespace_component "${checkpoint_model}")
      # Dry-run deliberately performs no model I/O and must not claim identity.
      model_revision=$(sanitize_namespace_component "${LMCACHE_MODEL_REVISION_ID:-unresolved-dry-run}")
      draft_revision=none
      if [[ ${speculator} == dflash2 ]]; then
        draft_revision=$(sanitize_namespace_component "${LMCACHE_DFLASH_REVISION_ID:-unresolved-dry-run}")
      fi
      namespace="${schema}/${model_id}-${model_revision}/kv-${kv_cache_quant}/tp-${tp}-dcp-${dcp}-i-${interleave}-g-${resolved_gather}/spec-${speculator}-${spec_depth}-draft-${draft_revision}/chunk-${lmcache_chunk_size}"
      export LMCACHE_L2_PATH="${l2_root%/}/${namespace}"
      export LMCACHE_INSTANCE_ID=${LMCACHE_INSTANCE_ID:-"glm53-${schema}-tp${tp}-dcp${dcp}-${kv_cache_quant}"}
      if [[ ${checkpoint_policy} == request_boundaries ]]; then
        export LMCACHE_CHECKPOINT_INDEX_PATH=${LMCACHE_CHECKPOINT_INDEX_PATH:-"${LMCACHE_L2_PATH}/semantic-directory.sqlite3"}
      fi
    fi
    ;;
esac

if [[ ${cache_mode} == lmcache ]] || ((policy_seen)); then
  set -- "$@" --recurrent-checkpoint-policy "${checkpoint_policy}"
fi

if [[ ${CACHE_CONFIG_DRY_RUN:-0} == 1 ]]; then
  printf 'CACHE_MODE=%q\n' "${cache_mode}"
  printf 'KV_CACHE_QUANT=%q\n' "${kv_cache_quant}"
  printf 'TP=%q\nDCP=%q\nDCP_CKV_GATHER=%q\n' \
    "${tp}" "${dcp}" "${resolved_gather}"
  printf 'SPECULATOR=%q\nSPECULATIVE_DEPTH=%q\n' \
    "${speculator}" "${spec_depth}"
  printf 'VLLM_GLM53_SPLIT_TARGET_BLOCK_SIZE=%q\n' \
    "${VLLM_GLM53_SPLIT_TARGET_BLOCK_SIZE}"
  printf 'VLLM_GLM53_SPLIT_MAMBA_BLOCK_SIZE=%q\n' \
    "${VLLM_GLM53_SPLIT_MAMBA_BLOCK_SIZE}"
  if [[ ${cache_mode} == lmcache ]]; then
    printf 'RECURRENT_CHECKPOINT_POLICY=%q\n' "${checkpoint_policy}"
    printf 'LMCACHE_TRANSFER_MODE=%q\n' "${LMCACHE_TRANSFER_MODE}"
    printf 'LMCACHE_CHUNK_SIZE=%q\nLMCACHE_TARGET_TOKEN_BUDGET=%q\n' \
      "${LMCACHE_CHUNK_SIZE}" "${LMCACHE_TARGET_TOKEN_BUDGET}"
    printf 'GPU_MEMORY_UTILIZATION=%q\n' \
      "${GPU_MEMORY_UTILIZATION:-launcher-default}"
    printf 'LMCACHE_L2_PATH=%q\n' "${LMCACHE_L2_PATH:-disabled}"
  fi
  printf 'ARGV:'
  if (($# > 0)); then
    printf ' %q' "$@"
  fi
  printf '\n'
  exit 0
fi

exec "${cache_launcher}" "$@"
