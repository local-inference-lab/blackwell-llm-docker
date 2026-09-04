#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
builder="${repo_root}/build-deepseek-jovian-judgement-cu133-torch213.sh"
compose="${repo_root}/examples/docker-compose-ds4-vision-jovian-judgement-r3.yml"
composition_root="${repo_root}/patches/releases/jovian-judgement-ds4-r3"

for component in vllm b12x lmcache; do
  lock="${composition_root}/${component}/integration.lock.json"
  patch="${composition_root}/${component}/integration.patch"

  test -f "${lock}"
  test -f "${patch}"
  jq -e '
    .schema_version == 1 and
    (.composition_strategy == "merge" or
      .composition_strategy == "cherry_pick") and
    (.base.repository | type == "string") and
    (.base.ref | type == "string") and
    (.base.commit | test("^[0-9a-f]{40}$")) and
    (.result.tree | test("^[0-9a-f]{40}$")) and
    (.result.patch_sha256 | test("^[0-9a-f]{64}$")) and
    ((.research_changes // []) | length) == 0 and
    (.source_patches | length) == 0
  ' "${lock}" >/dev/null
  echo "$(jq -er '.result.patch_sha256' "${lock}")  ${patch}" |
    sha256sum -c - >/dev/null
done

jq -e '
  .composition_strategy == "cherry_pick" and
  .base.ref == "refs/heads/dev/jovian-judgement" and
  .base.commit == "a50ebee1d2460d22386b54e79f46236376e2b486" and
  .result.tree == "d6f9e777bdf23304ace1ce3b311935390009a149" and
  [.pull_requests[].number] == [628, 630, 634] and
  .pull_requests[0].head ==
    "cbb66bdff1763c174ebc794a7f968930e956580f" and
  .pull_requests[1].head ==
    "5b6fb80f5c868b62da2c01c3f52861b34c84d8ac" and
  .pull_requests[2].head ==
    "19f6d7b0f75ee3cf77e13795523886b37bbf5b06"
' "${composition_root}/vllm/integration.lock.json" >/dev/null

jq -e '
  .composition_strategy == "cherry_pick" and
  .base.ref == "refs/heads/master" and
  .base.commit == "1a7e3ec286b0ff0b7c2aabee22dce08daab7e011" and
  .result.tree == "283a63ee552d38e6a2ffa8a9ec2859ddcb227201" and
  [.pull_requests[].number] == [246, 302, 301, 306] and
  .pull_requests[0].head ==
    "ea76030d6c2353d8cf35522b4eeedfa29e7aca67" and
  .pull_requests[1].head ==
    "f8a7b9c4070754ed4399c71fc4306d8970711e1c" and
  .pull_requests[2].head ==
    "223f88c23601689057d07700bedc650eac521526" and
  .pull_requests[3].head ==
    "3f6896dcd3ad221fb2ecb4737384131305d7c0ef"
' "${composition_root}/b12x/integration.lock.json" >/dev/null

jq -e '
  .base.ref == "refs/heads/release/v0.5.2-glm52-dcp-base" and
  .base.commit == "a128b2e286ebb3556cb43124149e600ff99fe481" and
  .composition_strategy == "merge" and
  .result.tree == "eb4c227f68a4e1c45d6b8edf6b4934e18f6d1f8b" and
  (.pull_requests | length) == 14 and
  .pull_requests[-1].number == 44 and
  .pull_requests[-1].head ==
    "97ede799d6605ca1bd5285582df4e74a3d3c7b0d"
' "${composition_root}/lmcache/integration.lock.json" >/dev/null

output="$(
  REVISION=r3 \
    COMPOSITION_ROOT=patches/releases/jovian-judgement-ds4-r3 \
    PRINT_RELEASE_CONFIG=1 \
    "${builder}"
)"
grep -Fxq 'release=jovian-judgement-deepseek-v4-flash-cu133-torch213' \
  <<<"${output}"
grep -Fxq 'revision=r3' <<<"${output}"
grep -Fxq 'vllm_ref=dev/jovian-judgement' <<<"${output}"
grep -Fxq 'vllm_tree=d6f9e777bdf23304ace1ce3b311935390009a149' \
  <<<"${output}"
grep -Fxq 'b12x_tree=283a63ee552d38e6a2ffa8a9ec2859ddcb227201' \
  <<<"${output}"
grep -Fq 'fi803c466-cu133-torch213-20260904-r3' <<<"${output}"
grep -Fq 'releases/jovian-judgement-ds4-r3/vllm/integration.patch' \
  <<<"${output}"
grep -Fq 'releases/jovian-judgement-ds4-r3/b12x/integration.patch' \
  <<<"${output}"

config="$(docker compose -f "${compose}" config)"
grep -Fq 'DS4_MODEL_VARIANT: vision' <<<"${config}"
grep -Fq 'MODEL: deepseek-ai/DeepSeek-V4-Flash-Vision-Exp' <<<"${config}"
grep -Fq 'MODEL_REVISION: 6821d6ad3681a4b137b066b76094fa82ebd0a380' \
  <<<"${config}"
grep -Fq 'MODE: dspark' <<<"${config}"
grep -Fq 'BACKEND: b12x-a8-dglin' <<<"${config}"
grep -Fq 'TP_SIZE: "2"' <<<"${config}"
grep -Fq 'DCP_SIZE: "1"' <<<"${config}"
grep -Fq 'DSPARK_DEPTH_MODE: fixed' <<<"${config}"
grep -Fq 'DSPARK_TOKENS: "3"' <<<"${config}"
grep -Fq 'MAX_NUM_SEQS: "4"' <<<"${config}"
grep -Fq 'GRAPH: auto' <<<"${config}"
grep -Fq 'MAX_MODEL_LEN: ""' <<<"${config}"
grep -Fq 'MAX_NUM_BATCHED_TOKENS: "4096"' <<<"${config}"
grep -Fq 'GPU_MEMORY_UTILIZATION: ""' <<<"${config}"
grep -Fq 'LMCACHE_MODE: "off"' <<<"${config}"
grep -Fq 'LMCACHE_TRANSFER_MODE: auto' <<<"${config}"
grep -Fq 'LOAD_FORMAT: instanttensor' <<<"${config}"
grep -Fq 'INSTANTTENSOR_BACKEND: BUFFERED' <<<"${config}"
grep -Fq 'jovian-judgement-vllmd6f9e77-b12x283a63e-fi803c466' <<<"${config}"
! grep -Fq 'KV_OFFLOADING_SIZE:' <<<"${config}"
! grep -Fq 'NATIVE_L2_' <<<"${config}"

for component in VLLM B12X LMCACHE; do
  grep -Fq -- \
    "--build-arg \"${component}_UPSTREAM_BASE=\${${component}_UPSTREAM_BASE}\"" \
    "${builder}"
  grep -Fq -- \
    "--build-arg \"${component}_MERGE_HEADS=\${${component}_MERGE_HEADS}\"" \
    "${builder}"
done

grep -Fq 'flashinfer-wheels-fi803c466-cu133-torch213-20260904-r1@sha256:79edbc91874d9468e3e6268e1584503e3dec55f2a4d3bdd70d5c43e9b41675c7' \
  "${builder}"
grep -Fq 'FLASHINFER_COMMIT:-803c4664f4771ddc418f20a57f752469a237a825' \
  "${builder}"
