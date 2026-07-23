#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="$(mktemp -d)"
trap 'rm -rf "${tmp_root}"' EXIT

cat >"${tmp_root}/params-good" <<'EOF'
EnableResizableBar: 1
DmaRemapPeerMmio: 1
GrdmaPciTopoCheckOverride: 1
RegistryDwords: "ForceP2P=0x11;RMForceP2PType=1;RMPcieP2PType=2;GrdmaPciTopoCheckOverride=1;EnableResizableBar=1"
EOF

cat >"${tmp_root}/params-bad" <<'EOF'
EnableResizableBar: 1
EOF

run_launcher() {
  DRY_RUN="${3:-1}" \
  XDG_CACHE_HOME="${tmp_root}/cache" \
  TMPDIR="${tmp_root}/tmp" \
  MODEL=/tmp/model \
  LOAD_FORMAT=dummy \
  NVIDIA_PARAMS_PATH="$1" \
  P2P_PREFLIGHT="$2" \
  bash "${repo_root}/launchers/serve-glm52-v16.sh"
}

mkdir -p "${tmp_root}/bin"
cat >"${tmp_root}/bin/vllm" <<'EOF'
#!/usr/bin/env bash
printf 'FAKE_VLLM_REACHED\n'
EOF
chmod +x "${tmp_root}/bin/vllm"

good_output="$(run_launcher "${tmp_root}/params-good" strict 2>&1)"
grep -Fxq 'P2P_DRIVER_PREFLIGHT=pass' <<<"${good_output}"

# Exercise the real launcher path outside command substitution. Bash suppresses
# errexit in some substitution contexts, which previously hid a false `[[ ]]`
# result inside the preflight function.
PATH="${tmp_root}/bin:${PATH}" \
  run_launcher "${tmp_root}/params-good" strict 0 \
  >"${tmp_root}/non-dry.log" 2>&1
grep -Fxq 'FAKE_VLLM_REACHED' "${tmp_root}/non-dry.log"

if bad_output="$(run_launcher "${tmp_root}/params-bad" strict 2>&1)"; then
  echo "strict P2P preflight unexpectedly accepted an incomplete config" >&2
  exit 1
fi
grep -Fq 'RMForceP2PType=1' <<<"${bad_output}"
grep -Fq 'DmaRemapPeerMmio: 1' <<<"${bad_output}"

run_launcher "${tmp_root}/params-bad" off >/dev/null
