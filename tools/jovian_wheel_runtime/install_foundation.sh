#!/usr/bin/env bash
# Install the locked Python foundation into an isolated Python 3.12 environment.
set -euo pipefail

bundle_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
venv_path=${1:-.venv-jovian}
uv_binary=${UV_BIN:-uv}

if ! uv_path=$(command -v "${uv_binary}"); then
  printf 'uv is required; set UV_BIN to its absolute path.\n' >&2
  exit 1
fi
expected_uv_version=$(awk -F= \
  '$1 == "uv.version" {print $2; found=1} END {exit !found}' \
  "${bundle_dir}/foundation.lock")
expected_uv_sha256=$(awk -F= \
  '$1 == "uv.sha256" {print $2; found=1} END {exit !found}' \
  "${bundle_dir}/foundation.lock")
test "$("${uv_path}" --version | awk '{print $2}')" = "${expected_uv_version}"
test "$(sha256sum "${uv_path}" | awk '{print $1}')" = "${expected_uv_sha256}"

(cd "${bundle_dir}" && sha256sum --check SHA256SUMS)
"${uv_path}" venv --python 3.12 "${venv_path}"
"${uv_path}" pip install \
  --python "${venv_path}/bin/python" \
  --require-hashes \
  --no-index \
  --find-links "${bundle_dir}/wheels" \
  --no-deps \
  -r "${bundle_dir}/requirements-foundation.txt"

printf 'foundation_venv=%s status=installed native_runtime=required\n' \
  "$(realpath "${venv_path}")"
