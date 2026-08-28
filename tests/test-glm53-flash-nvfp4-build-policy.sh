#!/usr/bin/env bash
set -euo pipefail

# Validate that qualified tags are immutable with respect to recipe commits and
# that build-input overrides cannot inherit qualified release provenance.

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_script="${repo_root}/build-glm53-flash-nvfp4-jovian-cu133-torch213.sh"

expect_failure() {
  local expected=$1
  shift
  local output
  if output="$("$@" 2>&1)"; then
    printf 'Command unexpectedly succeeded: %q' "$1" >&2
    printf ' %q' "${@:2}" >&2
    printf '\n' >&2
    exit 1
  fi
  grep -Fq -- "${expected}" <<<"${output}" || {
    printf 'Expected failure text %q, got:\n%s\n' "${expected}" "${output}" >&2
    exit 1
  }
}

cd "${repo_root}"
head_commit="$(git rev-parse HEAD)"
qualified_config="$(
  env -u BASE_IMAGE -u VLLM_PACKAGE_VERSION -u IMAGE \
    -u ALLOW_DIRTY_BUILD -u PUSH_IMAGE \
    PRINT_RELEASE_CONFIG=1 "${build_script}"
)"
grep -Fxq "status=qualified" <<<"${qualified_config}"
grep -Fxq "docker_commit=${head_commit}" <<<"${qualified_config}"
grep -Eq "^image=.*-git${head_commit:0:12}$" <<<"${qualified_config}"

expect_failure 'require an explicit non-release IMAGE tag' \
  env PRINT_RELEASE_CONFIG=1 BASE_IMAGE=example.invalid/runtime@sha256:deadbeef \
  "${build_script}"
expect_failure 'require an explicit non-release IMAGE tag' \
  env PRINT_RELEASE_CONFIG=1 VLLM_PACKAGE_VERSION=0.0.0 "${build_script}"
expect_failure 'require an explicit non-release IMAGE tag' \
  env PRINT_RELEASE_CONFIG=1 ALLOW_DIRTY_BUILD=1 "${build_script}"
expect_failure 'must contain a dev, test, or scratch marker' \
  env PRINT_RELEASE_CONFIG=1 VLLM_PACKAGE_VERSION=0.0.0 \
  IMAGE=voipmonitor/vllm:qualified-override "${build_script}"
expect_failure 'PUSH_IMAGE=1 is forbidden when ALLOW_DIRTY_BUILD=1' \
  env PRINT_RELEASE_CONFIG=1 ALLOW_DIRTY_BUILD=1 PUSH_IMAGE=1 \
  IMAGE=voipmonitor/vllm:glm53-dev-dirty "${build_script}"

override_config="$(
  env PRINT_RELEASE_CONFIG=1 VLLM_PACKAGE_VERSION=0.0.0 \
    IMAGE=voipmonitor/vllm:glm53-dev-package-override "${build_script}"
)"
grep -Fxq 'status=research-only' <<<"${override_config}"
grep -Fxq 'image=voipmonitor/vllm:glm53-dev-package-override' <<<"${override_config}"
grep -Fxq 'vllm_package_version=0.0.0' <<<"${override_config}"
