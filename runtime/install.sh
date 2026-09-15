#!/usr/bin/env bash
# Install inside a model-neutral image that already contains its serving packages.
set -euo pipefail
if (($# < 2)); then
    echo 'Usage: install.sh FOUNDATION_INSPECT_JSON RUNTIME_LOCK_SHA256 [BOOTSTRAP_EXECUTABLE]' >&2
    exit 2
fi
source_root=$(cd -- "$(dirname -- "$0")/.." && pwd)
cd "$source_root"
bootstrap=()
if (($# == 3)); then bootstrap=(--bootstrap "$3"); fi
# Audit before installing entrypoints. Profile-owned Docker ENV is a build error.
/opt/venv/bin/python -m runtime.packaging \
    --image-inspect "$1" --runtime-lock-sha256 "$2" "${bootstrap[@]}" >/dev/null
install -d /opt/lil/runtime /usr/local/bin
for file in runtime/*.py runtime/schema.json runtime/options.yaml \
    runtime/requirements.txt runtime/install.sh runtime/lil-serve; do
    install -m644 "$file" "/opt/lil/runtime/$(basename "$file")"
done
cp -a runtime/profiles runtime/hardware /opt/lil/runtime/
/opt/venv/bin/python -m runtime.packaging \
    --image-inspect "$1" --runtime-lock-sha256 "$2" "${bootstrap[@]}" \
    > /opt/lil/runtime/image-contract.json
install -m755 runtime/lil-serve /usr/local/bin/lil-serve
# Legacy cache entrypoints are deliberately not replaced until their external
# lifecycle adapters satisfy the migration contract in runtime/README.md.
