# syntax=docker/dockerfile:1.7

ARG SOURCE_IMAGE
FROM ${SOURCE_IMAGE} AS repack

ARG SOURCE_DATE_EPOCH
COPY tools/jovian_wheel_runtime/repack_installed_distribution.py /opt/local-inference/
RUN mkdir -p /artifacts/wheels \
 && python /opt/local-inference/repack_installed_distribution.py \
      --output-dir /artifacts/wheels \
      --source-date-epoch "${SOURCE_DATE_EPOCH}" \
      triton triton-kernels flash-attn \
      > /artifacts/repack-provenance.json

FROM ${SOURCE_IMAGE} AS validate
COPY --from=repack /artifacts /artifacts
RUN mkdir -p /tmp/repack-validation \
 && python -m pip install --no-deps --target /tmp/repack-validation \
      /artifacts/wheels/*.whl \
 && python - <<'PY'
import importlib
import importlib.metadata
import json
import sys
from pathlib import Path

root = Path("/tmp/repack-validation").resolve()
sys.path.insert(0, str(root))
manifest = json.loads(Path("/artifacts/repack-provenance.json").read_text())
installed = {
    distribution.metadata["Name"].lower().replace("_", "-"): distribution.version
    for distribution in importlib.metadata.distributions(path=[root])
}
for package in manifest["packages"]:
    name = package["name"].lower().replace("_", "-")
    assert installed[name] == package["version"]
for module in ("triton", "triton_kernels", "flash_attn"):
    imported = importlib.import_module(module)
    assert Path(imported.__file__).resolve().is_relative_to(root)
print("Installed-distribution wheel repack: PASS")
PY

FROM scratch AS artifacts
COPY --from=validate /artifacts /
