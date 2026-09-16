#!/usr/bin/env bash
# Build the revision-pinned Qwen3.8 QSRT K5 plus rank-16 serving image.

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$repo_root"

base_image=${BASE_IMAGE:-voipmonitor/vllm@sha256:ff9d4f2402ed88b1ae7ca3a6886c80a64d72993f1a593380c8cb6f193437567d}
release_date=${RELEASE_DATE:-20260821}
revision=${REVISION:-r7}
b12x_runtime_tree=28f10076d5df9898d0d68ac41edec0f787c93b57
vllm_runtime_tree=188217905a37f04ce50659441a03fa4f7256435f
reconciliation=patches/qwen38-qsrt/b12x-pr236-r16-reconciliation.patch
launcher=launchers/serve-qwen38-qsrt-k5
vllm_prs='285@d6b7377cbf27836241d04c9dd08725b7426d88eb,286@9dbec23e2b2ad5f788838ee3dafc3f6623b500b5,287@a5e76352ba295bf1bbc80c9bce52b2b7e280c24e,288@29f13ebc717d10d4aef5f728598c9ab0f75b082e,289@91fd9f8f74de3882af4802d48e66f23347e02cd0,290@c13a214bd70319938136fa38757a9ea739569639,292@fb8d983acd642f059e5f3d3466e0f1c13e48b083,293@0e41faa811ce46df77acce3e9e7c6887954925de,294@826bb4088f4671fe79e01183e0301f79134662c1,295@71b93dc6f3d59e004c18535cf08b8fe750b64c19,296@31472fd994f99802006a6297557d641b88652ceb,298@b8d92cb5d26e1c6fc4c5ad513b806dae961a4afd,300@901a7c50e5d344b5bea975c3393c4dc23c958fc1,301@2255f632485c0f8a4e6cc81bc052a818b52a38a8,302@7d1c21353cf4563b5344c83cf53acecac1f2f99c,303@4b297d1a07bfcc1bf0ab14c1dc25fe59c3e8f081,304@229de6270e511701045fd73af592620901c7422b,308@053e6351d0b3b3e35c969c9e3933db64d30a7164,309@dc0c026df62448d1bec747d9dd6fb0a01d838f3e,320@e9534672129b961399b1625d33d83c79eacded30,415@c805ebd0896ccfbd2569bc0b2a7944d3282106ff,417@2511e5df2b1e4dfd2360a28e89899c90b7b3fcc7,461@0942707892ca26c5b379fb9a6b88b3f8468a5adc,462@1d341d8482e4bcc6fd2404efbd245207d6deddf1'
b12x_prs='145@7f88972df71d580951115220b75923078b769fe8,221@413f96e889dad1ae0752fd1f4be9d37f56849600,223@e99775f552c4f28cf1f345ded28bb77a57ea6a83,227@e38436d76a95c586c57e06646f8ea5b8c8ed11c7,228@50046df84a15cc5f76b94260e897fd39072b2fdf,229@2cdd9e265cd6c4dca43e7d42c5a8cb265c92adfb,230@156920046e858f413db0c51e53cd25b9020d5f40,236@1dfe87039951735b39e937e4a1d1bffe25b79e79'

if [[ -n $(git status --porcelain --untracked-files=all) ]] \
    && [[ ${ALLOW_DIRTY_BUILD:-0} != 1 ]]; then
    printf 'The Qwen runtime recipe must be committed before build.\n' >&2
    git status --short >&2
    exit 1
fi

if ! docker image inspect "$base_image" >/dev/null 2>&1; then
    docker pull "$base_image"
fi
base_image_id=$(docker image inspect "$base_image" --format '{{.Id}}')
test "$base_image_id" = sha256:b8ce67bd8ed86ad9a77affe63105b1ace4f7a6a8e09b41e1ba5deb9379a3e81e

docker_commit=$(git rev-parse HEAD)
reconciliation_sha256=$(sha256sum "$reconciliation" | cut -d' ' -f1)
launcher_sha256=$(sha256sum "$launcher" | cut -d' ' -f1)
image=${IMAGE:-voipmonitor/vllm:qwen38-qsrt-k5-r16-vllm${vllm_runtime_tree:0:7}-b12x${b12x_runtime_tree:0:7}-cu133-torch213-${release_date}-${revision}}

printf 'image=%s\nbase=%s\nvllm_tree=%s\nb12x_tree=%s\n' \
    "$image" "$base_image" "$vllm_runtime_tree" "$b12x_runtime_tree"

DOCKER_BUILDKIT=1 docker build \
    --pull=false \
    --build-arg "BASE_IMAGE=$base_image" \
    --build-arg "BASE_IMAGE_ID=$base_image_id" \
    --build-arg "B12X_RECONCILIATION_SHA256=$reconciliation_sha256" \
    --build-arg "VLLM_PRS=$vllm_prs" \
    --build-arg "B12X_PRS=$b12x_prs" \
    --build-arg "RELEASE_DATE=$release_date" \
    --build-arg "DOCKER_COMMIT=$docker_commit" \
    --build-arg "LAUNCHER_SHA256=$launcher_sha256" \
    --file Dockerfile.qwen38-qsrt-k5-runtime \
    --tag "$image" \
    .

labels=$(docker image inspect "$image" --format '{{json .Config.Labels}}')
assert_label() {
    local key=$1 expected=$2
    jq -e --arg key "$key" --arg expected "$expected" \
        '.[$key] == $expected' <<<"$labels" >/dev/null
}
assert_label local-inference.runtime.base-id "$base_image_id"
assert_label local-inference.vllm.integration.tree "$vllm_runtime_tree"
assert_label local-inference.b12x.integration.tree "$b12x_runtime_tree"
assert_label local-inference.b12x.reconciliation.sha256 "$reconciliation_sha256"
assert_label local-inference.qwen38.launcher.sha256 "$launcher_sha256"
assert_label local-inference.qwen38.cudagraphs unsupported
assert_label local-inference.qwen38.torch-compile enabled

docker run --rm --entrypoint /opt/venv/bin/python "$image" -c \
    'import importlib.metadata as m; e={(x.name,x.value) for x in m.entry_points(group="vllm.general_plugins")}; assert ("b12x_qsrt", "b12x.integration.vllm.qsrt_plugin:register_b12x_qsrt") in e'
docker run --rm --entrypoint /bin/bash "$image" -n \
    /usr/local/bin/serve-qwen38-qsrt-k5

if [[ ${PUSH_IMAGE:-0} == 1 ]]; then
    docker push "$image"
fi

docker image inspect "$image" --format \
    'image={{.Id}} size={{.Size}} entrypoint={{json .Config.Entrypoint}}'
printf '%s\n' "$image"
