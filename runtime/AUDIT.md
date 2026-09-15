# Launch configuration source audit

Status: **implemented** source/configuration analysis, not GPU qualification.
Inspection date: 2026-09-15. No running service, clock, image tag or published
wiki recipe was changed by this audit.

## Source boundaries

- Docker recipes: `local-inference-lab/blackwell-llm-docker`, main commit
  `fa539cf1b58b734e3f89fc0bdfe4ac20eb6ecb13`.
- Model wiki/Compose: `local-inference-lab/rtx6kpro`, master commit
  `2b76bbd9985cbace7137a4dc653a2c9b7055c60e`.
- Native DeepSeek script inspection: `local-inference-lab/vllm`,
  `serve-ds4-flash.sh` and `serve-ds41-flash.sh` as represented in the #739
  review checkout containing JJ `5bca5a58d970216bd46be82575e824c6e424c465`.

## Findings

| Hypothesis | Source evidence | Conclusion |
|---|---|---|
| GLM MTP depends on a cache environment setting | `serve-glm53-flash-cache-complete.sh` takes its early delegation branch before normalizing MTP_DEPTH; the base launcher reads NUM_SPECULATIVE_TOKENS/MTP, not MTP_DEPTH | Confirmed interface defect; model selection/speculation must resolve before cache dispatch |
| DS4.1 overwrites explicit OMP=1/capture=256 | `serve-ds41-jovian.sh` unsets values equal to GLM image defaults, then supplies 8/128 | Confirmed; identical baked and explicit values cannot be distinguished inside the process |
| GLM lacks scheduler environment mappings | `serve-glm53-flash-nvfp4-dflash2-scheduler-qos.sh` handles share/half-life/lanes/policy/refill, auto values and CLI precedence | Not true at the inspected recipe boundary; preserve the implemented behavior |
| TP/DCP and LMCache capacities have inconsistent aliases | GLM uses TP/DCP and LMCACHE_L1_SIZE_GB; DS4 accepts TP_SIZE/DCP_SIZE and LMCACHE_L1_GB; DS4.1 wrapper supplies only TP_SIZE/DCP_SIZE | Confirmed interface inconsistency, not a reason to make every model's default TP or memory budget equal |
| Qwen depends on an independent Compose policy copy | Wiki Compose contains a complete native argv, private-head/PLE/kernel settings, and qwen38-r281 JIT paths while choosing the R35 image | Confirmed duplication; a tag update alone cannot update those settings |
| Wheel and community runtime launch behavior are identical | The wheel entrypoint initializes CUDA/NCCL paths and executes its command; it does not implement community model/cache policy | Not identical; preserve bootstrap and validate package capabilities independently |

Additional preservation constraints:

- `BACKEND=b12x-a8-dglin` in the DS4 native script omits an explicit dense
  backend. Preserve that behavior and verify actual DeepGEMM dispatch; do
  not describe an omitted argument as proof of a particular kernel.
- DS4.1 wrapper's compilation JSON replaces the native script's JSON and
  omits `custom_ops:[all]`. The DS4.1 profile preserves the effective wrapper
  object; a separate optimization/compatibility decision is needed to change it.
- The community image's global CUDA capture-size list is GLM-specific. The
  native DS4.1 script does not consume that list. The DS4.1 profile therefore
  does not acquire GLM graph rows just because it shares a runtime image.
- Qwen's 6,019-token budget and two OpenMP threads differ from the GLM
  4,096-token budget/one thread. These are recorded model-policy differences,
  not arithmetic mistakes to normalize away.
- The community build installs dependency patches and native libraries in
  addition to vLLM/B12X/LMCache Python sources. Profiles cannot substitute for
  source-lock/native-ABI checks. No dependency patch is removed by this work.

## Commit scope

The configuration implementation and generated examples are additive. They
have no default deployment cutover. `recipes/glm53`, the CUDA wheel build
entrypoints, and the published wiki recipes remain operationally unchanged.
Docker #39's canonical DS4 entrypoint correction and the separate source-overlay
cleanup are independent changes; this configuration branch does not supersede
or merge them.

The [runtime specification](README.md) defines the production migration gates,
including adapters for cache process lifecycles and a neutral final image
environment. They remain explicit implementation work, not claims implied
by a passing configuration test count.
