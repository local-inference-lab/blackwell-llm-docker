# Kimi-K3 QSRT TP8 DSpark scheduler safety

Status: **qualified** for QSRT-K2 TP8/DCP8 with Inferact DSpark depth 3.

## Runtime contract

Kimi-K3 TP8/DCP8 exposes a 1,536-token recurrent scheduler block. Parallel
DSpark drafting adds `depth - 1` batch slots per active request. The total
worker batch capacity and target scheduler budget are therefore different
quantities:

```text
parallel draft reserve = (3 - 1) * 2 = 4 slots
max_num_scheduled_tokens = 1536
max_num_batched_tokens = 1536 + 4 = 1540
```

`serve-kimi-k3-qsrt-dspark-tp8` applies this profile. The generic QSRT DSpark
launcher accepts an explicit `MAX_NUM_SCHEDULED_TOKENS` and rejects a total
batch budget that cannot hold both the target scheduler batch and every draft
slot.

The generic launcher's historical defaults remain unchanged: TP16/DCP8,
DSpark depth 7, 4,096 total batch tokens, and eight active sequences. Operators
who do not select an explicit target scheduler budget retain vLLM's automatic
budget derivation.

## Failure mode

The prior TP8 adaptation set `max_num_batched_tokens=1536` with depth 3 and two
sequences. vLLM reserved four draft slots and silently reduced the target
scheduler budget to 1,532 tokens. A 1,536-token prompt was split as `1532 + 4`.
The four-token partial-prefill tail entered fixed-M W4A16 execution with graph
padding routes.

A B12X compatibility overlay provided the Kimi QSRT atoms-v2 planner ABI but
predated the inactive-route behavior tracked by B12X PR #227. The fixed-M
kernel could therefore address an inactive expert route instead of giving it
zero effective weight. Direct vLLM SSE already contained Kimi control markers,
low-entropy repetition, or token salad; the gateway only preserved those
bytes.

Greedy, unique-salt boundary probes isolated the failure:

| Prompt tokens | Pre-fix result |
|---:|---|
| 1,532 | stable |
| 1,533 | stable |
| 1,534 | stable |
| 1,536 | 4/4 NaN logprobs, HTTP 400 |

The API error was:

```text
Out of range float values are not JSON compliant: nan
```

## Fail-closed package validation

The image now installs `validate-kimi-k3-qsrt-runtime.py`. The build and the
QSRT launcher require all of these contracts from the active imported B12X
tree:

- weight planning accepts `qsrt_storage_format` and `qsrt_profile`;
- weight preparation accepts the atoms-v2 payload, first-slot, and layer index;
- fixed-M W4A16 phase 0/1 stages inactive routes;
- FC2-only runtime-M phase 2 validates routes inline rather than staging a
  fixed route table.

This catches the packaging class where image labels describe a safe B12X tree
but a later bind mount or replacement Python package shadows it with an older
compatibility tree.

The B12X product correction remains owned by B12X PR #227. The QSRT
compatibility-base port is B12X PR #237. This deployment change does not
duplicate either kernel patch; it validates the behavior at the image and
launcher boundary.

## Qualification

The exact four-token tail was re-exercised at 1,540 prompt tokens after keeping
the target scheduler budget at 1,536. Ten unique-salt greedy runs completed,
all selected the same first token, and none emitted NaN, control markers, or
token salad.

The captured production incident payload then passed five direct-vLLM runs and
five llmconduit runs. Same-process local-prefix replay reused 1,536 tokens and
matched cold output.

A 14,505-token deterministic cache fixture returned `K3_CACHE_OK` across all
three cache arms:

| Arm | Local hit | External hit | Result |
|---|---:|---:|---|
| cold producer | 0 | 0 | `K3_CACHE_OK` |
| APC reset / LMCache L1 | 0 | 12,288 | byte-equivalent |
| same-image restart / LMCache L2 | 0 | 12,288 | byte-equivalent |

The final runtime had zero restarts, no OOM, no NaN/nonfinite/CUBLAS/illegal
memory/worker-death markers, and no thermal slowdown.

Machine-readable receipt:

```text
validation/kimi-k3-qsrt-tp8-scheduler-inactive-routes-20260819.json
```

## Compatibility and limits

- This qualification covers the language-only QSRT-K2 target, not the official
  MXFP4 native-vision TP16 contract.
- The scheduler invariant is generic, but the provided TP8 defaults are
  specific to a 1,536-token recurrent block, depth 3, and two active sequences.
- Changing speculative depth or sequence count requires recomputing the draft
  reserve and re-running exact boundary probes.
- Repetition detection and client retry remain containment only; they do not
  repair the first corrupt logit.

AI assistance was used for root-cause analysis, implementation, test design,
and PR drafting. The submitting human must review every changed line and be
able to defend the scheduler and B12X contracts before merge.
