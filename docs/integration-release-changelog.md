# Integration container changelog

The community container publisher builds one changelog from the exact vLLM and
B12X revisions in each container assembly. The changelog is published in three
forms:

- a readable **Changes since** section in the GitHub container release;
- a `release-changelog.json` release asset for tools and the website;
- the same structured data under `release_changelog` in the runtime
  `manifest.json`.

The implementation change and its release description therefore travel through
the integration branch together. The container repository does not maintain a
second hand-written list of component changes.

## Add a fragment to a component change

Every runtime-affecting vLLM or B12X change destined for an `integration/*`
branch must add one JSON file under `.lil/changes/`. Add the fragment to the PR
that implements the behavior, or to the commit that imports the reviewed PR
into the integration branch.

Use a globally unique lowercase identifier prefixed by the component name. A PR
number is the preferred suffix:

```text
.lil/changes/vllm-816.json
.lil/changes/b12x-402.json
```

Example:

```json
{
  "schema": "local-inference-release-change/v1",
  "id": "vllm-816",
  "category": "fix",
  "summary": "Merge QSA selection across DCP ranks",
  "models": ["Qwen3.8-Flash-Next"],
  "compatibility": "No user action required.",
  "details": [
    "DCP1, DCP2, and DCP4 use the same distributed selection contract."
  ],
  "pull_requests": [816],
  "requires": ["b12x-402"],
  "evidence": [
    "https://github.com/local-inference-lab/vllm/pull/816#issuecomment-example"
  ]
}
```

Required fields:

| Field | Contract |
|---|---|
| `schema` | Must be `local-inference-release-change/v1`. |
| `id` | Stable component-prefixed lowercase identifier; it must match the filename. |
| `category` | `breaking`, `compatibility`, `feature`, `fix`, `internal`, or `performance`. |
| `summary` | One user-readable sentence describing resulting behavior. |
| `models` | Exact affected model names, or `all` for a runtime-wide change. |
| `compatibility` | Required user action, or exactly `No user action required.` |

Optional fields:

| Field | Contract |
|---|---|
| `details` | Short behavior details that belong in release notes. |
| `pull_requests` | PR numbers in the fragment's component repository. The publisher adds links and contributor names. |
| `authors` | GitHub handles for a direct integration commit with no PR. At least one author is required when `pull_requests` is empty. |
| `requires` | Fragment identifiers for cross-repository dependencies. |
| `evidence` | URLs for a benchmark, validation report, or correctness reproducer. |

Write for a container user. Do not put source hashes, branch choreography,
rejected attempts, or claims that are not supported by the linked validation in
the fragment. Exact commits are added automatically from the assembly.

## Importing or pushing a change

1. Verify the implementation and its contributor attribution.
2. Add or preserve the component fragment in the same commit series.
3. Push the complete series to the appropriate `integration/*` branch.
4. Let the component workflow publish the source-addressed wheel. Its dispatch
   starts the container workflow.
5. Read the generated container prerelease. Confirm the human changelog and the
   attached `release-changelog.json` before announcing the image.

A direct integration commit must still have a fragment. Use `category:
internal` for a runtime-relevant maintenance change that does not alter user
behavior; do not omit the fragment.

## Immutability and corrections

Once a container release contains a fragment, the publisher rejects deletion or
modification of that fragment in later component revisions. This keeps old and
new release notes reproducible from source.

To correct a published description or behavior, add a new fragment that states
the correction and, when applicable, names the earlier fragment in `requires`.
Do not edit the published fragment.

## Publisher behavior

For each release channel, the publisher finds the most recent qualified
container release and compares its exact component commits with the proposed
assembly. It then:

1. reads `.lil/changes/` at both exact component revisions;
2. rejects changed or deleted published fragments;
3. rejects an integration component revision change with no added fragment;
4. validates cross-component `requires` identifiers;
5. resolves PR links and contributor handles from GitHub;
6. embeds the compiled changelog in the runtime manifest and publishes it as a
   separate release asset;
7. renders the readable section of the existing GitHub container prerelease.

Stable channels use the same fragments, but compare against the previous stable
container release. A change can therefore appear once in a beta channel and
once later in the corresponding stable channel without maintaining two texts.

The channel policy is defined alongside branch selection in
`tools/jovian_wheel_runtime/community-channel.json`. vLLM and B12X fragments are
mandatory for the `beta` and `karmic-kraken-beta` release channels.
