# Repository guidance

## Integration container changelog

Changes to the community container publisher or its integration-channel policy
must preserve the source-fragment workflow documented in
`docs/integration-release-changelog.md`. Do not add a separately maintained
release-history file or hand-written component list. The GitHub container
release, `release-changelog.json`, and the runtime manifest must be generated
from the exact component revisions in the assembly.

When changing the fragment schema, validation, rendering, or channel policy,
update the runbook and the publisher tests in the same pull request.
