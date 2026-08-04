# Security policy

## Supported versions

Security fixes are applied to the latest commit on `main`. There is no stable
release branch yet.

## Reporting a vulnerability

Do not open a public issue for a vulnerability. Use GitHub's private security
advisory feature for the repository. Include affected commit, reproduction,
impact, and any suggested mitigation. If private advisories are unavailable,
contact the repository owner privately before disclosure.

## Current security boundary

The inference server binds to loopback by default. A non-loopback bind requires
`EXPERT_API_KEY`, but the built-in server provides neither TLS nor per-tenant
authorization. Put it behind a hardened reverse proxy for any remote use.

Model containers and tokenizer files are trusted local inputs but are validated
for structure, size, ABI, and checksum. The runtime does not execute model
repository code (`trust_remote_code=False`). Model download and conversion are
operator actions outside the request path.

Never expose `/metrics` or `/model-info` publicly without proxy policy; they
reveal capacity and build identity. Do not store API keys in scripts, task
arguments, Git, logs, or model manifests.
