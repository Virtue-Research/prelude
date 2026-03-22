# Prelude Open-Source Security Review

Date: March 11, 2026

## Executive Summary

This review covered the Prelude workspace before open source publication, with emphasis on server exposure, authentication boundaries, streaming behavior, model-loading safety, and release hygiene around native artifacts and third-party downloads.

I confirmed 8 findings:

- 3 High
- 4 Medium
- 1 Low

Pre-open-source blockers:

1. Authentication bypass on classification via the non-`/v1` route.
2. Local model index path escape leading to arbitrary file mmap outside the model directory.
3. Streaming DoS / orphaned generation risk from unbounded buffering and no disconnect cancellation.

The remaining findings are confirmed hardening and release-hygiene issues. They are lower priority than the three blockers above, but they should still be addressed before a public launch if you want a safer default posture.

## Methodology

Manual review focused on:

- HTTP routing, auth, and default exposure in the server crate
- Streaming/SSE control flow and resource cleanup behavior
- Local and remote model loading paths
- Build/release handling of native shared objects
- Browser exposure and deployment defaults

Tooling used:

```bash
trivy fs --quiet --scanners vuln,secret,misconfig --severity HIGH,CRITICAL --format table .
```

Observed limitations:

- I did not do a deep line-by-line audit of the vendored CUDA/C++ kernels.
- I did not fuzz model parsers or test exploitability dynamically.
- I reviewed additional unsafe/SIMD paths, but I did not include the RoPE/SIMD concern as a formal finding because the current evidence was weaker than the confirmed issues below.

## Findings Table

| Severity | Title | Affected Area | Release Impact |
| --- | --- | --- | --- |
| High | Authentication bypass on classification | HTTP auth / routing | Blocker |
| High | Local model index path escape | Local model loading | Blocker |
| High | Streaming DoS / orphaned generation | SSE / generation lifecycle | Blocker |
| Medium | Unsafe-by-default server exposure | Deployment defaults | Hardening |
| Medium | Unpinned Hugging Face artifact loading | Supply chain / startup | Hardening |
| Medium | Over-large default chat token budget | Request handling / DoS | Hardening |
| Medium | Wide-open browser access | CORS policy | Hardening |
| Low | Public health endpoint fingerprinting | Recon / metadata exposure | Hardening |

## Detailed Findings

### Release Blockers

#### High: Authentication bypass on classification

Evidence:

- [`/v1/classify` route](/home/siavash/prelude/crates/prelude-server/src/lib.rs#L63)
- [`/classify` route](/home/siavash/prelude/crates/prelude-server/src/lib.rs#L64)
- [`auth_middleware()` skips auth for all non-`/v1/` paths](/home/siavash/prelude/crates/prelude-server/src/auth.rs#L22)

Why this matters:

When API keys are configured, `/v1/*` routes require auth, but the same classification handler is also exposed at `/classify`, which bypasses auth entirely because the middleware only checks `"/v1/"` paths. That leaves a compute-heavy inference endpoint reachable without credentials even on an otherwise authenticated deployment.

Exploit scenario:

An external caller repeatedly hits `/classify` instead of `/v1/classify` and consumes inference capacity without an API key.

Recommended remediation:

- Require auth for `/classify`, or remove the public non-`/v1` route entirely.
- Add a test that proves unauthenticated `/classify` fails when API keys are configured.

#### High: Local model index path escape

Evidence:

- [`find_safetensor_files()` joins shard names directly onto `model_path`](/home/siavash/prelude/crates/prelude-core/src/engine/helpers.rs#L197)
- [`parse_weight_map_filenames()` accepts filenames from `model.safetensors.index.json` as-is](/home/siavash/prelude/crates/prelude-core/src/engine/helpers.rs#L264)
- [`load_var_builder_from_filenames()` mmaps the resolved files](/home/siavash/prelude/crates/prelude-core/src/engine/helpers.rs#L235)

Why this matters:

A malicious local model bundle can place absolute paths or `../` traversal segments in `model.safetensors.index.json`. The loader currently trusts those values, resolves them outside the model directory, and then memory-maps the resulting files.

Exploit scenario:

A user downloads or is given a crafted local model directory and starts the server with `--model-path`. The bundle causes the loader to open arbitrary files on the host outside the intended model directory.

Recommended remediation:

- Reject absolute shard paths.
- Reject any shard name that escapes the model directory after canonicalization.
- Consider allowlisting only plain filenames or in-tree relative subpaths.

#### High: Streaming DoS / orphaned generation risk

Evidence:

- [`stream_sse()` uses `tokio::sync::mpsc::unbounded_channel()`](/home/siavash/prelude/crates/prelude-server/src/sse.rs#L14)
- [`stream_sse()` spawns generation without disconnect-driven cancellation](/home/siavash/prelude/crates/prelude-server/src/sse.rs#L17)
- [`generate_stream_sync()` ignores `send` failures](/home/siavash/prelude/crates/prelude-core/src/task/generate.rs#L120)
- [`stream_decode_with_blocks()` also ignores `send` failures](/home/siavash/prelude/crates/prelude-core/src/cache/paged.rs#L374)

Why this matters:

There is no bounded backpressure between token production and SSE delivery, and there is no visible cancellation path tied to client disconnect. A slow reader can force unbounded buffering, and a disconnected client can leave expensive generation running to completion.

Exploit scenario:

An attacker opens many streaming requests and either reads very slowly or disconnects after generation starts. The server continues spending memory and compute on requests that no longer have a consumer.

Recommended remediation:

- Replace the unbounded channel with a bounded one.
- Stop generation when the receiver drops or send fails.
- Wire client disconnects into cancellation for `generate_stream`.

### Confirmed Hardening Findings

#### Medium: Unsafe-by-default server exposure

Evidence:

- [`--host` defaults to `0.0.0.0`](/home/siavash/prelude/crates/prelude-server/src/main.rs#L15)
- [`--api-key` is optional](/home/siavash/prelude/crates/prelude-server/src/main.rs#L83)
- [`auth_middleware()` disables auth entirely when no keys are configured](/home/siavash/prelude/crates/prelude-server/src/auth.rs#L17)

Why this matters:

The default startup path exposes the server on all interfaces and leaves it unauthenticated unless the operator explicitly supplies an API key. That is a weak default for public release.

Recommended remediation:

- Default to `127.0.0.1`.
- Require an explicit opt-in for unauthenticated mode, or make API-keyless mode loudly non-production.

#### Medium: Unpinned Hugging Face artifact loading

Evidence:

- [`Engine::from_hf_hub_with_task()` loads model artifacts from Hugging Face at runtime](/home/siavash/prelude/crates/prelude-core/src/engine/load.rs#L51)
- [`ModelChatTemplate::from_hf_hub()` loads template/config files from Hugging Face at runtime](/home/siavash/prelude/crates/prelude-server/src/chat_template.rs#L44)

Why this matters:

Startup depends on mutable third-party artifacts with no revision or checksum pinning. That is a supply-chain risk and also leaks runtime egress behavior to an external service by default.

Recommended remediation:

- Support explicit revision pinning for remote loads.
- Prefer local, pinned assets in docs and production examples.
- Consider checksum verification for downloaded model files.

#### Medium: Over-large default chat token budget

Evidence:

- [`/v1/chat/completions` sets `max_new_tokens` to `u32::MAX` when the request omits both max-token fields](/home/siavash/prelude/crates/prelude-server/src/routes/chat_completions.rs#L247)
- [`prepare_generate_request()` only clamps that value later against context length](/home/siavash/prelude/crates/prelude-core/src/task/generate.rs#L22)

Why this matters:

An omitted max-token field does not fall back to a conservative server default. Instead it becomes effectively “generate until another limit stops you,” which increases the per-request resource ceiling and weakens DoS resistance.

Recommended remediation:

- Apply a conservative server-side default max generation length before request execution.
- Enforce an upper bound independent of model context length.

#### Medium: Wide-open browser access

Evidence:

- [`CorsLayer::permissive()` is applied globally](/home/siavash/prelude/crates/prelude-server/src/lib.rs#L84)

Why this matters:

This allows any origin, method, and header to call the API from a browser. That materially enlarges abuse surface for the default-open server posture and makes accidental client-side API-key exposure more damaging.

Recommended remediation:

- Replace `CorsLayer::permissive()` with an explicit allowlist.
- Tie browser access defaults to an explicit deployment mode.

#### Low: Public health endpoint fingerprinting

Evidence:

- [`/health` returns model ID and uptime](/home/siavash/prelude/crates/prelude-server/src/routes/health.rs#L8)

Why this matters:

The endpoint is intentionally public and exposes deployment metadata that helps fingerprint a running instance. This is a low-severity reconnaissance issue, not a direct compromise path.

Recommended remediation:

- Keep `/health` minimal in public deployments.
- Consider a shallow readiness response that omits model identity and uptime.

## Scan Notes

No high/critical dependency or secret findings detected by Trivy in scanned lockfiles/worktree.

At the configured severity threshold, the Trivy filesystem scan reported zero dependency vulnerabilities in the scanned Cargo lockfiles and did not report any high/critical secret findings. The security concerns above are therefore first-party code and release-hygiene issues rather than known vulnerable dependencies surfaced by that scan.
