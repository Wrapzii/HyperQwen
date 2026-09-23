# Intel B60 port: starting point and guardrails

This is the Wrapzii fork of [syv-ai/HyperQwen](https://github.com/syv-ai/HyperQwen). Intel work belongs on `intel/split-kv` and its topic branches, in an adjacent [`intel/`](intel/README.md) lane. Keep HyperQwen's CUDA source, existing `patches/series`, Dockerfile, Compose defaults, and benchmark claims working as they are. `patches/spec-decode-attn.patch` is an **algorithm reference**, not a patch to apply to the Intel runtime.

## Two distinct baselines

| | HyperQwen upstream lane | Frozen B60 reference |
| --- | --- | --- |
| vLLM | CUDA `0.28.0`, pinned in `docker/requirements.txt` | XPU `0.26.1rc1.dev457+gc810e5ee9` |
| Attention | CUDA/Triton split-KV verify patch in `patches/spec-decode-attn.patch` | native Xe2 K8/V4 online kernel |
| Serving | HyperQwen profiles and their CUDA image | `vllm-q38-int8-gpu0`, API `:8100`, MTP6 |
| Checkout | this fork | `/home/dan/xe2_test_build/ov_vllm` and `/home/dan/xe2_form_build/overlay_runtime` |

The two vLLM versions and backends are **not interchangeable**. Before integrating an Intel feature, record the exact XPU-capable vLLM source commit, torch/oneAPI versions, and whether the selected base can carry the needed HyperQwen behavior. If porting patches, use a separately documented, ordered Intel patch series against that exact base; do not feed the CUDA `patches/series` to the XPU install or silently edit exported upstream patches. Keep the first split-KV kernel standalone until this base is pinned and tested.

## Frozen live server (2026-09-22)

The HyperQwen checkout at `/home/dan/hyperqwen_intel_port` is **not mounted** into either live vLLM container. Keep it that way during development. Do not restart, remount, or retune the live server for an experiment; run a separate test process or container.

| Item | Value |
| --- | --- |
| GPU0 container / API | `vllm-q38-int8-gpu0` / `:8100` |
| Model / cache | Qwen3.8-27B GPTQ embed-int8, K8/V4 KV |
| Max context / utilization | 57,344 tokens / 0.90 |
| Speculation | MTP6, `XE2_KV_MTP_GRAPH_S1=1` |
| Online kernel | `/home/dan/xe2_form_build/overlay_runtime/libxe2_kv_online.so` |
| Kernel SHA256 | `2757c986815f7dd48d36c79b1634ecfb4d0012d56391d0ebbe2b58f3289c1dfd` |
| Weights | `/srv/models/Qwen3.8-27B-GPTQ-Int4-baked-v2-embed-int8` (read-only) |

The one-sync-per-forward GDN fix is in `/home/dan/xe2_test_build/ov_vllm/_xpu_ops.py`; preserve its behavior. GPU1 FP8 at `:8101` is a read-only correctness reference. The old overlay is a reference for interfaces and failure cases, not the source tree for the new Intel lane.

Matched full 1–80 count results for the frozen GPU0 binary, MTP6, thinking off (decode is generated tokens / measured decode time, not wall throughput):

| Prompt tokens | Decode tok/s | Prefill tok/s | Mean accepted length |
| ---: | ---: | ---: | ---: |
| 1,427 | 119.8–120.2 warm | ~1,292 warm | 6.96 |
| 28,027 | 90.1–90.2 | ~1,207 | 6.96 |
| 56,027 | 71.6 | ~1,051 | 6.96 |

At 59,527 prompt tokens, an earlier K8 binary produced an incorrect count twice while FP8 passed the identical prompt. That is outside the current 57,344 limit, but it is a required diagnostic before claiming a higher safe limit. Full-count success alone does not establish coding, reasoning, or retrieval quality.

## Work and git boundaries

Start with [`FIRST_CHANGE.md`](FIRST_CHANGE.md). Add Intel-only kernels, tests, benchmarks, build files, and eventual XPU patch series under `intel/`. A minimal opt-in hook in a shared file is acceptable only when necessary; show in the diff that CUDA behavior and default profiles are unchanged. Do not put an Intel kernel into HyperQwen's existing CUDA patch files or replace the live `.so` as a development shortcut.

Each implementation agent works in its own topic branch and worktree from `origin/intel/split-kv`, commits source plus reproducible tests/measurements, and pushes only to `origin` for review into `intel/split-kv`. Never push to `upstream`, force-push a shared branch, commit model weights/logs/build binaries, or have two agents edit the same checkout. Keep the fork's `main` as a clean upstream mirror; bring upstream changes into the Intel branch deliberately after checking the patch series.

The existing `patch integrity` workflow checks the CUDA series on pull requests against vLLM 0.28.0. It does **not** test the B60 lane. Intel changes need their own source-base check plus B60 correctness and benchmark evidence until Intel CI exists.

## Promotion gate

The first milestone is an isolated split-KV kernel that beats the frozen kernel's **attention burst** without numerical or causal failures. Later, integrate it into a separate Intel serving stack. Before considering any cutover, compare the same prompts and MTP controls against the frozen server at 1.4k, 28k, and near the configured cap; require complete outputs, stable acceptance, prefill, decode, and coding/retrieval quality. Test MTP1/2/4/6 and repeated requests, including poisoned scratch and page/chunk boundaries. A short-prompt win does not compensate for a long-context regression.

The existing K8 attention burst (16 verifier Q7 + 5 draft Q1 launches) grows from ~2.2 ms at 1.4k to ~39.2 ms at 56k. Invalid-output probes attribute ~11 ms at 56k to the remaining global Part update, but the V/PV work and full-history scans still dominate. Split-KV must be measured, not assumed to flatten that slope. The 200 decode tok/s and 1,500 prefill tok/s goals remain open.

## Remotes and locations

- `origin`: https://github.com/Wrapzii/HyperQwen (push here)
- `upstream`: https://github.com/syv-ai/HyperQwen (fetch only)
- Local checkout: `C:\Users\WhiteWidow\ai01work\hyperqwen_intel_port`
- Server checkout: `/home/dan/hyperqwen_intel_port` (not mounted into live containers)
