# Clean Intel port of HyperQwen

This repository is the Wrapzii fork of [syv-ai/HyperQwen](https://github.com/syv-ai/HyperQwen). Intel work happens on branch `intel/split-kv`. `upstream` is the original repo. Nothing here is mounted into the live K8 server.

Paused here on 2026-09-22. The live K8/V4 server is the baseline.

## Frozen server

Do not restart, remount, or edit these until a new server matches them:

| Item | Value |
| --- | --- |
| Container | `vllm-q38-int8-gpu0` |
| API | `:8100` |
| Max context | 57,344 tokens |
| Utilization | 0.90 |
| Speculation | MTP6, `XE2_KV_MTP_GRAPH_S1=1` |
| Online kernel | `/home/dan/xe2_form_build/overlay_runtime/libxe2_kv_online.so` |
| Kernel SHA256 | `2757c986815f7dd48d36c79b1634ecfb4d0012d56391d0ebbe2b58f3289c1dfd` |
| Weights | `/srv/models/Qwen3.8-27B-GPTQ-Int4-baked-v2-embed-int8` (read-only) |
| Last check | 1,427-token full count, 119.2 tok/s, acceptance 6.96 |

The one-sync-per-forward GDN fix stays in `/home/dan/xe2_test_build/ov_vllm/_xpu_ops.py`. Do not revert it as part of this port.

GPU1 FP8 `:8101` is the correctness reference. Do not relaunch it from a drifted serve script.

## What this branch is for

One change at a time, on top of HyperQwen, aimed at Intel XPU. The first experiment is split KV attention: independent partials from separate workgroups, merged once, instead of the live kernel's per-page global Part update.

That update is about 11 ms of the 39 ms attention burst at 56k. Putting it in registers, local memory, fp16, or subgroup vector loads did not remove it without getting slower. The current overlay is not the place to try the next decomposition.

## Rules

- Do not bind this checkout into `vllm-q38-int8-gpu0` or `vllm-q38-baked`.
- Do not copy `xe2_kv_vllm_overlay.py` in as the starting point.
- Do not replace `libxe2_kv_online.so`.
- Do not retune the 57,344 server to make this branch look better.
- Port one change, then measure. The next change waits until that one is understood.
- Cut over only when a new server matches the frozen one on stability and throughput: full 1–80 counts, acceptance near 6.96, and about 120 tok/s at 1,427 tokens. Longer context is not a reason to retire the frozen server early.

## Remotes

- `origin` — https://github.com/Wrapzii/HyperQwen
- `upstream` — https://github.com/syv-ai/HyperQwen
- Branch — `intel/split-kv`
