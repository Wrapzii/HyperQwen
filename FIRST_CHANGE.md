# First Intel experiment: a different split-KV schedule

Work in [`intel/`](intel/README.md) on a topic branch. The frozen GPU0 server at `:8100` is the comparison target, not the development container. Keep its kernel SHA256 and MTP6 configuration in [`PORT.md`](PORT.md).

## What is already implemented

The live Xe2 kernel **already splits KV** across about 63 workgroups per KV head at 56k and merges their partials in Stage2. Each workgroup scans roughly 14 of the 64-token pages in its segment. Inside that scan, it updates the 192 output columns not held in registers through global FP32 `Part` on every page. Thus a new kernel that merely assigns KV segments to workgroups and merges partials once duplicates the existing design.

The measured 16×Q7 + 5×Q1 burst is ~2.2 ms at 1.4k and ~39.2 ms at 56k. Incorrect-output ablations put an **upper bound** of ~11 ms at 56k on the remaining global Part update; they are not a production-speed result. The half PV matmul and repeated full-history reads remain major costs. Simply raising workgroup count to one page per chunk was slower in the earlier sweep, because merge and dispatch costs rose.

HyperQwen's CUDA [`spec-decode-attn.patch`](patches/spec-decode-attn.patch) is the design reference: a `(request, KV head, query tile, KV segment)` grid, online-softmax partials, then a merge. Its CUDA/Triton implementation and vLLM 0.28.0 patch cannot be applied to the current XPU 0.26.1 runtime. Its 3090 timings are **not** Xe2 predictions.

## Question for the first prototype

Can a different **query-row tile × KV-segment** schedule retain each tile's complete running output in registers, publish it once, and pay less in duplicate KV reads and partial merging than it saves from the global Part recurrence?

Build a standalone Xe2 kernel and small merge under `intel/` using the current K8/V4 page format as the speed-comparison input. FP16 may be used as a separate mathematical oracle, but a FP16-only prototype cannot claim a K8/V4 speed win. Preserve transient INT8-Q, native integer QK DPAS, INT4-V, the exact causal visible lengths for Q1 and Q7, and complete MTP6 output semantics. Do not copy the production overlay into the fork or edit the CUDA patch. If integration requires vLLM changes, first pin an XPU source base and add an opt-in Intel patch series against it.

Sweep query rows per tile and KV pages per segment rather than choosing one layout by intuition. Track the number of KV rereads, active workgroups, register spills, partial-buffer bytes, and Stage1 **plus merge** time. A narrow query tile can eliminate the global accumulator yet reread the entire KV history several times; a one-page segment can make merging expensive. Measure both effects. Keep the existing algorithm as the fallback in the Intel lane.

## Exit criteria before vLLM integration

- Match the frozen kernel and a high-precision oracle on Q1 and Q7, varied query widths, partial/exact page and chunk boundaries, long contexts, poisoned scratch, and repeated launches with one final synchronization.
- Use the same K8/V4 bytes, prompt lengths, query counts, and warmup when comparing 1.4k, 28k, and 56k. Report separate Stage1, merge, and full-burst times; a Stage1-only win is insufficient.
- Demonstrate a material improvement in the **long-context full burst** without a short-context regression, and show the extra workspace fits the B60 memory budget at the target max length.
- Record exact source commit, build command, binary hash, test outputs, and benchmark JSON in the topic branch or a linked artifact. Never commit weights, logs, or built `.so` files.

Only after these pass should an opt-in Intel serving stack test MTP1/2/4/6, complete outputs, acceptance, prefill, decode, and coding/retrieval quality against the frozen server. Do not replace `libxe2_kv_online.so` or mount this checkout into a live container for the standalone experiment.
