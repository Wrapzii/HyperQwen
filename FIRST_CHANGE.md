# First change: split-KV verify attention

Do not apply this on `:8100`. The frozen server stays at 57,344 tokens until a separate server matches it.

HyperQwen already has the decomposition we need, in `patches/spec-decode-attn.patch`. FlashAttention-2 only splits the KV sequence across thread blocks when a request has one query token. An MTP verify step has several query tokens, so a 24-head model gets one block per head and leaves the rest of the GPU idle. Their Triton kernel gives every request, KV head, and query tile several blocks. Each block writes an online-softmax partial. A later kernel merges those partials once.

That is the same shape as the live Xe2 problem. The online kernel keeps one running `Part` and updates it from global memory on every KV page. At 56k that update is about 11 ms of a 39 ms attention burst. Registers, local memory, fp16, and subgroup vector loads all failed to remove it. Splitting the KV sequence into independent partials, then merging once, is the change that avoids that inner-loop update.

## What lands first

One isolated kernel, not the overlay:

- Input is one request, a few query rows, and the paged K/V for that request.
- Workgroups own disjoint KV segments. They do not read each other's partials.
- Each workgroup writes one partial and its softmax statistics.
- A second, small kernel merges those partials.

No `xe2_kv_vllm_overlay.py`. No change to `libxe2_kv_online.so`. bf16 or the existing K8/V4 layout can wait; the first version only has to show that the merge-once schedule is correct and faster than a running global update on a standalone benchmark.

## What stays out of this change

The patch also wires environment flags, CUDA graphs, and a bf16-only switch inside vLLM. Those are later steps. The measured CUDA result (about 250 us versus 2,085 us per layer at 25k context and 8 query tokens, on a 3090) is a target to compare against, not a claim about Xe2.
