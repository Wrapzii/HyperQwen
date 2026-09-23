# Adjacent Intel/B60 lane

This directory is the home for Intel-only work. HyperQwen's existing CUDA Dockerfile, Compose profiles, patch series, and source are the reference lane and retain their current behavior. `PORT.md` records the frozen B60 server and promotion gates; `FIRST_CHANGE.md` defines the first kernel experiment.

Planned layout as components land:

| Path | Purpose |
| --- | --- |
| `intel/kernels/` | Native Xe2 kernels and explicit ABI/format definitions |
| `intel/tests/` | Numerical, causal, boundary, poisoned-scratch, and repeated-launch checks |
| `intel/bench/` | Matched short/long Stage1-plus-merge and end-to-end harnesses |
| `intel/patches/` | An ordered, XPU-version-pinned vLLM patch series, only when integration begins |
| `intel/docker/` | Intel-specific build/serve path, not a replacement for the CUDA image |

These paths are a source boundary, not a claim that an Intel build already exists. Do not copy the production overlay into this directory as a new source of truth. Port individual behavior with tests and provenance. Keep model weights and benchmark output outside Git.

## Starting an agent task

From a clean fork checkout, use a separate worktree and a task-specific branch. For example:

```bash
git fetch origin
git worktree add ../hyperqwen_intel_split_schedule -b intel/split-schedule origin/intel/split-kv
cd ../hyperqwen_intel_split_schedule
git status --short --branch
```

Choose a different branch and directory for each agent. Commit the implementation and its verification, push to `origin`, and open review against the fork's `intel/split-kv` branch. The repository's existing patch-integrity workflow tests the CUDA series on pull requests; it does not run a B60 kernel, so attach the Intel test/benchmark results explicitly.
