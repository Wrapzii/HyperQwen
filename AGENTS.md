# Intel branch agent instructions

Read `PORT.md`, `FIRST_CHANGE.md`, and `intel/README.md` before changing this branch. The task is an adjacent Intel/B60 port of HyperQwen, not a rewrite of its CUDA implementation.

- Put Intel-only source, tests, benchmarks, build files, and eventual XPU patches under `intel/`. Keep `Dockerfile`, `docker-compose.yml`, `patches/series`, existing CUDA patches, and default CUDA behavior intact. If shared wiring becomes necessary, make it minimal and opt-in, and show CUDA regression checks.
- The HyperQwen CUDA pin is vLLM 0.28.0; the frozen live B60 server uses XPU 0.26.1rc1. Never apply an existing CUDA patch to the XPU runtime by assumption. Pin and document the selected XPU source base before integrating with vLLM.
- For implementation tasks, use a separate worktree and topic branch from `origin/intel/split-kv` for each agent. Do not share a mutable checkout. Commit only your own changes; push to `origin` and target the fork's `intel/split-kv` branch for review. Never push to `upstream`, force-push the shared branch, or commit weights, logs, caches, or compiled libraries.
- `/home/dan/hyperqwen_intel_port` is not mounted into the live containers. Keep the production `:8100` K8/V4 server and `:8101` FP8 reference unchanged during standalone work. Do not replace the live `.so` to test a prototype.
- Validate the full Stage1-plus-merge result, causal Q1/Q7 behavior, long-context slope, workspace size, and complete generated outputs. Report matched baseline/candidate timings, acceptance, and any failures. A microbenchmark gain alone is not a cutover.
