# Aleph Inference Runtime v2 — foundation and measurement plan

Status: **design and validation baseline only**. No new model backend or CUDA kernel has been implemented in this branch.
Owner: Aleph Technologies
Baseline: Strata fork `alephhq-tech/Strata`; upstream `Niko1221/Strata`
License: keep the upstream MIT license and all copyright notices.

## Context

Target host (to be independently confirmed with diagnostics): Ubuntu, two RTX 5090 GPUs with 32 GiB-class VRAM each, 64 GiB system RAM, NVMe storage. Existing deployment: Qwen3.8-Flash-Next IQ3_S using Strata, historically configured up to a 524,288-token context.

Problems:
1. One request is fast; concurrent requests become slow.
2. The Strata core is architecture-specific, not a generic GGUF loader.
3. Aleph needs explicit *user-selected* model IDs and predictable latency as load increases.

**Do not infer** that two 32 GiB cards form one unified 64 GiB VRAM pool. Do not infer throughput from parameter counts or a single-stream benchmark.

## Critical upstream findings

- Strata already offers opt-in concurrency via a model config `"parallel": N` or engine `--batch N`. Do not specify both independent and conflicting values; explicit engine args take precedence.
- On a layer split, recent upstream versions offer batch groups and grouped pipeline processing. Check engine logs for actual `batch_slots` and `batch_groups`; compare with `GET /v1/status` -> `concurrency.serving`.
- MTP speculation may improve single-stream speed but change batch throughput. Recent upstream has `--batch-mtp` options with important layer-split and cache interactions. Do not assume positive performance without A/B testing.
- Every simultaneous active conversation needs session and cache state; more slots may evict GPU-resident experts and cause expensive host RAM / PCIe traffic.
- The Aleph fork was 235 upstream commits behind when inspected on 2026-10-11. Update and validate on a dedicated branch before changing the core inference path; preserve the production deployment.

References:
- https://github.com/Niko1221/Strata/blob/main/docs/BATCHING.md
- https://github.com/Niko1221/Strata/blob/main/docs/MULTI_GPU.md
- https://github.com/Niko1221/Strata/blob/main/serve/test_parallel.py

## Architecture

```text
Aleph Desktop / Aleph Cloud caller
               |
      OpenAI-compatible API
  /v1/models, /v1/chat/completions
               |
      Aleph Inference Gateway
      - explicit requested model ID
      - authentication and quotas
      - per-model bounded admission queues
      - streaming and cancellation
      - request/slot/cache observability
               |
       +-------+------------------+
       |                          |
  Strata backend             Generic model backend
  Qwen3.8-Flash-Next         llama.cpp / SGLang / vLLM
  custom expert offload      models supported by each backend
       |                          |
    GPU0+GPU1             GPU0 instance + GPU1 instance
  layer/pipeline split     (or a deliberate multi-GPU split)
               |
    GPU Memory / CPU RAM / NVMe accounting
```

Key invariants:
- Model routing only by the model ID chosen by the user; no silent replacement with another model.
- Each backend advertises its true capabilities (architectures, quantization, tool calling, multimodal, context) rather than claiming generic compatibility.
- NVMe paging/offload is a capacity fallback, not a concurrency throughput optimization.
- Model weights, CUDA kernels, sessions and caches remain separate for engines with incompatible layouts.
- Data isolation, authentication and bounded per-tenant queue lengths must be implemented before exposing the gateway publicly.
- Bind dev servers to localhost or private VPN only. Never expose unprotected Strata/llama.cpp ports to the internet.

## Recommended experimental modes

### A. Existing large model: two-card Strata split

1. Preserve existing config and engine binary.
2. Record cold/warm single-request metrics with `parallel=1`.
3. On a separate deployment or staging run, reduce the configured maximum context to **32K or 64K** as a baseline; keep the same model weights, dataset and quant.
4. Test `parallel=2`, then `parallel=4`, with actual serving slots confirmed from logs/status. Cap queue size.
5. Compare `batch-groups` pipeline settings, batch MTP on/off only if supported by the actual build, and GPU expert hit/miss rates.
6. Test 128K and larger contexts incrementally, not automatically at 524K, while monitoring RAM, KV allocations and context correctness.
7. Validate streaming, cancel, reasoning output, vision inputs and tool calling after each change.

### B. Two independent GPU backends

For small enough quantized models, keep GPU0 and GPU1 independent to avoid multi-GPU transfer and maximize simultaneous work.

Examples **to validate by benchmarks**:
- GPU0: Qwen3.8-27B UD-Q4_K_M, approximately 16.5 GB GGUF weights, with its own llama.cpp batch slots.
- GPU1: Qwen3.5-35B-A3B Q4, roughly 22 GB GGUF weights, with its own slots.
- Alternative: two Qwen3.8-27B replicas and deterministic per-request balancing within the *same* user-selected model.

These are weights-only estimates; leave room for KV caches, model vision encoders, graph buffers and CUDA/runtime allocations. Qwen3.8-Flash-Next currently cannot co-reside across both GPUs if it has reserved their VRAM.

Example llama.cpp settings: `--ctx-size 32768 --parallel 2 --cont-batching -ngl all`. Run with an updated llama.cpp build supporting the model's architecture. Increase slots only when confirmed to fit.

## Model support plan

| Backend | Candidate model | First target |
|---|---|---|
| Strata | Qwen3.8-Flash-Next IQ3_S | Maintain, optimize multi-request |
| llama.cpp / suitable GGUF loader | Qwen3.8-27B Q4/Q6 | One GPU, test 2/4 slots |
| llama.cpp / suitable loader | Qwen3.5-35B-A3B Q4 | One GPU, test throughput |
| compatible vLLM / SGLang | Qwen3.8-27B NVFP4 | Validate RTX 5090-specific speed and API semantics |
| suitable multi-GPU engine | Qwen3.5-122B-A10B low-bit | Research-only until measured |

The larger models are not new built-in Strata kernel support. Introducing them requires backend adapters or new architecture-specific kernels.

## Benchmark methodology

Collect independent metrics at simultaneous request levels **1, 2, 4, 8**, with *both* short and long prompts, first cold then warm, at fixed output token limits.

Record:
- request queue wait vs prefill time vs time to first token (TTFT)
- TTFT p50/p95, end-to-end latency p50/p95
- aggregate generated tokens/second; per-stream decode rate
- GPU memory, GPU utilization, GPU power, GPU PCIe links
- CPU utilization, RAM pressure/swap, disk I/O
- GPU-resident expert hit rate and offloaded/PCIe expert rate
- active slots, pending queue size, canceled requests, OOM and errors

Compare on **identical hardware**, kernel/driver versions and prompts. Warm cache state must be reproducible. Do not substitute total throughput for request latency. Do not claim `n` concurrent requests produce `n` times the single-request throughput.

Minimum passing checks:
- Streaming begins without waiting for all other requests to finish.
- Backpressure rejects/queues above known memory-safe capacity, rather than OOM.
- Cancellation releases a slot and does not corrupt other conversations.
- Output/tool-call correctness and vision behavior do not regress.
- Single-user baseline remains within an agreed tolerance.
- Benchmark results have reproducible configs, SHA, logs and raw metrics.

## Development stages

1. **Instrumentation / diagnosis:** gather sanitized host info and benchmark Strata's existing `parallel` settings.
2. **Upstream integration:** reconcile 235 missing commits on a non-production branch and run the test suite, especially `serve/test_parallel.py` and `tools/batch_test.py`.
3. **Gateway adapter interface:** `ModelRegistry`, `BackendAdapter`, `AdmissionController`, `GpuResourceManager`, `MetricsExporter`. Explicit model IDs, cancellation and SSE.
4. **Generic backend:** connect llama.cpp (first), then benchmark SGLang/vLLM with supported models.
5. **Scheduler:** bounded per-model/tenant queues, fairness, prompt preemption, KV budget, telemetry-driven admission. Do not schedule based only on GPU utilization.
6. **Kernel optimization:** profile (Nsight Systems + Nsight Compute) before changing expert routing or GPU kernels; A/B test `batch-mtp` and pipelining in Strata first.
7. **Acceptance:** correctness tests, fault injection, 1/2/4/8 load tests, and production rollback path.

## Safe quick-start diagnostics

```bash
# Always inspect the current setup before changing flags.
curl -fsS http://127.0.0.1:8080/v1/status | jq '.concurrency'
curl -fsS http://127.0.0.1:8080/metrics | head -n 60
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu,utilization.memory --format=csv
iostat -xz 1 3
```

Never upload `.env`, API tokens, user conversations or model auth credentials in an issue or performance report.

## Decision required after the host report

Choose either **large-model quality at modest concurrency** (Strata split) or **high concurrency of smaller models** (two GPU-isolated instances), or offer both as explicit deployment profiles. No source-level bottleneck diagnosis is complete until real GPU, RAM, PCIe and concurrent request traces are available.
