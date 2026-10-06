---
name: tt-metal-trace
description: Make a tt-metal model, or part of one, run under Metal trace capture/replay (ttnn.begin_trace_capture / execute_trace). Use when adding trace to a model, converting an op to take per-call values from device tensors, debugging a trace that replays wrong/hangs/corrupts, or mixing traced and untraced code. Based on the deepseek_v3_d_p traced prefill.
---

# tt-metal-trace

Metal trace = record the dispatch commands of a forward into a DRAM buffer once, then replay them with
zero host work. This is NOT Tracy and NOT ttnn graph capture.

**Replay re-executes addresses and values frozen at capture time.** No Python runs, no program factory
runs, no allocator runs, nothing is checked. Every rule below follows from that sentence.

## Load table

| Need | Read |
|---|---|
| How capture/replay, the caches and sub-device managers work internally | `internals.md` |
| Recipe for converting a per-call scalar into a device metadata tensor (op + kernel side) | `metadata-pattern.md` |
| Branch-free traced path with device masks / one-hot selection | `fixed-path.md` |
| Symptom → cause catalogue from past bugs, grouped by scenario | `bugs.md` |
| In-flight task state (local only, gitignored; may be absent) | `worklog.md` |

## Hard rules

1. **Warm before capture.** Every program the capture records must already be in the program cache.
   "A capture cannot absorb a program-cache miss." Warm with the SAME shapes, configs, memory configs,
   sub-device manager and callbacks (e.g. the no-op layer ack) as the capture.
2. **No allocation that outlives the capture inside it.** Temporaries created and freed within the
   captured region are fine (the trace owns those addresses). Anything read after replay must be a
   persistent buffer allocated BEFORE capture and written in place (`optional_output_tensor=`,
   `output_tensor=`, `ttnn.copy(src, persistent)`).
3. **No H2D writes, D2H reads, events or blocking syncs inside capture.** `ttnn.from_torch(...).to_device`,
   `ttnn.to_torch`, `ttnn.zeros` on device all break it. Inputs are copied into the persistent input
   OUTSIDE the trace.
4. **No SDM load/clear inside capture** (fatal: "Cannot reset worker state during trace capture").
   Split the capture at each swap — `models/demos/deepseek_v3_d_p/utils/sub_device_trace.py`
   (`SubDeviceTraceController`).
5. **Per-call values cannot be Python ints.** Anything that changes between calls (chunk offset, slot,
   valid length, position) must live in a persistent device tensor the kernel reads. Hash on presence +
   memory_config, never on value. See `metadata-pattern.md`.
6. **Hidden compile-time args recompile.** e.g. `ttnn.slice(t, k)` with varying int `k`. Everything else
   that varies per call must be made call-invariant (`k_chunk_size = min(k_chunk, glob)`) or moved to
   device.
7. **Python control flow is frozen.** One trace per path, split into sections replayed in order, or a
   fixed path that computes every branch and selects with device masks (workflow step 2).
8. **Nothing allocated after capture may be alive at replay** — this includes program-cache entries
   created by a first-time op after capture (kernel binary DRAM buffer + op-owned tensors like the
   reshape mapping table live in device DRAM and are never freed). See `internals.md` § caches.
9. **Release traces before closing the mesh** (MeshTraceBuffer destructor segfaults otherwise), under the
   same SDM they were captured in.

## Workflow: making a model part traceable

1. **Inventory per-call variance.** List every Python value that differs between calls of the region:
   ints passed to ops, shapes, branches, host-computed tables, lazily-allocated tensors, `if x is None`
   first-call paths, per-call `ttnn.deallocate`/re-allocate. Each is a trace blocker.
2. **Classify each blocker:**
   - varies per call and an op consumes it → metadata tensor (`metadata-pattern.md`);
   - host-computed from data → move to a device op (e.g. `moe_padding_config`) or memoize before capture;
   - lazy allocation / per-call free → persistent buffer allocated during warmup, written in place;
   - branch → decide at build time (constant per rank/config), separate traces, or a **fixed path**
     (every branch runs, device masks select; stays bit-exact): `fixed-path.md`;
   - host-side value asserts (tile-align, range) cannot run on the metadata path; move them to the code
     that packs the metadata.
3. **Decide the traced boundary.** Mixing is legal (see below). Start with the smallest region that
   removes the host cost you care about; extend later.
4. **Wire the runtime** (mirror `tt_prefill_runtime.py`): `_prepare_trace` allocates persistent input,
   metadata tensors and outputs → one warm forward → `synchronize_device`. `capture_trace()` runs eagerly
   at compile time (not lazily on first request), AFTER D2D endpoints and completion sinks exist, after a
   warm pass with the same callbacks. Per call: copy input + metadata into persistent buffers on-device →
   `controller.replay()` → read persistent outputs. Pipeline runtimes also need:
   - the per-chunk staging ops (input/metadata copies, metadata slices) warmed before capture;
   - capture after the first socket receive, so the inbound socket op is already compiled;
   - non-last ranks: a warm-up send down the pipeline before capture (else the outbound socket op
     compiles after capture). It also makes all ranks capture in parallel instead of serially;
   - warm-ack count = every layer that acks (including extra layers beyond the main stack), and
     a D2H ack FIFO large enough to hold all warm acks (`bugs.md`).
5. **Validate** (below). Only then measure perf.

## Scenarios that need extra care

Spot these early; each has a group in `bugs.md`.
- **Extra stage beside the main stack** (draft model, prediction heads, post-processing): register it
  with the trace controller, count its acks, extend the inter-rank metadata, warm its programs.
- **Traced and untraced parts in one call**: handoff ownership (borrow, never free; no free on send).
- **Pipeline-parallel ranks**: capture order, warm-up send, ack FIFO size, first-request TTFT.
- **Eager code between capture and replay** (reference runs in tests, eager fallbacks): module-level
  keepalives and first-time compiles survive into the replay.
- **Data-dependent kernel choice** (token-count thresholds): eager differs too; not trace.

## Mixing traced and untraced code

Legal and used in production (e.g. a traced main forward followed by an untraced post-processing
step). Conditions:
- Untraced code runs strictly between replays, never inside a capture.
- Its programs were warmed BEFORE the first capture (rule 8). A first-time compile after capture is a
  latent corruption.
- Its temporaries are freed before the next replay.
- Handoff between traced and untraced parts only via persistent buffers allocated before capture.
- The untraced part still pays host dispatch, and needs host-side scalars (it has no metadata path).

## Validation method

1. **Bit-exact, not PCC.** Traced vs untraced on the same inputs must match exactly
   (run both on the same inputs, assert `torch.equal`). A stale buffer (previous call's data) still
   correlates highly; PCC hides it.
2. **Multi-call.** Always ≥2 calls/chunks with DIFFERENT metadata. One call proves nothing: replay of
   call 0 with call 0's values is trivially correct.
3. **Ladder.** 1 layer → few layers → full model; 1 chunk → 2 → many. Each rung vs untraced reference.
   Before tracing, run eager-with-host-scalars vs eager-with-device-metadata/masks: it isolates the
   conversion from trace.
4. **Allocation tracker on** (tracks buffers allocated AFTER `end_trace_capture` while a trace exists, `trace_allocation_tracker.cpp`; with segmented capture, later segments count as "after" earlier ones): `TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_TRACEBACKS=1` set
   before `import ttnn`. `execute_trace` raises with the allocating traceback. Never set
   `TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=1` for sign-off — it hides late compiles.
   A trace-owned output allocated inside the capture (held, rewritten by each replay) is a false
   positive: wrap begin/end capture in `corruptible_allocation_scope` (`trace_compiler.py` pattern) and
   keep post-capture code outside the scope so late compiles are still caught.
5. **Control first on suspect hardware.** On a host with a marginal die, run eager vs eager before
   blaming trace for a bit-exact diff. Likewise, if eager shows the same diff, it is not trace.
6. **Metadata is consumed:** per-chunk device time must grow with chunk index (longer KV). Flat = replaying
   chunk 0.
7. **Perf:** Tracy op-to-op gaps on the worst device; traced gain shows mostly on dispatch-bound small ops.
   A fixed path runs ops that eager skipped; profile it per op (one single-core op can dominate).

## Key files

- `ttnn/cpp/ttnn/operations/trace.cpp` — Python API → begin/end/replay/release_mesh_trace.
- `tt_metal/impl/trace/dispatch.cpp` — replay command issue, post-replay worker state.
- `models/demos/deepseek_v3_d_p/utils/sub_device_trace.py` — segmenting capture at SDM swaps / acks.
- `models/demos/deepseek_v3_d_p/tt/tt_prefill_runtime.py` — `_prepare_trace`, `capture_trace`,
  `_metadata_from_msg`, traced `prefill_chunk`, `release_trace`.
- `models/demos/common/prefill/runners/prefill_runner.py` — `PREFILL_USE_TRACE`, region size, asserts.
- `models/common/llm_runtime/trace_compiler.py` — generic multi-trace capture/replay manager (cq0, no SDM).
- `tech_reports/AdvancedPerformanceOptimizationsForModels/TraceCorrectness.md` — official constraints.
- `tests/ttnn/unit_tests/base_functionality/test_single_device_trace.py` — allocation-tracker semantics.

## Maintenance

Governed by the skill-evolution rule of this plugin (`rules/skill-evolution.md`): record root-caused
bugs, retractions and user corrections here as they happen, keep `worklog.md` for task state only.
Verify file:line references against HEAD before relying on them.
