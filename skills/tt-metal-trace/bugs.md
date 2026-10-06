# Trace bug catalogue (symptom → cause → fix)

Grouped by where it shows up. Scan the group matching your setup first.

## Capture and program cache

| Symptom | Cause | Fix |
|---|---|---|
| Illegal writes / program-cache miss FATAL during capture (576-608 writes) | SDM load/clear cleared program cache (#47921) + host-computed MoE padding config recomputed | Revert (#48499); memoize padding config, later move on-device (`moe_padding_config`). Debug trick: temporarily turn the TT_FATAL into a log to list all offenders in one run |
| Recompile per chunk | Chunk-varying value hashed (e.g. k_chunk_size) | Make it chunk-invariant: `min(k_chunk, glob)` |
| Corruption after first request | First-call `from_torch` (e.g. a lazily built ring buffer) after capture | Pre-warm before capture |
| Unit test bit-exact, production run corrupts or trips the tracker | Test runs an eager reference pass first, which warms programs the production compile path never warms (e.g. an untraced post-step after replay) | Warm explicitly in the runtime's prepare step; confirm with the allocation tracker on the production path |
| Fatal "Cannot reset worker state during trace capture" after adding a block | New block (draft model, prediction head) not registered with the trace controller, so its SDM swap runs inside the capture | Add it to the controller / SDM release lists the trunk blocks use |
| Host-callback ack inside the captured region breaks capture | Callback path calls `synchronize_device` | Route the ack via the controller's layer ack (D2H path), as the trunk layers do |

## Per-call values

| Symptom | Cause | Fix |
|---|---|---|
| Per-layer KV all reads layer 0 | Slot from metadata not offset by layer | slot = meta[0]*N + layer, layer hashed |
| Some layers correct, rope/last-layer wrong | Gate / `kv_only` last-layer path still took scalar overload | Route every call site to the metadata overload |
| Value off / zero | Metadata read at nonzero CB/dst offset | Read at offset 0 |
| Garbage from tiny all-gather CB | Fabric traffic clobbered a 32B scratch CB | Use output CB as scratch |
| PCC 0.65 instead of 0.93, only under trace | Stale L1 cache line from previous call | `invalidate_l1_cache()` after barrier |
| Queries roped at position 0 | Host int position frozen | Position from metadata tensor |
| Wrong slot with `cache_user_id=0` | Slot from host arg frozen at capture | Slot from metadata |
| Downstream rank uses stale or default values for an extra stage | Inter-rank metadata message carries only the trunk's words; the extra stage needs more (e.g. levels provided) | Extend the message; size the warm-up record to match |

## Buffer ownership

| Symptom | Cause | Fix |
|---|---|---|
| Traced→untraced handoff tensor overwritten | Handoff tensor allocated after capture sat on trace scratch | Persistent buffer allocated before capture; clone after replay |
| Persistent input corrupted or freed after the first replay | Consumer inside the capture frees the tensor it imports (owned semantics) | Borrow: read the persistent buffer, never free it inside the capture |
| Next chunk reads garbage on the receiving rank | Sender deallocates the tensor it sends, which under trace is the persistent output | Do not deallocate on send when traced |
| "1 device buffer still alive before replay" | An eager forward between capture and replay left a module-level keepalive buffer (e.g. `shared_rs_input_keepalive`) holding until the next forward | Sync and drop the keepalive before replay; clear it after `end_capture` |
| "N device buffers alive before replay": staging copies, metadata slices, socket ops | Per-call staging, inbound socket and outbound send ops first compile after capture | Warm staging in prepare; capture after the first receive; warm-up send on non-last ranks |
| Tracker raises on the trace output | Output allocated inside the capture is trace-owned (false positive) | `corruptible_allocation_scope` around the capture only |
| Segfault at shutdown | Trace not released before mesh close | `release_trace()` first, under capturing SDM |

## Pipeline and acks

| Symptom | Cause | Fix |
|---|---|---|
| Rank-1 KV garbage (pipeline) | Captured before D2D endpoints existed | `capture_trace()` after D2D + completion sink setup |
| Capture misses on ack path | Ack/callback path not warmed | Warm pass with a no-op ack before capture |
| Hang in the warm-ack drain after capture (e.g. 68 of 82 records read) | D2H ack records are 64 B (12 B aligned to PCIe); the default 4 KB FIFO holds 64. The warm pass fires every ack with no host reader, and the reader waits for an exact count, so overshoot stalls forever. Hits any rank with more than ~64 acks | Size the FIFO to pow2 >= acks * 64 B (`LAYER_ACK_RECORD_BYTES`, #59286) |
| Warm-ack drain hangs or leaves acks behind | Warm-ack count omits acking layers beyond the main stack (draft layers, prediction heads) | Count every acking layer (`layer_ack_layers(0, num_layers)`) |

## Perf and startup

| Symptom | Cause | Fix |
|---|---|---|
| First request very slow | Lazy capture on first request | Capture eagerly in compile |
| First request ~15 s slower (TTFT) | Ranks captured serially as chunk 0 arrived, inside the measured request | Warm-up send reaches all ranks first, so they capture in parallel at setup |
| Chunk 0 compute 10x+ the steady chunk and first send ~300 ms, later chunks normal | Programs first compiled after capture (suspected, not root-caused) | Same warm-up fixes as above; check with the allocation tracker |
| Traced fixed path much slower than the eager path (~8x) | The fixed path runs an op every call that eager skipped; `ttnn.argmax` on TILE, dim=-1 is single-core (~100 ms over a large vocab) | `to_layout(ROW_MAJOR)` before argmax (~0.8 ms) |
| `trace_bytes` prints 0.00 MB after a good capture | Reads the active SDM's memory view after capture (reporting artifact) | Trust replay of chunks >= 1 matching eager |
| Short-context throughput differs 20-30% between runs, long context flat | Single-sample noise at short context | Compare several samples before claiming a short-context regression |

Retracted theories (do not re-chase):
- "allocation non-determinism" across runs.

## Not trace (eager shows it too)

- Single-shot vs multi-turn KV not bit-exact when chunk contents differ: count-dependent MoE kernel
  choice (token-count threshold selects a different kernel), ~1 bf8 ulp between kernels. Padding and
  neighbouring rows change per-expert counts. Check: disable the threshold, rows before the change match.
- Bit-exact diffs on a host with a marginal die (non-deterministic matmul). Control with eager vs eager.
- Prediction-head KV rows at a turn boundary differ between single-shot and resumed runs: the lookahead
  rows were generated, not real, and are not recomputed when the resume point equals the end.
