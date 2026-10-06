# Trace bug catalogue (symptom → cause → fix)

| Symptom | Cause | Fix |
|---|---|---|
| Illegal writes / program-cache miss FATAL during capture (576-608 writes) | SDM load/clear cleared program cache (#47921) + host-computed MoE padding config recomputed | Revert (#48499); memoize padding config, later move on-device (`moe_padding_config`). Debug trick: temporarily turn the TT_FATAL into a log to list all offenders in one run |
| Per-layer KV all reads layer 0 | Slot from metadata not offset by layer | slot = meta[0]*N + layer, layer hashed |
| Some layers correct, rope/last-layer wrong | Gate / `kv_only` last-layer path still took scalar overload | Route every call site to the metadata overload |
| Value off / zero | Metadata read at nonzero CB/dst offset | Read at offset 0 |
| Garbage from tiny all-gather CB | Fabric traffic clobbered a 32B scratch CB | Use output CB as scratch |
| PCC 0.65 instead of 0.93, only under trace | Stale L1 cache line from previous call | `invalidate_l1_cache()` after barrier |
| Rank-1 KV garbage (pipeline) | Captured before D2D endpoints existed | `capture_trace()` after D2D + completion sink setup |
| Capture misses on ack path | Ack/callback path not warmed | Warm pass with a no-op ack before capture |
| Queries roped at position 0 (GLM) | Host int position frozen | Position from metadata tensor |
| Wrong slot with `cache_user_id=0` | Slot from host arg frozen at capture | Slot from metadata |
| Recompile per chunk | Chunk-varying value hashed (e.g. k_chunk_size) | Make it chunk-invariant: `min(k_chunk, glob)` |
| Corruption after first request | First-call `from_torch` (e.g. `get_indexer_ring_k_buffer`) after capture | Pre-warm before capture |
| Segfault at shutdown | Trace not released before mesh close | `release_trace()` first, under capturing SDM |
| First request very slow | Lazy capture on first request | Capture eagerly in compile |
| Traced drafter partial overwritten | Handoff tensor allocated after capture sat on trace scratch | Persistent buffer before capture (`_trace_partial_in`), clone after replay |
| Hang in the warm-ack drain after capture (e.g. 68 of 82 records read) | D2H ack records are 64 B (12 B aligned to PCIe); the default 4 KB FIFO holds 64. The warm pass fires every ack with no host reader, and the reader waits for an exact count, so overshoot stalls forever. Hits any rank with > ~64 acks (GLM 78 layers), not Kimi (61) | Size the FIFO to pow2 >= acks * 64 B (`LAYER_ACK_RECORD_BYTES`, #59286) |
| Warm-ack drain hangs or leaves acks behind | Warm-ack count omits extra acking layers (drafter, MTP levels) | Count every acking layer (`layer_ack_layers(0, num_layers)`) |
| "1 device buffer still alive before replay" (shared-expert RS input) | An eager MoE forward between capture and replay holds `shared_rs_input_keepalive` until the next forward | Sync + `set_shared_rs_input_keepalive(None)` before replay; clear it after `end_capture` |
| "N device buffers alive before replay": staging copies, metadata slices, socket ops | Per-chunk staging, inbound socket and D2D send ops first compile after capture | Warm staging in `_prepare_trace`; capture after the first receive; warm-up send on non-last ranks |
| Tracker raises on the trace output | Output allocated inside the capture is trace-owned (false positive) | `corruptible_allocation_scope` around the capture only |
| Traced fixed path ~8x slower than the eager host path (776 vs 96 ms/chunk) | The fixed path runs the per-level argmax every chunk; `ttnn.argmax` on TILE, dim=-1 is single-core (~100 ms over the vocab) | `to_layout(ROW_MAJOR)` before argmax (~0.8 ms) |
| First request ~15 s slower (TTFT) | Ranks captured serially as chunk 0 arrived, inside the measured request | Warm-up send reaches all ranks first, so they capture in parallel at setup |
| `trace_bytes` prints 0.00 MB after a good capture | Reads the active SDM's memory view after capture (reporting artifact) | Trust replay of chunks >= 1 matching eager |

Retracted theories (do not re-chase):
- "allocation non-determinism" across runs.

Not trace (eager shows it too):
- Single-shot vs multi-turn KV not bit-exact when chunk contents differ: count-dependent MoE kernel
  choice (GLM-5.3 `ROUTED_EXPERT_HYBRID_TOKEN_THRESHOLD`), ~1 bf8 ulp between kernels.
- Bit-exact diffs on a host with a marginal die (non-deterministic matmul). Control with eager vs eager.
