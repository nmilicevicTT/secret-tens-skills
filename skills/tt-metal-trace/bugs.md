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

Retracted theories (do not re-chase): "allocation non-determinism" across runs.
