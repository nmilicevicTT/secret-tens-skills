# Metadata-tensor pattern (per-call scalars on device)

Reference PRs: #48903 rotary, #48905 zero_pad, #48906 update_padded, #48907 ring_mla, #51624 runtime,
#55085 GLM indexer ops, #56988 valid_end.

## Host side

1. One persistent 1-element uint32 tensor per scalar, replicated, DRAM, allocated once before capture
   (`ChunkMetadata` in `tt_prefill_runtime.py`: slot_id, actual_start, actual_end).
2. Per call, refresh them ON DEVICE from a packed message (`_metadata_from_msg`), or with
   `ttnn.copy` from a freshly uploaded tensor — outside the trace.
3. Derived values (slot = base*N + layer, etc.) are computed in-kernel from the raw scalars plus
   constants hashed into the program (`layer_idx`, `kv_cache_num_layers`). Constants hash; values don't.

## Op side

1. **Dual signature:** keep the scalar overload, add a metadata-tensor overload (optional tensor arg).
   Test they are bit-identical for the same values.
2. **Hash:** presence of the metadata tensor + its memory_config. Never the value.
3. **Pass the metadata buffer as a named buffer arg** (common/runtime arg bound to the buffer), not a raw
   address baked into runtime args. `// smuggled-rta-ok:` annotation exists for the pre-commit check.
4. **Kernel template** `template <bool HasMeta>` + `if constexpr` so the unused path is discarded.

## Kernel read (dataflow)

`ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/metadata_scalar_read.hpp`
(`read_metadata_scalar_u32`):

```
noc_async_read(meta_addr → cb page)   // read at CB page OFFSET 0
noc_async_read_barrier();
invalidate_l1_cache();                // else stale value from a previous call (PCC 0.65 vs 0.93)
volatile tt_l1_ptr uint32_t* p = ...; value = p[0];
```

- Nonzero byte offsets have read 0 twice (update_padded +4B, ring_mla CB offset). Read at 0.
- Compute kernels can't NoC-read: reader derives the value and publishes it via a small mailbox CB.
- Tiny scratch CBs can be clobbered by fabric traffic in CCL ops; use an output CB as scratch.

## Checklist per converted op

- [ ] Scalar and metadata overload bit-exact on ≥2 different values.
- [ ] Program hash identical across different metadata values (no recompile per call).
- [ ] Kernel invalidates L1 cache after the read.
- [ ] Used under trace for ≥2 calls with different metadata; output changes as expected.
