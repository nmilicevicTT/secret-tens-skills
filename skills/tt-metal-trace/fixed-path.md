# Fixed path: tracing a region with per-call branches

Use when the region's control flow depends on per-call values (how many levels run, whether a split
exists, which rows feed the next step), so one captured path cannot serve every call and separate
traces per branch would explode. Typical cases: multi-token prediction heads with a variable number of
provided levels, speculative/draft heads, windows that straddle a device or chunk boundary.

Idea: run every branch on every call and select results with persistent device tensors. Python sees
one path; the per-call choice lives in tensor values written before replay.

## Building blocks (all exact, so traced == eager bit for bit)

| Need | Device form |
|---|---|
| Keep or clear rows | `x * keep` with `keep` in {0, 1} |
| Patch rows from another source | `x * keep + select @ patch` (`select` zero when no patch) |
| Pick or place rows (row index varies per call) | one-hot matmul `P @ x`, `P` [rows_out, rows_in] |
| Choose between two candidates | `a * m + b * (1 - m)` |
| Pick one tile for a head (e.g. the last valid row's tile) | one-hot `[32, W] @ h`, then run the head on that tile |

- `x * 1`, `x + 0` and a one-hot matmul with HiFi4 + fp32 dest accumulation reproduce the input exactly.
  Lower fidelity or bf16 accumulation breaks bit-exactness.
- Every mask/selector is a persistent tensor allocated before capture and written in place per call.

## Geometry tensors

Group all per-call selectors in one object (keep masks, selectors, placement matrices, head row picks)
with a `write(call_values)` that fills them before replay.

- Phase 1: build on host, write with `copy_host_to_device_tensor` outside the capture. Simple, validates
  the math, costs one H2D per call.
- Phase 2: derive them on device from the metadata tensor (removes the H2D; needed when only device
  metadata exists, e.g. a serving engine).
- Host value asserts (range, alignment) move to the code that builds the geometry.

## Validation

1. eager with host scalars vs eager with geometry: bit-exact (proves the masks, no trace involved).
2. eager with geometry vs replay: bit-exact, ≥3 calls, schedules covering every branch (none provided,
   some provided, all provided; split / no split; short last call).
3. Then end-to-end PCC vs the golden reference, identical eager vs traced.

## Cost

Every branch runs every call. Profile per op: an op that eager ran only on a rare branch now runs
always, and one slow op dominates. Known trap: `ttnn.argmax` on TILE layout, dim=-1, is single-core
(~100 ms over a large vocab); `to_layout(ROW_MAJOR)` first (~1 ms). After that fix a fixed path can be
faster than the eager host path because host dispatch is gone.
