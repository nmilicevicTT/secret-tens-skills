# Metal trace internals

## Capture

- `begin_trace_capture` puts `FDMeshCommandQueue` in bypass mode (`fd_mesh_command_queue.cpp` ~:1407):
  enqueued programs are staged on host instead of sent. `end_trace_capture` writes them into a
  `BufferType::TRACE` DRAM buffer.
- Trace storage: fixed region (`trace_region_size` at device open; FATAL if too small) or, with
  `trace_region_size=0`, dynamic top-down DRAM, validated against DRAM high-water marks of the capture
  and all live traces (capture fails rather than risk overlap).
- Capture start zeroes host dispatch state (`tt_metal/impl/trace/dispatch.cpp` ~:28-58) so recorded
  commands are position-independent of prior dispatch.

## Replay

- `execute_trace` → per sub-device go-signal mcast with `RUN_MSG_REPLAY_TRACE`, stream wait, then
  `add_prefetch_exec_buf`: the prefetcher streams trace pages from DRAM until `exec_buf_end`
  (`issue_trace_commands`, dispatch.cpp ~:81).
- After replay, host marks the worker config buffer full (`update_worker_state_post_trace_execution`,
  ~:200) → the next eager op stalls until workers drain.
- `blocking=False` returns immediately; the host must not read outputs before a sync/event.

## Caches — host vs device

| Cache | Lives in | Holds |
|---|---|---|
| JIT build cache `~/.cache/tt-metal-cache` | host disk | compiled kernel ELFs; cross-process compile skip; irrelevant to trace safety |
| Program cache | **host RAM** table: op hash → `Program` | factory output, override-runtime-args callback, shared vars |
| Buffers owned by program-cache entries | **device DRAM**, ordinary allocator | kernel binaries (`ProgramImpl::allocate_kernel_bin_buf_on_device`, `tt_metal/impl/program/program.cpp` ~:2985, allocated top-down) and op-owned tensors (e.g. reshape mapping table, `reshape_tiled_program_factory.cpp` ~:301-316) |

Launch: host looks up the Program, uploads binaries to its DRAM buffer once per device, then each launch
sends "binaries at DRAM addr X + runtime args"; the prefetcher copies X into the workers' L1 kernel-config
ring. A trace stores X. Those buffers are never freed while the entry lives → a first-time compile after
capture can land on trace scratch and be overwritten by the next replay → garbage binary (hang) or
garbage table (silent corruption).

## Allocation tracker

- `TT_METAL_TRACE_ALLOC_TRACKING=1` (+ `_TRACEBACKS=1`, `_REFERRER_DEPTH=N`, `_SKIP_PROGRAM_CACHE=1`),
  read once at `import ttnn`.
- Accounting per (SDM, trace id). A buffer allocated between capture A and capture B is unsafe for A,
  safe for B. Checked right before `execute_trace`; runs Python GC once before raising.
- `trace_allocation_tracker.acknowledge_corruptible(t)` / `corruptible_allocation_scope(device)` only
  silence the checker — the program must still overwrite before use / copy out before another replay.
- Does not check warmup completeness, cache keys, shapes or path selection.

## Sub-device managers (SDM)

- Traces live per SDM (`SubDeviceManager::trace_buffer_pool_`); begin/end/execute/release resolve under
  the ACTIVE SDM. `remove_sub_device_manager` destroys that SDM's traces.
- Load/clear inside capture is fatal (`fd_mesh_command_queue.cpp` ~:1214) → segment the capture:
  `SubDeviceTraceController._split` ends the current trace, performs the load/clear, begins a new one; the
  program is a list of (TRACE, tid) / (LOAD, id) / (CLEAR) / (ACK) steps. `replay()` re-walks it
  (non-blocking execute per segment, host swaps between, sync before each ACK, final
  `synchronize_device`); `release()` re-walks LOAD/CLEAR without ACKs.
- Past trap: #47921 made SDM load/clear clear the program cache → every segment recompiled during capture.
  Reverted in #48499.

## Multi-CQ

- Max 2 CQs; cross-CQ ordering via events. Input copy on CQ1, trace on CQ0; the op consuming the input
  stays outside the trace.
- L1-resident input trick: deallocate the output, re-allocate the input inside capture, assert the address
  equals the captured one.
