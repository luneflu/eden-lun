# Adaptive pipeline build throttle — plan

Status: plan only, no code. Branch context: `pipeline-drop-stutter` already drops unbuilt draws,
counts them, throttles background threads via EcoQoS and clamps the worker count.

## Goal

Keep the guest cores scheduled while a compile burst runs. Trade compile throughput for frame-time:
when the host is saturated *because* of pipeline builds, let fewer builds run concurrently. When it
is not, use all workers.

Explicitly **not** the goal: reduce worker count (that just halves drain rate and lengthens the
stall), reduce total compile work, or touch warm runs.

## Measured starting point

Cold run, this machine (4C/8T):

```
guest cores 3 x 0.66-0.73 core = 2.1
pipeline builders (4)          = 1.42 core
GPU thread                     = 0.52 core
                                4.04 cores demanded on 4 physical
```

Warm run: `builds=0 drops=0`, builders idle. Class A stalls there are Dynarmic block compiles and
are out of scope for this change.

Per-build cost: ~40-115 ms CPU, 97% inside `amdvlk64.dll` (SPIR-V -> ISA), inline on builder
threads. State variants on an already-compiled shader cost 0.1-4.7 ms, so the burst is per distinct
shader.

## Signal

One number, cheap, already available:

```
process_core_rate = d(GetProcessTimes(CPU)) / d(wall)      // one syscall per sample
```

Sampled 4 Hz, EMA over ~0.5 s. No new thread, no per-thread handles, works on any machine.
Saturation threshold from logical/physical core count, not a hard-coded 4.

Secondary gate: only throttle while builds are actually pending (branch already tracks
`pending_builds`), so an idle build pool cannot be throttled.

## Mechanism

Counting gate in front of each build task, limit variable at runtime:

```
limit = process_core_rate >= cores_physical - 0.5   -> 1 build at a time
        process_core_rate <= cores_physical - 1.0   -> all workers (default)
        between                                     -> keep previous (hysteresis band)
```

Builder task body becomes `EnterGate(); ...existing build...; LeaveGate();` where the gate is an
atomic `active` counter plus `Sleep(1)` when `active >= limit`. `Sleep(1)` yields the core instead
of spinning, so a gated worker costs ~0 CPU.

Placement: wrap the task lambda in `vk_graphics_pipeline.cpp` (`GraphicsPipeline` ctor, the
`func{...}` that queues `MakePipeline`) and the equivalent in `vk_compute_pipeline.cpp`. Gate state
lives beside `workers` in `PipelineCache` so both pipelines share one limit.

Worker count stays `GetTotalPipelineWorkers()` as on master (4 here). The branch's clamp to 2 is
dropped: measured steal with 4 workers is 1.42 cores because they yield; clamped to 2 they
saturate and steal up to 2.0.

Also restore `QueueWorkFront` for the frame's own pipeline: with draws dropped rather than stalled,
front-loading shortens consecutive dropped frames for what is on screen at no CPU cost.

## Knobs (A/B, no rebuild)

```
EDEN_PIPELINE_THROTTLE=0        disable the gate entirely
EDEN_PIPELINE_THROTTLE_LIMIT=N  force limit N (0 = auto)
EDEN_PIPELINE_WORKERS=N         already on the branch
```

## Instrumentation already on the branch

`PipelineRuntime builds= build_ms= frame_waits= wait_ms=`, `FrameStat ... drops=`. Gate adds one
counter: `throttled_ms`, i.e. wall time workers spent in the gate, so fill slowdown is attributable
rather than guessed.

## Test plan

Cold run each, cache dir renamed, same area, 60 s:

1. `EDEN_PIPELINE_THROTTLE=0` (baseline: today's branch behaviour)
2. auto (gate on)
3. forced limit 1 and 2, to bracket

Compare: fps mean and p1/p99 ms/frame, `drops=`, `throttled_ms`, total builds, and wall time to
reach the end of the burst. Success = p99 frame time down with the burst finishing later. Failure =
frame time unchanged and burst-only-slower, in which case revert.

Warm run sanity: gate must never engage (`pending_builds` 0), `builds=0`, no regression.

## Risks

- Oscillation between limits: handled by the hysteresis band and 0.5 s EMA.
- Total fill time up ~min(limit/workers): acceptable per requirement, must be reported.
- Process CPU includes guest cores; on a machine where guests alone saturate, the gate stays closed
  and builds are collapsed to 1 permanently. Mitigated by requiring pending builds and by
  `EDEN_PIPELINE_THROTTLE_LIMIT` override; watch `throttled_ms` vs `builds` in the report.
- Non-Windows: the gate is portable; EcoQoS part stays Windows-only as today.

## Deliberately out of scope

- Reducing distinct-shader count (spec constants do not help: driver re-hashes and recompiles per
  value set; only runtime-data variation would, which means interpreting register arrays).
- Off-thread translate/EmitSPIRV (two-phase pipeline object, 300-500 LOC, negligible GPU-thread win).
- Class A warm stalls (Dynarmic block cache, separate plan in `docs/JITBlockCachePlan.md`).
