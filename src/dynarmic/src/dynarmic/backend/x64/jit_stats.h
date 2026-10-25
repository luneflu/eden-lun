// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>

#include <immintrin.h>

namespace Dynarmic::Backend::X64 {

// Diagnostic counters for block compilation. Increments are compiled only under EDEN_JIT_STATS;
// the counters themselves always exist so the HUD can read them without conditional compilation.
struct JitStats {
    std::atomic<std::uint64_t> blocks{};
    std::atomic<std::uint64_t> alloc_scopes{};
    std::atomic<std::uint64_t> tsc_ticks{};
    std::atomic<std::uint64_t> invalidate_calls{};
    std::atomic<std::uint64_t> invalidate_ranges{};
};

inline JitStats& Stats() noexcept {
    static JitStats stats;
    return stats;
}

// Nanoseconds per TSC tick, calibrated once against the steady clock.
inline double TscTickNs() noexcept {
    static const double ns_per_tick = [] {
        const auto start_clock{std::chrono::steady_clock::now()};
        const std::uint64_t start_tsc{__rdtsc()};
        while (std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::steady_clock::now() - start_clock)
                   .count() < 2) {
        }
        const std::uint64_t ticks{__rdtsc() - start_tsc};
        const double elapsed_ns{static_cast<double>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() -
                                                                 start_clock)
                .count())};
        return ticks != 0 ? elapsed_ns / static_cast<double>(ticks) : 1.0;
    }();
    return ns_per_tick;
}

} // namespace Dynarmic::Backend::X64
