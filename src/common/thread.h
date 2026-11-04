// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: 2013 Dolphin Emulator Project
// SPDX-FileCopyrightText: 2014 Citra Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <thread>
#include "common/common_types.h"
#include "common/polyfill_thread.h"

#ifdef _MSC_VER
#include <intrin.h>
#endif

namespace Common {

class Event {
public:
    void Set() {
        std::scoped_lock lk{mutex};
        if (!is_set) {
            is_set = true;
            condvar.notify_one();
        }
    }

    void Wait() {
        std::unique_lock lk{mutex};
        condvar.wait(lk, [&] { return is_set.load(); });
        is_set = false;
    }

    bool WaitFor(const std::chrono::nanoseconds time);

    template<class Clock, class Duration>
    bool WaitUntil(const std::chrono::time_point<Clock, Duration> time) {
        std::unique_lock lk{mutex};
        if (!condvar.wait_until(lk, time, [this] { return is_set.load(); }))
            return false;
        is_set = false;
        return true;
    }

    void Reset() {
        std::unique_lock lk{mutex};
        // no other action required, since wait loops on the predicate and any lingering signal will
        // get cleared on the first iteration
        is_set = false;
    }

    [[nodiscard]] bool IsSet() const {
        return is_set;
    }

private:
    alignas(64) std::atomic<bool> is_set{false};
    std::condition_variable condvar;
    std::mutex mutex;
};

class Barrier {
public:
    explicit Barrier(std::size_t count_) : count(count_) {}

    /// Blocks until all "count" threads have called Sync()
    bool Sync(std::stop_token token = {}) {
        std::unique_lock lk{mutex};
        const std::size_t current_generation = generation;

        if (++waiting == count) {
            generation++;
            waiting = 0;
            condvar.notify_all();
            return true;
        } else {
            condvar.wait(lk, token,
                        [this, current_generation] { return current_generation != generation; });
            return !token.stop_requested();
        }
    }

private:
    std::condition_variable_any condvar;
    std::mutex mutex;
    std::size_t count;
    std::size_t waiting = 0;
    std::size_t generation = 0; // Incremented once each time the barrier is used
};

enum class ThreadPriority : u32 {
    Low = 0,
    Normal = 1,
    High = 2,
    VeryHigh = 3,
    Critical = 4,
};

enum class ThreadPlacement : u32 {
    Default = 0,
    Background = 1,
    Efficiency = 2,
};

class FastRecursiveMutex {
public:
    FastRecursiveMutex() = default;
    ~FastRecursiveMutex() = default;

    FastRecursiveMutex(const FastRecursiveMutex&) = delete;
    FastRecursiveMutex& operator=(const FastRecursiveMutex&) = delete;

    void lock() noexcept {
        const auto me = Self();
        if (owner.load(std::memory_order_relaxed) == me) {
            ++count;
            return;
        }
        u32 expected = 0;
        if (!state.compare_exchange_strong(expected, 1, std::memory_order_acquire,
                                           std::memory_order_relaxed)) {
            LockContended();
        }
        owner.store(me, std::memory_order_relaxed);
        count = 1;
    }

    void unlock() noexcept {
        if (--count != 0) {
            return;
        }
        owner.store(0, std::memory_order_relaxed);
        if (state.exchange(0, std::memory_order_release) == 2) {
            state.notify_one();
        }
    }

    bool try_lock() noexcept {
        const auto me = Self();
        if (owner.load(std::memory_order_relaxed) == me) {
            ++count;
            return true;
        }
        u32 expected = 0;
        if (!state.compare_exchange_strong(expected, 1, std::memory_order_acquire,
                                           std::memory_order_relaxed)) {
            return false;
        }
        owner.store(me, std::memory_order_relaxed);
        count = 1;
        return true;
    }

private:
    static u64 Self() noexcept {
        static thread_local char token;
        return static_cast<u64>(reinterpret_cast<uintptr_t>(&token));
    }

    void LockContended() noexcept {
        for (u32 spins = 0; spins < 64; ++spins) {
            u32 expected = 0;
            if (state.load(std::memory_order_relaxed) == 0 &&
                state.compare_exchange_weak(expected, 1, std::memory_order_acquire,
                                            std::memory_order_relaxed)) {
                return;
            }
#ifdef _MSC_VER
            _mm_pause();
#else
            __builtin_ia32_pause();
#endif
        }
        while (state.exchange(2, std::memory_order_acquire) != 0) {
            state.wait(2, std::memory_order_relaxed);
        }
    }

    std::atomic<u32> state{0};
    std::atomic<u64> owner{0};
    u32 count{0};
};

void SetCurrentThreadPriority(ThreadPriority new_priority);
void SetCurrentThreadName(const char* name);
void SetCurrentThreadToPerformanceCores();
void SetCurrentThreadToEfficiencyCores();
void SetCurrentThreadToBackgroundWork();
void SetCurrentThreadToAllCores();

void RefreshThreadPolicies();
void PollThreadPolicies();
[[nodiscard]] size_t GetPhysicalCoreCount();

} // namespace Common
