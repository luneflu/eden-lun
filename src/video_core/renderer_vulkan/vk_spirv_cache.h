// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstddef>
#include <filesystem>
#include <mutex>
#include <span>
#include <unordered_map>
#include <vector>

#include "common/common_types.h"

namespace Vulkan {

// On-disk cache of translated SPIR-V, keyed on the shader hash, the environment answers and the
// pipeline key. SPIR-V is device agnostic, so records are reusable across GPUs; identical code
// from different keys is stored once.
//
// The file is also the shareable shader pack: it carries a snapshot of the pipeline key list, so
// another machine can rebuild every covered pipeline for its own driver at boot.
class SpirvCache {
public:
    explicit SpirvCache(std::filesystem::path filename);
    ~SpirvCache();

    SpirvCache(const SpirvCache&) = delete;
    SpirvCache& operator=(const SpirvCache&) = delete;

    // Returns false on a miss; on a hit `code` holds the cached words.
    bool TryGet(u64 key, u32 stage, std::vector<u32>& code) const;

    // Stores `code` for (key, stage). Safe to call from any thread.
    void Insert(u64 key, u32 stage, std::span<const u32> code);

    // Unions records from another cache or shared pack into this one; local records win.
    void Merge(const std::filesystem::path& other);

    // Writes the cache atomically (temp file + rename).
    void Save();

    [[nodiscard]] bool Dirty() const;
    void ClearDirty();

    // Restores the embedded pipeline key list when the local one is gone.
    bool RestoreKeys(const std::filesystem::path& path) const;

private:
    struct Record {
        u64 key{};
        u32 stage{};
        u32 blob{};
    };

    void MergeSpirv(const std::filesystem::path& other);
    void Parse(std::span<const u8> bytes);

    [[nodiscard]] u64 BlobHash(std::span<const u32> code) const;

    std::filesystem::path filename;
    mutable std::mutex mutex;
    std::vector<std::vector<u32>> blobs;
    std::unordered_map<u64, u32> blob_index;
    std::vector<Record> records;
    std::unordered_map<u64, u32> record_index; // (key << 8 | stage) -> record index
    std::vector<u8> keys;                      // snapshot of vulkan.bin, may be empty
    size_t total_words{};
    bool dirty{};
};

} // namespace Vulkan
