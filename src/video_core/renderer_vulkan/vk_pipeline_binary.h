// SPDX-FileCopyrightText: Copyright 2025 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <filesystem>
#include <span>
#include <unordered_map>
#include <vector>

#include "common/common_types.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {

/// Compiled pipeline blob captured through VK_KHR_pipeline_binary, keyed by pipeline cache key.
struct PipelineBinaryEntry {
    VkPipelineBinaryKeyKHR key{};
    std::vector<u8> data;
};

/// Persistent store of pipeline binaries. Invalidated as a whole when the driver changes.
class PipelineBinaryStore {
public:
    void Insert(u64 hash, const VkPipelineBinaryKeyKHR& key, std::span<const u8> data);
    [[nodiscard]] const PipelineBinaryEntry* Find(u64 hash) const;

    void Load(const std::filesystem::path& filename, u64 driver_signature);
    void Save(const std::filesystem::path& filename, u64 driver_signature) const;

    [[nodiscard]] size_t Size() const noexcept {
        return entries.size();
    }

private:
    std::unordered_map<u64, PipelineBinaryEntry> entries;
};

} // namespace Vulkan
