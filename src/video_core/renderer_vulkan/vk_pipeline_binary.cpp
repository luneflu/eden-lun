// SPDX-FileCopyrightText: Copyright 2025 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <fstream>

#include "common/fs/fs.h"
#include "common/fs/path_util.h"
#include "common/logging.h"
#include "video_core/renderer_vulkan/vk_pipeline_binary.h"

namespace Vulkan {
namespace {
constexpr std::array<char, 8> MAGIC_NUMBER{'e', 'd', 'e', 'n', 'p', 'b', 'i', 'n'};
constexpr u32 CACHE_VERSION = 1;

struct Header {
    std::array<char, 8> magic;
    u32 version;
    u64 driver_signature;
    u64 entry_count;
};
} // Anonymous namespace

void PipelineBinaryStore::Insert(u64 hash, const VkPipelineBinaryKeyKHR& key,
                                 std::span<const u8> data) {
    if (data.empty()) {
        return;
    }
    auto& entry{entries[hash]};
    entry.key = key;
    entry.key.sType = VK_STRUCTURE_TYPE_PIPELINE_BINARY_KEY_KHR;
    entry.key.pNext = nullptr;
    entry.data.assign(data.begin(), data.end());
}

const PipelineBinaryEntry* PipelineBinaryStore::Find(u64 hash) const {
    const auto it{entries.find(hash)};
    return it == entries.end() ? nullptr : &it->second;
}

void PipelineBinaryStore::Load(const std::filesystem::path& filename, u64 driver_signature) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        return;
    }
    Header header{};
    file.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!file || header.magic != MAGIC_NUMBER || header.version != CACHE_VERSION ||
        header.driver_signature != driver_signature) {
        LOG_INFO(Render_Vulkan, "Discarding stale pipeline binaries");
        return;
    }
    for (u64 i = 0; i < header.entry_count; ++i) {
        u64 hash{};
        PipelineBinaryEntry entry{};
        u32 key_size{};
        file.read(reinterpret_cast<char*>(&hash), sizeof(hash));
        file.read(reinterpret_cast<char*>(&key_size), sizeof(key_size));
        if (!file || key_size > VK_MAX_PIPELINE_BINARY_KEY_SIZE_KHR) {
            entries.clear();
            return;
        }
        entry.key.sType = VK_STRUCTURE_TYPE_PIPELINE_BINARY_KEY_KHR;
        entry.key.keySize = key_size;
        file.read(reinterpret_cast<char*>(entry.key.key), key_size);
        u64 data_size{};
        file.read(reinterpret_cast<char*>(&data_size), sizeof(data_size));
        if (!file || data_size == 0 || data_size > (64ULL << 20)) {
            entries.clear();
            return;
        }
        entry.data.resize(static_cast<size_t>(data_size));
        file.read(reinterpret_cast<char*>(entry.data.data()),
                  static_cast<std::streamsize>(data_size));
        if (!file) {
            entries.clear();
            return;
        }
        entries.emplace(hash, std::move(entry));
    }
    LOG_INFO(Render_Vulkan, "Loaded {} pipeline binaries from {}", entries.size(),
             Common::FS::PathToUTF8String(filename));
}

void PipelineBinaryStore::Save(const std::filesystem::path& filename,
                               u64 driver_signature) const {
    if (entries.empty()) {
        return;
    }
    std::ofstream file(filename, std::ios::binary | std::ios::trunc);
    if (!file.is_open()) {
        LOG_ERROR(Render_Vulkan, "Failed to open pipeline binary cache {}",
                  Common::FS::PathToUTF8String(filename));
        return;
    }
    const Header header{MAGIC_NUMBER, CACHE_VERSION, driver_signature, entries.size()};
    file.write(reinterpret_cast<const char*>(&header), sizeof(header));
    for (const auto& [hash, entry] : entries) {
        const u32 key_size{entry.key.keySize};
        const u64 data_size{entry.data.size()};
        file.write(reinterpret_cast<const char*>(&hash), sizeof(hash));
        file.write(reinterpret_cast<const char*>(&key_size), sizeof(key_size));
        file.write(reinterpret_cast<const char*>(entry.key.key), key_size);
        file.write(reinterpret_cast<const char*>(&data_size), sizeof(data_size));
        file.write(reinterpret_cast<const char*>(entry.data.data()),
                   static_cast<std::streamsize>(data_size));
    }
    LOG_INFO(Render_Vulkan, "Saved {} pipeline binaries", entries.size());
}

} // namespace Vulkan
