// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <fstream>

#include "common/cityhash.h"
#include "common/fs/fs.h"
#include "common/fs/path_util.h"
#include "common/logging.h"
#include "common/zstd_compression.h"
#include "video_core/renderer_vulkan/vk_spirv_cache.h"

namespace Vulkan {

namespace {

constexpr std::array<char, 8> MAGIC_NUMBER{'e', 'd', 'e', 'n', 's', 'p', 'v', '1'};
constexpr u32 SPIRV_CACHE_VERSION = 1;
// Keeps one title's store in the tens of MB; older entries simply stop being added.
constexpr size_t MAX_TOTAL_WORDS = 64ULL * 1024 * 1024 / sizeof(u32);
constexpr u32 NO_BLOB = 0xFFFFFFFFU;

u64 RecordKey(u64 key, u32 stage) {
    return key ^ (static_cast<u64>(stage) * 0x9E3779B97F4A7C15ULL);
}

} // namespace

SpirvCache::SpirvCache(std::filesystem::path filename_) : filename{std::move(filename_)} {
    if (!std::filesystem::exists(filename)) {
        return;
    }
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        return;
    }
    const auto end{file.tellg()};
    file.seekg(0, std::ios::beg);
    std::array<char, 8> magic{};
    u32 raw_size{};
    file.read(magic.data(), magic.size());
    file.read(reinterpret_cast<char*>(&raw_size), sizeof(raw_size));
    if (magic != MAGIC_NUMBER || raw_size == 0 || raw_size > (1ULL << 30)) {
        return;
    }
    const auto compressed_size{static_cast<size_t>(end) - magic.size() - sizeof(raw_size)};
    std::vector<u8> compressed(compressed_size);
    file.read(reinterpret_cast<char*>(compressed.data()), compressed.size());
    if (!file) {
        return;
    }
    const std::vector<u8> raw{Common::Compression::DecompressDataZSTD(compressed)};
    if (raw.size() != raw_size) {
        LOG_WARNING(Render_Vulkan, "SPIR-V cache '{}' is truncated, ignoring",
                    Common::FS::PathToUTF8String(filename));
        return;
    }

    size_t offset{};
    const auto read{[&](void* out, size_t size) {
        if (offset + size > raw.size()) {
            return false;
        }
        std::memcpy(out, raw.data() + offset, size);
        offset += size;
        return true;
    }};
    u32 version{};
    u32 blob_count{};
    u32 record_count{};
    if (!read(&version, sizeof(version)) || version != SPIRV_CACHE_VERSION ||
        !read(&blob_count, sizeof(blob_count)) || !read(&record_count, sizeof(record_count))) {
        return;
    }
    blobs.reserve(blob_count);
    for (u32 i = 0; i < blob_count; ++i) {
        u32 words{};
        if (!read(&words, sizeof(words)) || words > (MAX_TOTAL_WORDS)) {
            return;
        }
        std::vector<u32> blob(words);
        if (words != 0 && !read(blob.data(), words * sizeof(u32))) {
            return;
        }
        blobs.push_back(std::move(blob));
        total_words += words;
    }
    records.reserve(record_count);
    for (u32 i = 0; i < record_count; ++i) {
        Record record{};
        if (!read(&record.key, sizeof(record.key)) || !read(&record.stage, sizeof(record.stage)) ||
            !read(&record.blob, sizeof(record.blob)) || record.blob >= blobs.size()) {
            return;
        }
        record_index.emplace(RecordKey(record.key, record.stage),
                             static_cast<u32>(records.size()));
        records.push_back(record);
    }
    for (u32 i = 0; i < blobs.size(); ++i) {
        blob_index.emplace(BlobHash(blobs[i]), i);
    }
    LOG_INFO(Render_Vulkan, "Loaded {} cached SPIR-V programs ({} unique, {} MB)", records.size(),
             blobs.size(), total_words * sizeof(u32) / (1024 * 1024));
}

SpirvCache::~SpirvCache() = default;

u64 SpirvCache::BlobHash(std::span<const u32> code) const {
    return Common::CityHash64(reinterpret_cast<const char*>(code.data()),
                              code.size() * sizeof(u32));
}

bool SpirvCache::TryGet(u64 key, u32 stage, std::vector<u32>& code) const {
    std::scoped_lock lock{mutex};
    const auto it{record_index.find(RecordKey(key, stage))};
    if (it == record_index.end()) {
        return false;
    }
    const Record& record{records[it->second]};
    if (record.blob >= blobs.size()) {
        return false;
    }
    code = blobs[record.blob];
    return true;
}

void SpirvCache::Insert(u64 key, u32 stage, std::span<const u32> code) {
    if (code.empty()) {
        return;
    }
    std::scoped_lock lock{mutex};
    const u64 hash{BlobHash(code)};
    u32 blob{NO_BLOB};
    if (const auto it{blob_index.find(hash)}; it != blob_index.end()) {
        blob = it->second;
    } else {
        if (total_words + code.size() > MAX_TOTAL_WORDS) {
            return;
        }
        blob = static_cast<u32>(blobs.size());
        blobs.emplace_back(code.begin(), code.end());
        blob_index.emplace(hash, blob);
        total_words += code.size();
    }
    const u64 record_key{RecordKey(key, stage)};
    if (const auto it{record_index.find(record_key)}; it != record_index.end()) {
        records[it->second].blob = blob;
    } else {
        record_index.emplace(record_key, static_cast<u32>(records.size()));
        records.push_back(Record{key, stage, blob});
    }
    dirty = true;
}

void SpirvCache::Save() {
    std::vector<u8> raw;
    {
        std::scoped_lock lock{mutex};
        if (records.empty()) {
            return;
        }
        raw.reserve(total_words * sizeof(u32) + records.size() * sizeof(Record) + 64);
        const auto write{[&raw](const void* data, size_t size) {
            const auto* bytes = static_cast<const u8*>(data);
            raw.insert(raw.end(), bytes, bytes + size);
        }};
        const u32 version{SPIRV_CACHE_VERSION};
        const u32 blob_count{static_cast<u32>(blobs.size())};
        const u32 record_count{static_cast<u32>(records.size())};
        write(&version, sizeof(version));
        write(&blob_count, sizeof(blob_count));
        write(&record_count, sizeof(record_count));
        for (const auto& blob : blobs) {
            const u32 words{static_cast<u32>(blob.size())};
            write(&words, sizeof(words));
            write(blob.data(), blob.size() * sizeof(u32));
        }
        for (const Record& record : records) {
            write(&record.key, sizeof(record.key));
            write(&record.stage, sizeof(record.stage));
            write(&record.blob, sizeof(record.blob));
        }
        dirty = false;
    }

    const std::vector<u8> compressed{Common::Compression::CompressDataZSTDDefault(raw.data(),
                                                                                 raw.size())};
    const std::filesystem::path tmp{filename.string() + ".tmp"};
    {
        std::ofstream file(tmp, std::ios::binary | std::ios::trunc);
        if (!file.is_open()) {
            LOG_ERROR(Render_Vulkan, "Failed to open SPIR-V cache '{}'",
                      Common::FS::PathToUTF8String(tmp));
            return;
        }
        file.write(MAGIC_NUMBER.data(), MAGIC_NUMBER.size());
        const u32 raw_size{static_cast<u32>(raw.size())};
        file.write(reinterpret_cast<const char*>(&raw_size), sizeof(raw_size));
        file.write(reinterpret_cast<const char*>(compressed.data()), compressed.size());
    }
    std::error_code ec;
    std::filesystem::rename(tmp, filename, ec);
    if (ec) {
        // Rename across filesystems fails; fall back to overwrite.
        std::filesystem::remove(filename, ec);
        std::filesystem::rename(tmp, filename, ec);
    }
    if (ec) {
        LOG_ERROR(Render_Vulkan, "Failed to store SPIR-V cache '{}': {}",
                  Common::FS::PathToUTF8String(filename), ec.message());
        return;
    }
    LOG_INFO(Render_Vulkan, "Saved {} cached SPIR-V programs ({} MB compressed {:.1f} MB)",
             records.size(), raw.size() / (1024 * 1024),
             static_cast<double>(compressed.size()) / (1024.0 * 1024.0));
}

bool SpirvCache::Dirty() const {
    std::scoped_lock lock{mutex};
    return dirty;
}

void SpirvCache::ClearDirty() {
    std::scoped_lock lock{mutex};
    dirty = false;
}

} // namespace Vulkan
