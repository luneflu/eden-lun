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
// Shareable container: SPIR-V records plus the pipeline key list, so one file precompiles.
constexpr std::array<char, 8> PACK_MAGIC_V1{'N', 'X', 'S', 'P', 'A', 'C', 'K', '1'};
constexpr std::array<char, 8> PACK_MAGIC{'N', 'X', 'S', 'P', 'A', 'C', 'K', '2'};
// Sections are tagged by backend so a pack stays one format across renderers: everything
// outside the tag (shader hashes, guest answers, Maxwell state) is backend neutral.
constexpr std::array<char, 4> VULKAN_SECTION{'V', 'K', '0', '1'};
constexpr u32 SPIRV_CACHE_VERSION = 1;
// Keeps one title's store in the tens of MB; older entries simply stop being added.
constexpr size_t MAX_TOTAL_WORDS = 64ULL * 1024 * 1024 / sizeof(u32);
constexpr u32 NO_BLOB = 0xFFFFFFFFU;

u64 RecordKey(u64 key, u32 stage) {
    return key ^ (static_cast<u64>(stage) * 0x9E3779B97F4A7C15ULL);
}

std::vector<u8> ReadFileBytes(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    std::vector<u8> bytes;
    if (!file.is_open()) {
        return bytes;
    }
    const auto size{file.tellg()};
    if (size <= 0) {
        return bytes;
    }
    bytes.resize(static_cast<size_t>(size));
    file.seekg(0, std::ios::beg);
    file.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
    if (!file) {
        bytes.clear();
    }
    return bytes;
}

struct Section {
    std::array<char, 4> id{};
    std::vector<u8> data;
};

// Splits a pack body into its backend tagged sections; unknown ids are kept so a newer pack
// can still be read by an older build.
std::vector<Section> ReadSections(std::span<const u8> body) {
    std::vector<Section> sections;
    size_t offset{};
    const auto read{[&](void* out, size_t size) {
        if (offset + size > body.size()) {
            return false;
        }
        std::memcpy(out, body.data() + offset, size);
        offset += size;
        return true;
    }};
    while (offset < body.size()) {
        Section section;
        u32 size{};
        if (!read(section.id.data(), section.id.size()) || !read(&size, sizeof(size)) ||
            size > (1ULL << 30)) {
            sections.clear();
            return sections;
        }
        section.data.resize(size);
        if (size != 0 && !read(section.data.data(), size)) {
            sections.clear();
            return sections;
        }
        sections.push_back(std::move(section));
    }
    return sections;
}

bool SplitVulkanPayload(std::span<const u8> payload, std::vector<u8>& spirv,
                        std::vector<u8>& keys) {
    size_t offset{};
    const auto read{[&](void* out, size_t size) {
        if (offset + size > payload.size()) {
            return false;
        }
        std::memcpy(out, payload.data() + offset, size);
        offset += size;
        return true;
    }};
    u32 spirv_size{};
    if (!read(&spirv_size, sizeof(spirv_size)) || spirv_size > (1ULL << 30)) {
        return false;
    }
    spirv.resize(spirv_size);
    if (spirv_size != 0 && !read(spirv.data(), spirv_size)) {
        return false;
    }
    u32 keys_size{};
    if (!read(&keys_size, sizeof(keys_size)) || keys_size > (1ULL << 30)) {
        return false;
    }
    keys.resize(keys_size);
    return keys_size == 0 || read(keys.data(), keys_size);
}

} // namespace

SpirvCache::SpirvCache(std::filesystem::path filename_) : filename{std::move(filename_)} {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        return;
    }
    std::array<char, 8> magic{};
    file.read(magic.data(), magic.size());
    if (!file) {
        return;
    }
    file.close();
    const std::vector<u8> bytes{ReadFileBytes(filename)};
    std::vector<u8> payload;
    if (magic == PACK_MAGIC) {
        if (bytes.size() < magic.size()) {
            return;
        }
        for (Section& section : ReadSections(std::span<const u8>{bytes}.subspan(magic.size()))) {
            if (section.id == VULKAN_SECTION) {
                if (!SplitVulkanPayload(section.data, payload, keys)) {
                    return;
                }
            }
        }
    } else if (magic == PACK_MAGIC_V1) {
        // Older single backend container: SPIR-V block, then the pipeline key block.
        if (bytes.size() < magic.size() + sizeof(u32)) {
            return;
        }
        std::span<const u8> body{bytes};
        body = body.subspan(magic.size());
        u32 spirv_size{};
        std::memcpy(&spirv_size, body.data(), sizeof(spirv_size));
        if (body.size() < sizeof(spirv_size) + spirv_size + sizeof(u32)) {
            return;
        }
        payload.assign(body.begin() + sizeof(spirv_size),
                       body.begin() + sizeof(spirv_size) + spirv_size);
        u32 keys_size{};
        std::memcpy(&keys_size, body.data() + sizeof(spirv_size) + spirv_size,
                    sizeof(keys_size));
        keys.assign(body.begin() + sizeof(spirv_size) + spirv_size + sizeof(keys_size),
                    body.begin() + sizeof(spirv_size) + spirv_size + sizeof(keys_size) +
                        keys_size);
    } else {
        payload = bytes;
    }
    Parse(payload);

    if (!records.empty()) {
        LOG_INFO(Render_Vulkan, "Loaded {} cached SPIR-V programs ({} unique, {} MB)",
                 records.size(), blobs.size(), total_words * sizeof(u32) / (1024 * 1024));
    }
}

SpirvCache::~SpirvCache() = default;

void SpirvCache::Parse(std::span<const u8> bytes) {
    size_t offset{};
    const auto read{[&](void* out, size_t size) {
        if (offset + size > bytes.size()) {
            return false;
        }
        std::memcpy(out, bytes.data() + offset, size);
        offset += size;
        return true;
    }};
    std::array<char, 8> magic{};
    u32 raw_size{};
    if (!read(magic.data(), magic.size()) || magic != MAGIC_NUMBER ||
        !read(&raw_size, sizeof(raw_size)) || raw_size == 0 || raw_size > (1ULL << 30)) {
        return;
    }
    std::vector<u8> compressed(bytes.size() - offset);
    if (!compressed.empty() && !read(compressed.data(), compressed.size())) {
        return;
    }
    const std::vector<u8> raw{Common::Compression::DecompressDataZSTD(compressed)};
    if (raw.size() != raw_size) {
        LOG_WARNING(Render_Vulkan, "SPIR-V cache '{}' is truncated, ignoring",
                    Common::FS::PathToUTF8String(filename));
        return;
    }

    size_t raw_offset{};
    const auto read_raw{[&](void* out, size_t size) {
        if (raw_offset + size > raw.size()) {
            return false;
        }
        std::memcpy(out, raw.data() + raw_offset, size);
        raw_offset += size;
        return true;
    }};
    u32 version{};
    u32 blob_count{};
    u32 record_count{};
    if (!read_raw(&version, sizeof(version)) || version != SPIRV_CACHE_VERSION ||
        !read_raw(&blob_count, sizeof(blob_count)) || !read_raw(&record_count, sizeof(record_count))) {
        return;
    }
    blobs.reserve(blob_count);
    for (u32 i = 0; i < blob_count; ++i) {
        u32 words{};
        if (!read_raw(&words, sizeof(words)) || words > MAX_TOTAL_WORDS) {
            return;
        }
        std::vector<u32> blob(words);
        if (words != 0 && !read_raw(blob.data(), words * sizeof(u32))) {
            return;
        }
        blobs.push_back(std::move(blob));
        total_words += words;
    }
    records.reserve(record_count);
    for (u32 i = 0; i < record_count; ++i) {
        Record record{};
        if (!read_raw(&record.key, sizeof(record.key)) ||
            !read_raw(&record.stage, sizeof(record.stage)) ||
            !read_raw(&record.blob, sizeof(record.blob)) || record.blob >= blobs.size()) {
            return;
        }
        record_index.emplace(RecordKey(record.key, record.stage), static_cast<u32>(records.size()));
        records.push_back(record);
    }
    for (u32 i = 0; i < blobs.size(); ++i) {
        blob_index.emplace(BlobHash(blobs[i]), i);
    }
}

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

void SpirvCache::Merge(const std::filesystem::path& other) {
    std::ifstream file(other, std::ios::binary);
    if (!file.is_open()) {
        return;
    }
    std::array<char, 8> magic{};
    file.read(magic.data(), magic.size());
    if (!file) {
        return;
    }
    file.close();
    if (magic != PACK_MAGIC) {
        MergeSpirv(other);
        return;
    }
    const std::vector<u8> bytes{ReadFileBytes(other)};
    std::vector<u8> spirv;
    std::vector<u8> pack_keys;
    bool found_vulkan{};
    for (const Section& section : ReadSections(std::span<const u8>{bytes}.subspan(magic.size()))) {
        // Sections for other backends are ignored until one exists to consume them.
        if (section.id == VULKAN_SECTION) {
            found_vulkan = SplitVulkanPayload(section.data, spirv, pack_keys);
        }
    }
    if (!found_vulkan) {
        LOG_WARNING(Render_Vulkan, "Shader pack '{}' has no usable section, ignoring",
                    Common::FS::PathToUTF8String(other));
        return;
    }
    if (!pack_keys.empty()) {
        // The key list is what makes an imported pack precompile: the loader rebuilds every
        // record for the local driver on the next boot.
        const std::filesystem::path keys_path{filename.parent_path() / "vulkan_import.bin"};
        std::ofstream out(keys_path, std::ios::binary | std::ios::trunc);
        out.write(reinterpret_cast<const char*>(pack_keys.data()), pack_keys.size());
        if (out) {
            LOG_INFO(Render_Vulkan, "Imported {} bytes of pipeline keys from a shader pack",
                     pack_keys.size());
        }
        std::scoped_lock lock{mutex};
        keys = pack_keys;
    }
    const std::filesystem::path tmp{filename.string() + ".merge"};
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        out.write(reinterpret_cast<const char*>(spirv.data()), spirv.size());
    }
    MergeSpirv(tmp);
    std::error_code ec;
    std::filesystem::remove(tmp, ec);
}

void SpirvCache::MergeSpirv(const std::filesystem::path& other) {
    if (!std::filesystem::exists(other)) {
        return;
    }
    SpirvCache incoming{other};
    std::scoped_lock lock{mutex};
    std::vector<u32> remap(incoming.blobs.size(), NO_BLOB);
    for (size_t i = 0; i < incoming.blobs.size(); ++i) {
        const std::vector<u32>& blob{incoming.blobs[i]};
        if (blob.empty()) {
            continue;
        }
        const u64 hash{BlobHash(blob)};
        if (const auto it{blob_index.find(hash)}; it != blob_index.end()) {
            remap[i] = it->second;
            continue;
        }
        if (total_words + blob.size() > MAX_TOTAL_WORDS) {
            continue;
        }
        remap[i] = static_cast<u32>(blobs.size());
        blobs.push_back(blob);
        blob_index.emplace(hash, remap[i]);
        total_words += blob.size();
    }
    size_t added{};
    for (const Record& record : incoming.records) {
        if (record.blob >= remap.size() || remap[record.blob] == NO_BLOB) {
            continue;
        }
        const u64 record_key{RecordKey(record.key, record.stage)};
        if (record_index.contains(record_key)) {
            continue;
        }
        record_index.emplace(record_key, static_cast<u32>(records.size()));
        records.push_back(Record{record.key, record.stage, remap[record.blob]});
        ++added;
    }
    dirty = true;
    LOG_INFO(Render_Vulkan, "Merged {} SPIR-V programs from '{}' ({} new records, {} total)",
             incoming.records.size(), Common::FS::PathToUTF8String(other), added, records.size());
}

void SpirvCache::Save() {
    std::vector<u8> raw;
    std::vector<u8> snapshot;
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
        snapshot = keys;
    }

    const std::vector<u8> compressed{
        Common::Compression::CompressDataZSTDDefault(raw.data(), raw.size())};
    std::vector<u8> spirv;
    spirv.insert(spirv.end(), MAGIC_NUMBER.begin(), MAGIC_NUMBER.end());
    const u32 raw_size{static_cast<u32>(raw.size())};
    const auto* raw_size_bytes = reinterpret_cast<const u8*>(&raw_size);
    spirv.insert(spirv.end(), raw_size_bytes, raw_size_bytes + sizeof(raw_size));
    spirv.insert(spirv.end(), compressed.begin(), compressed.end());

    // Refresh the key snapshot from the pipeline key list so the file is also a shareable pack.
    const std::vector<u8> keys_on_disk{ReadFileBytes(filename.parent_path() / "vulkan.bin")};
    if (!keys_on_disk.empty()) {
        snapshot = keys_on_disk;
    }

    const std::filesystem::path tmp{filename.string() + ".tmp"};
    {
        std::ofstream file(tmp, std::ios::binary | std::ios::trunc);
        if (!file.is_open()) {
            LOG_ERROR(Render_Vulkan, "Failed to open shader pack '{}'",
                      Common::FS::PathToUTF8String(tmp));
            return;
        }
        const auto write_block{[&file](const std::vector<u8>& block) {
            const u32 size{static_cast<u32>(block.size())};
            file.write(reinterpret_cast<const char*>(&size), sizeof(size));
            file.write(reinterpret_cast<const char*>(block.data()), block.size());
        }};
        // One backend tagged section today; a pack from another renderer simply adds its own.
        std::vector<u8> vulkan_payload;
        {
            const u32 spirv_size{static_cast<u32>(spirv.size())};
            const auto* spirv_size_bytes = reinterpret_cast<const u8*>(&spirv_size);
            vulkan_payload.insert(vulkan_payload.end(), spirv_size_bytes,
                                  spirv_size_bytes + sizeof(spirv_size));
            vulkan_payload.insert(vulkan_payload.end(), spirv.begin(), spirv.end());
            const u32 keys_size{static_cast<u32>(snapshot.size())};
            const auto* keys_size_bytes = reinterpret_cast<const u8*>(&keys_size);
            vulkan_payload.insert(vulkan_payload.end(), keys_size_bytes,
                                  keys_size_bytes + sizeof(keys_size));
            vulkan_payload.insert(vulkan_payload.end(), snapshot.begin(), snapshot.end());
        }
        file.write(PACK_MAGIC.data(), PACK_MAGIC.size());
        file.write(VULKAN_SECTION.data(), VULKAN_SECTION.size());
        write_block(vulkan_payload);
    }
    std::error_code ec;
    std::filesystem::rename(tmp, filename, ec);
    if (ec) {
        // Rename across filesystems fails; fall back to overwrite.
        std::filesystem::remove(filename, ec);
        std::filesystem::rename(tmp, filename, ec);
    }
    if (ec) {
        LOG_ERROR(Render_Vulkan, "Failed to store shader pack '{}': {}",
                  Common::FS::PathToUTF8String(filename), ec.message());
        return;
    }
    LOG_INFO(Render_Vulkan, "Saved {} cached SPIR-V programs ({} MB raw, {:.1f} MB packed, {} MB "
                            "pipeline keys)",
             records.size(), raw.size() / (1024 * 1024),
             static_cast<double>(spirv.size() + snapshot.size() + 12) / (1024.0 * 1024.0),
             snapshot.size() / (1024 * 1024));
}

bool SpirvCache::Dirty() const {
    std::scoped_lock lock{mutex};
    return dirty;
}

void SpirvCache::ClearDirty() {
    std::scoped_lock lock{mutex};
    dirty = false;
}

bool SpirvCache::RestoreKeys(const std::filesystem::path& path) const {
    std::scoped_lock lock{mutex};
    if (keys.empty()) {
        return false;
    }
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char*>(keys.data()), keys.size());
    return static_cast<bool>(out);
}

} // namespace Vulkan
