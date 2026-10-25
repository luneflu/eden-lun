// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <QObject>
#include "common/common_types.h"

namespace QtCommon::Path {
bool OpenShaderCache(u64 program_id, QObject* parent);

// Imports a SPIR-V cache pack for this title. When `merge` is false the existing store is removed
// first, so the imported pack replaces it.
// `merge` false replaces the existing store with the imported pack.
bool ImportShaderPack(u64 program_id, bool merge, QObject* parent);

// Bundles the SPIR-V cache and the pipeline key list into one shareable pack.
bool ExportShaderPack(u64 program_id, QObject* parent);

// Removes the SPIR-V store and any imported pack for this title.
bool DeleteShaderPack(u64 program_id, QObject* parent);
}
