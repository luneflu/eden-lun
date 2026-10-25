// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <filesystem>
#include <fstream>
#include <vector>

#include <QDesktopServices>
#include <QFile>
#include <QFileDialog>
#include <QMessageBox>
#include <QString>
#include <QUrl>
#include <fmt/format.h>
#include "common/fs/fs.h"
#include "common/fs/path_util.h"
#include "qt_common/abstract/frontend.h"
#include "qt_common/util/path.h"

namespace QtCommon::Path {

bool OpenShaderCache(u64 program_id, QObject* parent) {
    const auto shader_cache_dir = Common::FS::GetEdenPath(Common::FS::EdenPath::ShaderDir);
    const auto shader_cache_folder_path{shader_cache_dir / fmt::format("{:016x}", program_id)};
    if (!Common::FS::CreateDirs(shader_cache_folder_path)) {
        QtCommon::Frontend::Warning(tr("Error Opening Shader Cache"),
                                    tr("Failed to create or open shader cache for this title, "
                                       "ensure your app data directory has write permissions."));
    }

    const auto shader_path_string{Common::FS::PathToUTF8String(shader_cache_folder_path)};
    const auto qt_shader_cache_path = QString::fromStdString(shader_path_string);
    return QDesktopServices::openUrl(QUrl::fromLocalFile(qt_shader_cache_path));
}

bool ImportShaderPack(u64 program_id, bool merge, QObject* parent) {
    const auto shader_cache_dir = Common::FS::GetEdenPath(Common::FS::EdenPath::ShaderDir);
    const auto shader_cache_folder_path{shader_cache_dir / fmt::format("{:016x}", program_id)};
    if (!Common::FS::CreateDirs(shader_cache_folder_path)) {
        QtCommon::Frontend::Warning(tr("Import Shader Pack"),
                                    tr("Failed to create the shader cache folder for this title, "
                                       "ensure your app data directory has write permissions."));
        return false;
    }

    const QString source{QFileDialog::getOpenFileName(
        qobject_cast<QWidget*>(parent), tr("Select shader pack"), QString(),
        tr("Nintendo Switch shader pack (*.nxspack *.bin);;All files (*)"))};
    if (source.isEmpty()) {
        return false;
    }

    if (!merge && std::filesystem::exists(shader_cache_folder_path / fmt::format("{:016x}.nxspack", program_id))) {
        const auto answer{QMessageBox::question(
            qobject_cast<QWidget*>(parent), tr("Import Shader Pack"),
            tr("This title already has a shader cache. Importing replaces it with the contents "
               "of the selected pack, together with the pipeline keys it carries."),
            QMessageBox::Yes | QMessageBox::No, QMessageBox::No)};
        if (answer != QMessageBox::Yes) {
            return false;
        }
    }

    // The renderer merges the pack on the next boot.
    const auto import_path{shader_cache_folder_path / "spirv_import.bin"};
    Common::FS::RemoveFile(import_path);
    if (!merge) {
        Common::FS::RemoveFile(shader_cache_folder_path / fmt::format("{:016x}.nxspack", program_id));
        Common::FS::RemoveFile(shader_cache_folder_path / "vulkan_import.bin");
    }
    if (!QFile::copy(source, QString::fromStdString(Common::FS::PathToUTF8String(import_path)))) {
        QtCommon::Frontend::Warning(tr("Import Shader Pack"),
                                    tr("Failed to copy the selected file into the shader cache."));
        return false;
    }

    QMessageBox::information(
        qobject_cast<QWidget*>(parent), tr("Import Shader Pack"),
        merge ? tr("Pack queued for import. It will be merged into this title's cache the next time the "
                   "game starts.")
              : tr("Pack queued for import. The existing cache for this title was replaced and "
                   "will be rebuilt the next time the game starts."));
    return true;
}

bool ExportShaderPack(u64 program_id, QObject* parent) {
    const auto shader_cache_dir = Common::FS::GetEdenPath(Common::FS::EdenPath::ShaderDir);
    const auto shader_cache_folder_path{shader_cache_dir / fmt::format("{:016x}", program_id)};
    // The store already is a pack, so export is a copy.
    const auto pack_path{shader_cache_folder_path / fmt::format("{:016x}.nxspack", program_id)};
    if (!std::filesystem::exists(pack_path)) {
        QtCommon::Frontend::Warning(tr("Export Shader Pack"),
                                    tr("No shader pack to export. Play the game once first."));
        return false;
    }
    const QString target{QFileDialog::getSaveFileName(
        qobject_cast<QWidget*>(parent), tr("Save shader pack"),
        QString::fromStdString(fmt::format("{:016x}.nxspack", program_id)),
        tr("Nintendo Switch shader pack (*.nxspack);;All files (*)"))};
    if (target.isEmpty()) {
        return false;
    }
    if (!QFile::copy(QString::fromStdString(Common::FS::PathToUTF8String(pack_path)), target)) {
        QtCommon::Frontend::Warning(tr("Export Shader Pack"),
                                    tr("Failed to write the pack. Pick another location."));
        return false;
    }
    QMessageBox::information(
        qobject_cast<QWidget*>(parent), tr("Export Shader Pack"),
        tr("Pack written. Anyone can import it: the SPIR-V is device agnostic and the pipeline "
           "keys are rebuilt for their own GPU on the next launch."));
    return true;
}

bool DeleteShaderPack(u64 program_id, QObject* parent) {
    const auto shader_cache_dir = Common::FS::GetEdenPath(Common::FS::EdenPath::ShaderDir);
    const auto shader_cache_folder_path{shader_cache_dir / fmt::format("{:016x}", program_id)};
    const auto store{shader_cache_folder_path / fmt::format("{:016x}.nxspack", program_id)};
    const auto imported{shader_cache_folder_path / "vulkan_import.bin"};
    if (!std::filesystem::exists(store) && !std::filesystem::exists(imported)) {
        QtCommon::Frontend::Warning(tr("Delete Shader Pack"),
                                    tr("This title has no shader pack to delete."));
        return false;
    }
    const auto answer{QMessageBox::warning(
        qobject_cast<QWidget*>(parent), tr("Delete Shader Pack"),
        tr("Deletes the stored SPIR-V and the imported pipeline keys for this title.") +
            QStringLiteral("\n\n") +
            tr("The game will stutter again while every pipeline is compiled from scratch, and ") +
            tr("an imported pack has to be imported again.") +
            QStringLiteral("\n\n") + tr("Delete anyway?"),
        QMessageBox::Yes | QMessageBox::No, QMessageBox::No)};
    if (answer != QMessageBox::Yes) {
        return false;
    }
    Common::FS::RemoveFile(store);
    Common::FS::RemoveFile(imported);
    return true;
}

} // namespace QtCommon::Path
