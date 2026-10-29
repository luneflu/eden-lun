#!/usr/bin/env bash
# Build driver for Eden on Linux and Windows (Git Bash / MSYS).
#
#   ./tools/build.sh build          # incremental RelWithDebInfo build (Qt GUI)
#   ./tools/build.sh build cli      # same, SDL3 frontend
#   ./tools/build.sh configure      # (re)configure only
#   ./tools/build.sh pgo-gen        # instrumented build for training
#   ./tools/build.sh pgo-use        # merge profiles and link optimised
#   ./tools/build.sh clean          # remove build directories
#   ./tools/build.sh run            # launch the current binary
#
# Builds live in the repo root as build/ (Qt), build-cli/, build-pgo/ and are gitignored.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JOBS="${JOBS:-4}"
CONFIG=RelWithDebInfo

case "$(uname -m)" in
    x86_64|amd64) ARCH=x86_64 ;;
    aarch64|arm64) ARCH=arm64 ;;
    *) echo "unsupported host arch: $(uname -m)" >&2; exit 2 ;;
esac

case "$(uname -s)" in
    Linux*)
        IS_WINDOWS=0
        ROOTW="$ROOT"
        EXE_EXT=""
        TARGET_GUI="yuzu"
        TARGET_CLI="yuzu-cmd"
        BIN_GUI="eden"
        BIN_CLI="eden-cli"
        C_FLAGS="-O2 -DNDEBUG -DARCHITECTURE_$ARCH=1"
        CXX_FLAGS="-O2 -DNDEBUG -DARCHITECTURE_$ARCH=1"
        ;;
    CYGWIN*|MINGW*|MSYS*)
        IS_WINDOWS=1
        export MSYS2_ARG_CONV_EXCL='*'
        ROOTW="$(cygpath -m "$ROOT")"
        EXE_EXT=".exe"
        TARGET_GUI="eden.exe"
        TARGET_CLI="eden-cli.exe"
        BIN_GUI="eden.exe"
        BIN_CLI="eden-cli.exe"
        C_FLAGS="-O2 -Ob2 -DNDEBUG -Zi -DARCHITECTURE_$ARCH=1"
        CXX_FLAGS="-O2 -Ob2 -DNDEBUG -Zi -DARCHITECTURE_$ARCH=1"
        ;;
    *)
        echo "unsupported platform: $(uname -s)" >&2
        exit 2
        ;;
esac

# Cache directory & symlink check
if [ "$IS_WINDOWS" -eq 1 ]; then
    CPM_CACHE="$ROOTW/.cache/cpm"
    GLSLANG="$ROOTW/.cache/toolchain/bin/glslang.exe"
    eval "$("$ROOT/tools/windows/vcvarsall.sh" x64)"
    export PATH="$ROOTW/.cache/toolchain/bin:$PATH"
    command -v cl.exe >/dev/null || {
        echo "cl.exe not on PATH: run this from Git Bash (the VS env comes from tools/windows/vcvarsall.sh)" >&2
        exit 2
    }
else
    # Ext4 / symlink-capable path for CPM if repo is on exfat/FAT
    mkdir -p "$ROOT/.cache"
    if ln -s . "$ROOT/.cache/.symlink_test" 2>/dev/null; then
        rm -f "$ROOT/.cache/.symlink_test"
        CPM_CACHE="$ROOT/.cache/cpm"
    else
        rm -rf "$ROOT/.cache/cpm"
        CPM_CACHE="${XDG_CACHE_HOME:-$HOME/.cache}/eden/cpm"
    fi
    mkdir -p "$CPM_CACHE"

    # Toolchain: glslangValidator
    if command -v glslangValidator >/dev/null 2>&1; then
        GLSLANG="$(command -v glslangValidator)"
    elif command -v glslang >/dev/null 2>&1; then
        GLSLANG="$(command -v glslang)"
    elif [ -x "$ROOT/.cache/toolchain/bin/glslangValidator" ]; then
        GLSLANG="$ROOT/.cache/toolchain/bin/glslangValidator"
        export PATH="$ROOT/.cache/toolchain/bin:$PATH"
    else
        echo "glslangValidator not found; downloading prebuilt toolchain..."
        mkdir -p "$ROOT/.cache/toolchain/bin"
        local_tar="$ROOT/.cache/toolchain/glslang.tar.gz"
        curl -sL "https://github.com/KhronosGroup/glslang/releases/download/16.6.0/glslang-16.6.0-linux-x86_64-release.tar.gz" -o "$local_tar"
        tar -xzf "$local_tar" -C "$ROOT/.cache/toolchain/bin" --strip-components=1 bin/glslang
        cp -f "$ROOT/.cache/toolchain/bin/glslang" "$ROOT/.cache/toolchain/bin/glslangValidator"
        chmod +x "$ROOT/.cache/toolchain/bin/glslangValidator"
        rm -f "$local_tar"
        GLSLANG="$ROOT/.cache/toolchain/bin/glslangValidator"
        export PATH="$ROOT/.cache/toolchain/bin:$PATH"
    fi

    # libdrm & libva helper if dev packages missing
    export PKG_CONFIG_PATH="$HOME/.local/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
    export LIBRARY_PATH="$HOME/.local/lib:${LIBRARY_PATH:-}"
    mkdir -p "$HOME/.local/lib/pkgconfig"
    if ! pkg-config --exists libdrm 2>/dev/null && [ -f /usr/lib/x86_64-linux-gnu/libdrm.so.2 ]; then
        [ -e "$HOME/.local/lib/libdrm.so" ] || ln -sf /usr/lib/x86_64-linux-gnu/libdrm.so.2 "$HOME/.local/lib/libdrm.so"
        if [ ! -f "$HOME/.local/lib/pkgconfig/libdrm.pc" ]; then
            cat << 'EOF' > "$HOME/.local/lib/pkgconfig/libdrm.pc"
prefix=/usr
includedir=${prefix}/include
libdir=${HOME}/.local/lib

Name: libdrm
Description: Userspace interface to kernel DRM services
Version: 2.4.125
Libs: -L${libdir} -ldrm
Cflags: -I${includedir}
EOF
        fi
    fi
    if [ -f /usr/lib/x86_64-linux-gnu/libva.so.2 ]; then
        [ -e "$HOME/.local/lib/libva.so" ] || ln -sf /usr/lib/x86_64-linux-gnu/libva.so.2 "$HOME/.local/lib/libva.so"
        [ -e "$HOME/.local/lib/libva-drm.so" ] || ln -sf /usr/lib/x86_64-linux-gnu/libva-drm.so.2 "$HOME/.local/lib/libva-drm.so"
        [ -e "$HOME/.local/lib/libva-x11.so" ] || ln -sf /usr/lib/x86_64-linux-gnu/libva-x11.so.2 "$HOME/.local/lib/libva-x11.so"
        if [ ! -f "$HOME/.local/lib/libva_compat.a" ]; then
            cat << 'EOF' > "$HOME/.local/lib/va_compat.c"
typedef void* VADisplay;
typedef unsigned int VABufferID;
typedef int VAStatus;
extern VAStatus vaMapBuffer(VADisplay dpy, VABufferID buf_id, void **pbuf);
VAStatus vaMapBuffer2(VADisplay dpy, VABufferID buf_id, void **pbuf, unsigned int flags) {
    (void)flags;
    return vaMapBuffer(dpy, buf_id, pbuf);
}
EOF
            gcc -c -O2 "$HOME/.local/lib/va_compat.c" -o "$HOME/.local/lib/va_compat.o"
            ar rcs "$HOME/.local/lib/libva_compat.a" "$HOME/.local/lib/va_compat.o"
            rm -f "$HOME/.local/lib/va_compat.c" "$HOME/.local/lib/va_compat.o"
        fi
    fi

    command -v g++ >/dev/null || command -v clang++ >/dev/null || {
        echo "no C++ compiler found (g++ or clang++)" >&2
        exit 2
    }
fi

common_configure() {   # $1 build dir
    local extra=()
    if [ "$IS_WINDOWS" -eq 0 ]; then
        extra+=("-DCMAKE_PREFIX_PATH=$HOME/.local")
        extra+=("-DCMAKE_INCLUDE_PATH=$HOME/.local/include")
        extra+=("-DCMAKE_LIBRARY_PATH=$HOME/.local/lib")
        extra+=("-DCMAKE_EXE_LINKER_FLAGS=-L$HOME/.local/lib -Wl,-rpath,\$ORIGIN/lib:\$ORIGIN/../lib:$HOME/.local/lib")
        extra+=("-DSDL_X11_XSCRNSAVER=OFF")
        if ! pkg-config --exists Qt6Core 2>/dev/null; then
            extra+=("-DYUZU_USE_BUNDLED_QT=ON")
        fi
        if ! pkg-config --exists libavcodec 2>/dev/null; then
            extra+=("-DYUZU_USE_BUNDLED_FFMPEG=ON")
        fi
        if ! command -v autoconf >/dev/null 2>&1; then
            extra+=("-DENABLE_LIBUSB=OFF")
        fi
    fi

    cmake -S "$ROOTW" -B "$1" -G Ninja \
        -DCMAKE_BUILD_TYPE="$CONFIG" \
        -DARCHITECTURE="$ARCH" \
        -DARCHITECTURE_$ARCH=ON \
        -DCMAKE_C_FLAGS_RELWITHDEBINFO="$C_FLAGS -I$HOME/.local/include" \
        -DCMAKE_CXX_FLAGS_RELWITHDEBINFO="$CXX_FLAGS -I$HOME/.local/include" \
        -DCPM_SOURCE_CACHE="$CPM_CACHE" \
        -DGLSLANGVALIDATOR="$GLSLANG" \
        -DENABLE_QT=ON \
        -DYUZU_TESTS=OFF \
        -DENABLE_WERROR=OFF \
        "${extra[@]}"
}

configure_cli() {
    common_configure "$ROOTW/build-cli"
    cmake -S "$ROOTW" -B "$ROOTW/build-cli" -DENABLE_QT=OFF \
        -DARCHITECTURE="$ARCH" -DARCHITECTURE_$ARCH=ON
}

build_dir_for() {      # $1 = "" | cli
    if [ "${1:-}" = "cli" ]; then echo "$ROOTW/build-cli"; else echo "$ROOTW/build"; fi
}

target_for() {         # $1 = "" | cli
    if [ "${1:-}" = "cli" ]; then echo "$TARGET_CLI"; else echo "$TARGET_GUI"; fi
}

bin_for() {            # $1 = "" | cli
    local dir; dir="$(build_dir_for "${1:-}")"
    if [ "${1:-}" = "cli" ]; then echo "$dir/bin/$BIN_CLI"; else echo "$dir/bin/$BIN_GUI"; fi
}

cmd_configure() {
    if [ "${1:-}" = "cli" ]; then
        configure_cli
    else
        common_configure "$ROOTW/build"
    fi
}

cmd_build() {
    local dir; dir="$(build_dir_for "${1:-}")"
    local tgt; tgt="$(target_for "${1:-}")"
    local bin; bin="$(bin_for "${1:-}")"
    [ -f "$dir/CMakeCache.txt" ] || cmd_configure "${1:-}"
    cmake --build "$dir" --target "$tgt" -- "-j$JOBS"
    echo "built: $bin"
}

cmd_run() {
    local bin; bin="$(bin_for "${1:-}")"
    if [ "$IS_WINDOWS" -eq 0 ]; then
        export LD_LIBRARY_PATH="$HOME/.local/lib:${LD_LIBRARY_PATH:-}"
    fi
    exec "$bin" "$@"
}

# --- PGO -------------------------------------------------------------------
PGO_DIR="$ROOTW/build-pgo"
PGO_PGD="$PGO_DIR/bin/eden.pgd"
PGO_PROF_DIR="$PGO_DIR/profiles"

cmd_pgo_gen() {
    local tgt; tgt="$(target_for "${1:-}")"
    local bin_name; if [ "${1:-}" = "cli" ]; then bin_name="$BIN_CLI"; else bin_name="$BIN_GUI"; fi
    local bin="$PGO_DIR/bin/$bin_name"
    common_configure "$PGO_DIR"
    if [ "${1:-}" = "cli" ]; then
        cmake -S "$ROOTW" -B "$PGO_DIR" -DENABLE_QT=OFF
    fi

    if [ "$IS_WINDOWS" -eq 1 ]; then
        cmake -S "$ROOTW" -B "$PGO_DIR" \
            -DCMAKE_C_FLAGS_RELWITHDEBINFO="$C_FLAGS /GL" \
            -DCMAKE_CXX_FLAGS_RELWITHDEBINFO="$CXX_FLAGS /GL" \
            -DCMAKE_EXE_LINKER_FLAGS="/LTCG /GENPROFILE /machine:x64"
    else
        mkdir -p "$PGO_PROF_DIR"
        cmake -S "$ROOTW" -B "$PGO_DIR" \
            -DCMAKE_C_FLAGS_RELWITHDEBINFO="$C_FLAGS -I$HOME/.local/include -fprofile-generate=$PGO_PROF_DIR" \
            -DCMAKE_CXX_FLAGS_RELWITHDEBINFO="$CXX_FLAGS -I$HOME/.local/include -fprofile-generate=$PGO_PROF_DIR" \
            -DCMAKE_EXE_LINKER_FLAGS="-fprofile-generate=$PGO_PROF_DIR -L$HOME/.local/lib -Wl,-rpath,\$ORIGIN/lib:\$ORIGIN/../lib:$HOME/.local/lib"
    fi

    cmake --build "$PGO_DIR" --target "$tgt" -- "-j$JOBS"
    echo "instrumented: $bin"
    echo "now run/train with typical workloads, quit normally, then: $0 pgo-use"
}

cmd_pgo_use() {
    local tgt; tgt="$(target_for "${1:-}")"
    local bin_name; if [ "${1:-}" = "cli" ]; then bin_name="$BIN_CLI"; else bin_name="$BIN_GUI"; fi
    local bin="$PGO_DIR/bin/$bin_name"

    if [ "$IS_WINDOWS" -eq 1 ]; then
        local pgc_count
        pgc_count=$(find "$PGO_DIR/bin" -name '*.pgc' 2>/dev/null | wc -l)
        [ "$pgc_count" -gt 0 ] || { echo "no .pgc data: run pgo-gen and train first" >&2; exit 1; }

        pgomgr /merge "$PGO_PGD" "$PGO_DIR"/bin/*.pgc
        rm -f "$PGO_DIR"/bin/*.pgc
        cp -f "$PGO_PGD" "$PGO_DIR/eden.pgd"

        cmake -S "$ROOTW" -B "$PGO_DIR" \
            -DCMAKE_EXE_LINKER_FLAGS="/LTCG:PGOPTIMIZE /PGD:eden.pgd /machine:x64"
    else
        local prof_count
        prof_count=$(find "$PGO_PROF_DIR" -name '*.gcda' 2>/dev/null | wc -l)
        [ "$prof_count" -gt 0 ] || { echo "no profile data in $PGO_PROF_DIR: run pgo-gen and train first" >&2; exit 1; }

        cmake -S "$ROOTW" -B "$PGO_DIR" \
            -DCMAKE_C_FLAGS_RELWITHDEBINFO="$C_FLAGS -I$HOME/.local/include -fprofile-use=$PGO_PROF_DIR -fprofile-correction -Wno-error=coverage-mismatch" \
            -DCMAKE_CXX_FLAGS_RELWITHDEBINFO="$CXX_FLAGS -I$HOME/.local/include -fprofile-use=$PGO_PROF_DIR -fprofile-correction -Wno-error=coverage-mismatch" \
            -DCMAKE_EXE_LINKER_FLAGS="-fprofile-use=$PGO_PROF_DIR -L$HOME/.local/lib -Wl,-rpath,\$ORIGIN/lib:\$ORIGIN/../lib:$HOME/.local/lib"
    fi

    cmake --build "$PGO_DIR" --target "$tgt" -- "-j$JOBS"

    if [ "$IS_WINDOWS" -eq 1 ]; then
        if strings -a "$bin" 2>/dev/null | grep -qi '__profcxx'; then
            echo "warning: binary still looks instrumented" >&2
        fi
    else
        if strings -a "$bin" 2>/dev/null | grep -qi '__gcov'; then
            echo "warning: binary still looks instrumented" >&2
        fi
    fi
    echo "pgo build: $bin"
}

cmd_appimage() {
    [ "$IS_WINDOWS" -eq 1 ] && { echo "appimage only supported on Linux" >&2; exit 1; }

    local bin="${1:-}"
    if [ -z "$bin" ]; then
        if [ -f "$ROOTW/build-pgo/bin/eden" ]; then
            bin="$ROOTW/build-pgo/bin/eden"
        elif [ -f "$ROOTW/build/bin/eden" ]; then
            bin="$ROOTW/build/bin/eden"
        else
            echo "error: no binary found. Run '$0 build' or '$0 pgo-use' first." >&2
            exit 1
        fi
    fi

    local tool="$HOME/.local/bin/appimagetool"
    if ! command -v appimagetool >/dev/null 2>&1 && [ ! -x "$tool" ]; then
        echo "downloading appimagetool..."
        mkdir -p "$HOME/.local/bin"
        wget -q --show-progress -c "https://github.com/AppImage/appimagetool/releases/latest/download/appimagetool-x86_64.AppImage" -O "$tool"
        chmod +x "$tool"
    fi
    local appimagetool_cmd="appimagetool"
    command -v appimagetool >/dev/null 2>&1 || appimagetool_cmd="$tool"

    local appdir="/tmp/eden-appdir.$$"
    rm -rf "$appdir"
    mkdir -p "$appdir/usr/bin" "$appdir/usr/lib" "$appdir/usr/plugins"

    cp "$bin" "$appdir/usr/bin/eden"
    cp "$ROOTW/dist/dev.eden_emu.eden.desktop" "$appdir/dev.eden_emu.eden.desktop"
    ln -sf dev.eden_emu.eden.desktop "$appdir/eden.desktop"
    cp "$ROOTW/dist/dev.eden_emu.eden.svg" "$appdir/dev.eden_emu.eden.svg"
    ln -sf dev.eden_emu.eden.svg "$appdir/.DirIcon"

    local qt_dir
    qt_dir="$(ls -d "$CPM_CACHE"/qt6/*-linux-amd64 2>/dev/null | head -n1)"
    if [ -d "$qt_dir" ]; then
        cp -P "$qt_dir"/lib/libQt6*.so* "$appdir/usr/lib/" 2>/dev/null || true
        cp -r "$qt_dir"/plugins/* "$appdir/usr/plugins/" 2>/dev/null || true
    fi

    cp -P "$HOME/.local/lib"/*.so* "$appdir/usr/lib/" 2>/dev/null || true

    cat << 'EOF' > "$appdir/AppRun"
#!/bin/sh
HERE="$(dirname "$(readlink -f "${0}")")"
export LD_LIBRARY_PATH="$HERE/usr/lib:${LD_LIBRARY_PATH}"
export QT_PLUGIN_PATH="$HERE/usr/plugins"
export QML2_IMPORT_PATH="$HERE/usr/qml"
exec "$HERE/usr/bin/eden" "$@"
EOF
    chmod +x "$appdir/AppRun"

    local out="Eden-x86_64.AppImage"
    ARCH=x86_64 APPIMAGE_EXTRACT_AND_RUN=1 "$appimagetool_cmd" "$appdir" "$out"
    rm -rf "$appdir"
    echo "built AppImage: $ROOTW/$out"
}

cmd_clean() {
    rm -rf "$ROOTW/build" "$ROOTW/build-cli" "$ROOTW/build-pgo" "$ROOTW"/*.AppImage
    echo "removed build directories"
}

case "${1:-build}" in
    configure) shift || true; cmd_configure "${1:-}" ;;
    build)     shift || true; cmd_build "${1:-}" ;;
    run)       shift || true; cmd_run "${1:-}" ;;
    pgo|pgo-gen) shift || true; cmd_pgo_gen "${1:-}" ;;
    pgo-use)   shift || true; cmd_pgo_use "${1:-}" ;;
    appimage)  shift || true; cmd_appimage "${1:-}" ;;
    clean)     cmd_clean ;;
    *) echo "usage: $0 {build [cli]|configure [cli]|run [cli]|pgo-gen|pgo-use|appimage [path]|clean}" >&2; exit 2 ;;
esac
