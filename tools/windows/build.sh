#!/usr/bin/env bash
# Build driver for Eden on Windows (Git Bash / MSYS).
#
#   ./tools/windows/build.sh build          # incremental RelWithDebInfo build (Qt GUI)
#   ./tools/windows/build.sh build cli      # same, SDL3 frontend
#   ./tools/windows/build.sh configure      # (re)configure only
#   ./tools/windows/build.sh pgo-gen        # instrumented build for training
#   ./tools/windows/build.sh pgo-use        # merge profiles and link optimised
#   ./tools/windows/build.sh clean          # remove build directories
#   ./tools/windows/build.sh run            # launch the current binary
#
# Builds live in the repo root as build/ (Qt), build-cli/, build-pgo/ and are gitignored.
set -euo pipefail
export MSYS2_ARG_CONV_EXCL='*'

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROOTW="$(cygpath -m "$ROOT")"
JOBS="${JOBS:-3}"
CPM_CACHE="$ROOTW/.cache/cpm"
GLSLANG="$ROOTW/.cache/toolchain/bin/glslang.exe"

eval "$("$ROOT/tools/windows/vcvarsall.sh" x64)"
ORIGINAL_PATH="$PATH"
export PATH="$ROOTW/.cache/toolchain/bin:$ORIGINAL_PATH"

CONFIG=RelWithDebInfo

# Pin the host architecture instead of letting DetectArchitecture probe for
# symbols: a failed probe leaves ARCHITECTURE_x86_64 unset, which makes CPMUtil
# fall back to "CI package" downloads and dies on openssl with
# "Unsupported platform for CI package".
case "$(uname -m)" in
    x86_64|amd64) ARCH=x86_64 ;;
    aarch64|arm64) ARCH=arm64 ;;
    *) echo "unsupported host arch: $(uname -m)" >&2; exit 2 ;;

esac
C_FLAGS="-O2 -Ob2 -DNDEBUG -Zi -DARCHITECTURE_$ARCH=1"

command -v cl.exe >/dev/null || {
    echo "cl.exe not on PATH: run this from Git Bash (the VS env comes from tools/windows/vcvarsall.sh)" >&2
    exit 2
}

common_configure() {   # $1 build dir
    cmake -S "$ROOTW" -B "$1" -G Ninja \
        -DCMAKE_BUILD_TYPE="$CONFIG" \
        -DARCHITECTURE="$ARCH" \
        -DARCHITECTURE_$ARCH=ON \
        -DCMAKE_C_FLAGS_RELWITHDEBINFO="$C_FLAGS" \
        -DCMAKE_CXX_FLAGS_RELWITHDEBINFO="$C_FLAGS" \
        -DCPM_SOURCE_CACHE="$CPM_CACHE" \
        -DGLSLANGVALIDATOR="$GLSLANG" \
        -DENABLE_QT=ON \
        -DYUZU_TESTS=OFF \
        -DENABLE_WERROR=OFF
}

configure_cli() {
    common_configure "$ROOTW/build-cli"
    cmake -S "$ROOTW" -B "$ROOTW/build-cli" -DENABLE_QT=OFF \
        -DARCHITECTURE="$ARCH" -DARCHITECTURE_$ARCH=ON
}

build_dir_for() {      # $1 = "" | cli
    if [ "${1:-}" = "cli" ]; then echo "$ROOTW/build-cli"; else echo "$ROOTW/build"; fi
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
    [ -f "$dir/CMakeCache.txt" ] || cmd_configure "${1:-}"
    cmake --build "$dir" --target eden.exe -- "-j$JOBS"
    echo "built: $dir/bin/eden.exe"
}

cmd_run() {
    local dir; dir="$(build_dir_for "${1:-}")"
    exec "$dir/bin/eden.exe"
}

# --- PGO -------------------------------------------------------------------
# Three steps, in order. The profile lands next to the build root because link.exe
# rejects a drive-letter colon inside /PGD:.
PGO_DIR="$ROOTW/build-pgo"
PGO_PGD="$PGO_DIR/bin/eden.pgd"

cmd_pgo_gen() {
    common_configure "$PGO_DIR"
    cmake -S "$ROOTW" -B "$PGO_DIR" \
        -DCMAKE_C_FLAGS_RELWITHDEBINFO="$C_FLAGS /GL" \
        -DCMAKE_CXX_FLAGS_RELWITHDEBINFO="$C_FLAGS /GL" \
        -DCMAKE_EXE_LINKER_FLAGS="/LTCG /GENPROFILE /machine:x64"
    cmake --build "$PGO_DIR" --target eden.exe -- "-j$JOBS"
    echo "instrumented: $PGO_DIR/bin/eden.exe"
    echo "now play the scenes you care about and quit normally, then: $0 pgo-use"
}

cmd_pgo_use() {
    local pgc_count
    pgc_count=$(find "$PGO_DIR/bin" -name '*.pgc' | wc -l)
    [ "$pgc_count" -gt 0 ] || { echo "no .pgc data: run pgo-gen and train first" >&2; exit 1; }

    pgomgr /merge "$PGO_PGD" "$PGO_DIR"/bin/*.pgc
    rm -f "$PGO_DIR"/bin/*.pgc
    cp -f "$PGO_PGD" "$PGO_DIR/eden.pgd"

    cmake -S "$ROOTW" -B "$PGO_DIR" \
        -DCMAKE_EXE_LINKER_FLAGS="/LTCG:PGOPTIMIZE /PGD:eden.pgd /machine:x64"
    cmake --build "$PGO_DIR" --target eden.exe -- "-j$JOBS"

    if strings -a "$PGO_DIR/bin/eden.exe" 2>/dev/null | grep -qi '__profcxx'; then
        echo "warning: binary still looks instrumented" >&2
    fi
    echo "pgo build: $PGO_DIR/bin/eden.exe"
}

cmd_clean() {
    rm -rf "$ROOTW/build" "$ROOTW/build-cli" "$ROOTW/build-pgo"
    echo "removed build directories"
}

case "${1:-build}" in
    configure) shift; cmd_configure "${1:-}" ;;
    build)     shift; cmd_build "${1:-}" ;;
    run)       shift; cmd_run "${1:-}" ;;
    pgo-gen)   cmd_pgo_gen ;;
    pgo-use)   cmd_pgo_use ;;
    clean)     cmd_clean ;;
    *) echo "usage: $0 {build [cli]|configure [cli]|run [cli]|pgo-gen|pgo-use|clean}" >&2; exit 2 ;;
esac
