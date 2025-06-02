#!/bin/bash
# build_with_crlgru.sh - 効率的submoduleビルドスクリプト

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CRLGRU_DIR="${PROJECT_ROOT}/external/crlGRU"
CRLGRU_BUILD_DIR="${CRLGRU_DIR}/build_cache"
CRLGRU_LIB="${CRLGRU_BUILD_DIR}/libcrlGRU.a"

# カラー出力
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# crlGRUの変更チェック
check_crlgru_changes() {
    local last_build_hash_file="${CRLGRU_BUILD_DIR}/.last_build_hash"
    local current_hash=""
    
    if [ -d "${CRLGRU_DIR}/.git" ]; then
        current_hash=$(cd "${CRLGRU_DIR}" && git rev-parse HEAD)
    else
        # Gitでない場合はファイルのタイムスタンプを使用
        if [ -d "${CRLGRU_DIR}/include" ] && [ -d "${CRLGRU_DIR}/src" ]; then
            current_hash=$(find "${CRLGRU_DIR}/include" "${CRLGRU_DIR}/src" -type f \( -name "*.hpp" -o -name "*.cpp" \) 2>/dev/null | \
                          xargs stat -f "%m" 2>/dev/null | sort -n | tail -1 || echo "0")
        else
            current_hash="0"
        fi
    fi
    
    if [ -f "$last_build_hash_file" ]; then
        local last_hash=$(cat "$last_build_hash_file")
        if [ "$current_hash" = "$last_hash" ] && [ -f "$CRLGRU_LIB" ]; then
            return 1  # 変更なし
        fi
    fi
    
    mkdir -p "$(dirname "$last_build_hash_file")"
    echo "$current_hash" > "$last_build_hash_file"
    return 0  # 変更あり
}

# crlGRUビルド関数
build_crlgru() {
    log_info "Building crlGRU library..."
    
    # ビルドディレクトリ作成
    mkdir -p "${CRLGRU_BUILD_DIR}"
    
    # CMake設定
    cmake -S "${CRLGRU_DIR}" -B "${CRLGRU_BUILD_DIR}" \
        -DCRLGRU_BUILD_TESTS=OFF \
        -DCRLGRU_BUILD_SHARED=OFF \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        > "${CRLGRU_BUILD_DIR}/cmake.log" 2>&1
    
    # ビルド実行
    local nproc_count=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo "4")
    cmake --build "${CRLGRU_BUILD_DIR}" --target crlGRU --parallel "$nproc_count" \
        >> "${CRLGRU_BUILD_DIR}/build.log" 2>&1
    
    if [ -f "$CRLGRU_LIB" ]; then
        log_info "crlGRU build completed: $CRLGRU_LIB"
        return 0
    else
        log_error "crlGRU build failed"
        log_error "CMake log: ${CRLGRU_BUILD_DIR}/cmake.log"
        log_error "Build log: ${CRLGRU_BUILD_DIR}/build.log"
        return 1
    fi
}

# メイン処理
main() {
    log_info "Checking crlGRU dependency..."
    
    # submodule初期化（必要な場合）
    if [ ! -f "${CRLGRU_DIR}/CMakeLists.txt" ]; then
        if [ -d ".git" ]; then
            log_warn "Initializing git submodules..."
            git submodule update --init --recursive
        else
            log_error "crlGRU directory not found and not a git repository"
            log_error "Please ensure crlGRU is available at: ${CRLGRU_DIR}"
            exit 1
        fi
    fi
    
    # 変更チェック
    if check_crlgru_changes; then
        log_info "crlGRU changes detected, rebuilding..."
        build_crlgru || exit 1
    else
        log_info "crlGRU is up to date: $CRLGRU_LIB"
    fi
    
    # 親プロジェクトビルド
    log_info "Building main project..."
    mkdir -p build
    cmake -S . -B build \
        -DCRLGRU_ROOT="${CRLGRU_DIR}" \
        -DCRLGRU_BUILD_DIR="${CRLGRU_BUILD_DIR}" \
        "$@"
    
    local nproc_count=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo "4")
    cmake --build build --parallel "$nproc_count"
    
    log_info "Build completed successfully!"
}

# オプション処理
case "${1:-}" in
    --force-rebuild-crlgru)
        log_info "Force rebuilding crlGRU..."
        rm -rf "${CRLGRU_BUILD_DIR}"
        shift
        ;;
    --clean)
        log_info "Cleaning all build directories..."
        rm -rf build "${CRLGRU_BUILD_DIR}"
        exit 0
        ;;
    --help)
        echo "Usage: $0 [OPTIONS] [CMAKE_ARGS...]"
        echo "Options:"
        echo "  --force-rebuild-crlgru  Force rebuild crlGRU even if unchanged"
        echo "  --clean                 Clean all build directories"
        echo "  --help                  Show this help"
        echo ""
        echo "Examples:"
        echo "  $0                      # Standard build"
        echo "  $0 --clean              # Clean all"
        echo "  $0 -DCMAKE_BUILD_TYPE=Debug  # Debug build"
        exit 0
        ;;
esac

main "$@"
