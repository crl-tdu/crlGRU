# crlGRU CMakeLists.txt (submodule対応版)
cmake_minimum_required(VERSION 3.18)

# プロジェクトがトップレベルかチェック
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(CRLGRU_MAIN_PROJECT TRUE)
else()
    set(CRLGRU_MAIN_PROJECT FALSE)
endif()

project(crlGRU 
    VERSION 1.0.0
    DESCRIPTION "Free Energy Principle GRU Library for Multi-Agent Systems"
    LANGUAGES CXX
)

# オプション設定（submoduleの場合はデフォルトOFF）
option(CRLGRU_BUILD_SHARED "Build shared library" ${CRLGRU_MAIN_PROJECT})
option(CRLGRU_BUILD_STATIC "Build static library" ON)
option(CRLGRU_BUILD_TESTS "Build tests" ${CRLGRU_MAIN_PROJECT})
option(CRLGRU_INSTALL "Install crlGRU" ${CRLGRU_MAIN_PROJECT})
option(CRLGRU_DISABLE_TORCH "Disable LibTorch support" OFF)

# C++標準設定
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# コンパイラオプション（メインプロジェクトでのみ適用）
if(CRLGRU_MAIN_PROJECT)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra")
    if(APPLE)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
    endif()
endif()

# カスタムCMakeモジュール
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# LibTorch検索（キャッシュを活用）
if(NOT CRLGRU_DISABLE_TORCH AND NOT TARGET torch)
    include(FindLibTorch)
    find_libtorch_for_project(CRLGRU)
    if(LIBTORCH_FOUND)
        set(CRLGRU_HAS_TORCH TRUE)
    endif()
endif()

# OpenMP検索（オプショナル）
if(NOT TARGET OpenMP::OpenMP_CXX)
    find_package(OpenMP QUIET)
endif()

# RPATH設定（メインプロジェクトのみ）
if(CRLGRU_MAIN_PROJECT)
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
    set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
endif()

# ==========================================
# ライブラリターゲット定義
# ==========================================

# ヘッダーオンリーライブラリ（常に作成）
add_library(crlGRU_headers INTERFACE)
target_include_directories(crlGRU_headers INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# LibTorch リンク
if(CRLGRU_HAS_TORCH)
    # Remove duplicate library entries to fix warning
    list(REMOVE_DUPLICATES LIBTORCH_LIBRARIES)
    target_link_libraries(crlGRU_headers INTERFACE ${LIBTORCH_LIBRARIES})
    target_compile_definitions(crlGRU_headers INTERFACE CRLGRU_HAS_TORCH)
    if(CRLGRU_MAIN_PROJECT)
        message(STATUS "✓ LibTorch found and linked")
        message(STATUS "  Version: ${TORCH_VERSION}")
        message(STATUS "  Libraries: ${LIBTORCH_LIBRARIES}")
    endif()
endif()

# OpenMP リンク
if(OpenMP_CXX_FOUND)
    target_link_libraries(crlGRU_headers INTERFACE OpenMP::OpenMP_CXX)
    if(CRLGRU_MAIN_PROJECT)
        message(STATUS "✓ OpenMP found and linked")
    endif()
endif()

target_compile_features(crlGRU_headers INTERFACE cxx_std_17)

# 実装ライブラリソース
set(CRLGRU_SOURCES
    src/core/fep_gru_cell.cpp
    src/core/fep_gru_network.cpp
    src/core/attention_evaluator.cpp
    src/core/embodied_fep_gru_cell.cpp
)

# 静的ライブラリ（推奨、submoduleでデフォルト）
if(CRLGRU_BUILD_STATIC)
    add_library(crlGRU_static STATIC ${CRLGRU_SOURCES})
    target_link_libraries(crlGRU_static PUBLIC crlGRU_headers)
    set_target_properties(crlGRU_static PROPERTIES
        OUTPUT_NAME crlGRU
        POSITION_INDEPENDENT_CODE ON
    )
    
    # RPATH設定（静的ライブラリ用）
    if(CRLGRU_HAS_TORCH AND CRLGRU_MAIN_PROJECT)
        set_target_properties(crlGRU_static PROPERTIES
            INSTALL_RPATH_USE_LINK_PATH TRUE
            BUILD_WITH_INSTALL_RPATH TRUE
        )
    endif()
    
    # メインターゲットエイリアス
    if(NOT TARGET crlGRU)
        add_library(crlGRU ALIAS crlGRU_static)
    endif()
    
    if(CRLGRU_MAIN_PROJECT)
        message(STATUS "✓ Static library target: crlGRU_static")
    endif()
endif()

# 共有ライブラリ（メインプロジェクトでオプション）
if(CRLGRU_BUILD_SHARED)
    add_library(crlGRU_shared SHARED ${CRLGRU_SOURCES})
    target_link_libraries(crlGRU_shared PUBLIC crlGRU_headers)
    set_target_properties(crlGRU_shared PROPERTIES
        OUTPUT_NAME crlGRU
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR}
        POSITION_INDEPENDENT_CODE ON
    )
    
    # RPATH設定（共有ライブラリ用）
    if(CRLGRU_HAS_TORCH AND CRLGRU_MAIN_PROJECT)
        set_target_properties(crlGRU_shared PROPERTIES
            INSTALL_RPATH_USE_LINK_PATH TRUE
            BUILD_WITH_INSTALL_RPATH TRUE
        )
    endif()
    
    # 共有ライブラリを優先（メインプロジェクトの場合）
    if(NOT TARGET crlGRU AND CRLGRU_MAIN_PROJECT)
        add_library(crlGRU ALIAS crlGRU_shared)
    endif()
    
    if(CRLGRU_MAIN_PROJECT)
        message(STATUS "✓ Shared library target: crlGRU_shared")
    endif()
endif()

# ==========================================
# 条件付きコンポーネント
# ==========================================

# テスト（メインプロジェクトのみ）
if(CRLGRU_BUILD_TESTS AND CRLGRU_MAIN_PROJECT)
    enable_testing()
    add_subdirectory(tests)
    message(STATUS "✓ Tests enabled")
endif()

# ==========================================
# インストール設定（オプショナル）
# ==========================================
if(CRLGRU_INSTALL)
    include(GNUInstallDirs)
    include(CMakePackageConfigHelpers)
    
    # ヘッダーインストール
    install(DIRECTORY include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
    )
    
    # ライブラリインストール
    set(INSTALL_TARGETS crlGRU_headers)
    
    if(CRLGRU_BUILD_STATIC)
        list(APPEND INSTALL_TARGETS crlGRU_static)
    endif()
    
    if(CRLGRU_BUILD_SHARED)
        list(APPEND INSTALL_TARGETS crlGRU_shared)
    endif()
    
    install(TARGETS ${INSTALL_TARGETS}
        EXPORT crlGRUTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )
    
    # CMake設定ファイル作成
    write_basic_package_version_file(
        crlGRUConfigVersion.cmake
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion
    )
    
    install(EXPORT crlGRUTargets
        FILE crlGRUTargets.cmake
        NAMESPACE crlGRU::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/crlGRU
    )
    
    # 設定ファイルテンプレート作成
    set(CONFIG_TEMPLATE "${CMAKE_CURRENT_BINARY_DIR}/crlGRUConfig.cmake")
    file(WRITE ${CONFIG_TEMPLATE}
"@PACKAGE_INIT@

include(\"\${CMAKE_CURRENT_LIST_DIR}/crlGRUTargets.cmake\")

# Provide modern CMake targets
if(TARGET crlGRU::crlGRU_static AND NOT TARGET crlGRU::crlGRU)
    add_library(crlGRU::crlGRU ALIAS crlGRU::crlGRU_static)
elseif(TARGET crlGRU::crlGRU_shared AND NOT TARGET crlGRU::crlGRU)
    add_library(crlGRU::crlGRU ALIAS crlGRU::crlGRU_shared)
endif()

check_required_components(crlGRU)
")
    
    install(FILES
        ${CONFIG_TEMPLATE}
        ${CMAKE_CURRENT_BINARY_DIR}/crlGRUConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/crlGRU
    )
    
    message(STATUS "✓ Install rules configured")
endif()

# ==========================================
# ビルド情報表示（メインプロジェクトのみ）
# ==========================================
if(CRLGRU_MAIN_PROJECT)
    message(STATUS "")
    message(STATUS "=== crlGRU Configuration Summary ===")
    message(STATUS "Version: ${PROJECT_VERSION}")
    message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
    message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
    message(STATUS "Build shared: ${CRLGRU_BUILD_SHARED}")
    message(STATUS "Build static: ${CRLGRU_BUILD_STATIC}")
    message(STATUS "Build tests: ${CRLGRU_BUILD_TESTS}")
    message(STATUS "Install enabled: ${CRLGRU_INSTALL}")
    message(STATUS "LibTorch support: ${CRLGRU_HAS_TORCH}")
    if(CRLGRU_HAS_TORCH)
        message(STATUS "LibTorch version: ${TORCH_VERSION}")
    endif()
    message(STATUS "OpenMP support: ${OpenMP_CXX_FOUND}")
    message(STATUS "=====================================")
    message(STATUS "")
endif()
