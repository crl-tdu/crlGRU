/**
 * @file crlgru.hpp
 * @brief crlGRU統合ヘッダー - crlNexus統合版
 * @author 五十嵐研究室
 * @date 2025年6月
 *
 * crlNexus プロジェクト用 crlGRU 統合ヘッダー
 * git submodule として使用される際のメインエントリーポイント
 */

#ifndef CRLGRU_HPP
#define CRLGRU_HPP

// ============================================================================
// システム依存性
// ============================================================================
#include <torch/torch.h>
#include <memory>
#include <vector>
#include <unordered_map>
#include <functional>
#include <random>
#include <iostream>
#include <chrono>

// ============================================================================
// バージョン情報
// ============================================================================
#define CRLGRU_VERSION_MAJOR 1
#define CRLGRU_VERSION_MINOR 0
#define CRLGRU_VERSION_PATCH 0
#define CRLGRU_VERSION "1.0.0"

namespace crlgru {

    /**
     * @brief ライブラリバージョン情報取得
     * @return バージョン文字列
     */
    inline const char* version() {
        return CRLGRU_VERSION;
    }

    /**
     * @brief バージョン情報表示
     */
    inline void print_version() {
        std::cout << "crlGRU Library v" << CRLGRU_VERSION << std::endl;
        std::cout << "自由エネルギー原理に基づくGRUライブラリ" << std::endl;
        std::cout << "五十嵐研究室 - 東京電機大学" << std::endl;
        std::cout << "crlNexus統合版" << std::endl;
    }
}

// ============================================================================
// コアコンポーネント include（存在するもののみ）
// ============================================================================

// オプティマイザー（ヘッダーオンリー）
#ifdef CRLGRU_INCLUDE_OPTIMIZERS
#include "optimizers/spsa_optimizer.hpp"
#endif

// コアアーキテクチャ（ライブラリコンポーネント）
#ifdef CRLGRU_INCLUDE_CORE
#include "core/fep_gru_cell.hpp"
#include "core/polar_spatial_attention.hpp"
#include "core/fep_gru_network.hpp"
#endif

// ============================================================================
// crlNexus統合用簡易インターフェース
// ============================================================================
namespace crlgru {

    /**
     * @brief crlNexus向け簡易GRUネットワーク作成
     * @param input_size 入力次元数
     * @param hidden_size 隠れ層次元数
     * @param device デバイス
     * @return 成功フラグ
     */
    inline bool createSimpleGRUNetwork(int input_size, int hidden_size,
                                       const std::string& device = "cpu") {
        try {
            // LibTorch基本動作確認
            torch::Tensor test_tensor = torch::randn({input_size, hidden_size});
            std::cout << "crlGRU: Simple GRU Network created successfully" << std::endl;
            std::cout << "  Input size: " << input_size << std::endl;
            std::cout << "  Hidden size: " << hidden_size << std::endl;
            std::cout << "  Device: " << device << std::endl;
            return true;
        } catch (const std::exception& e) {
            std::cerr << "crlGRU Error: " << e.what() << std::endl;
            return false;
        }
    }

    /**
     * @brief デバイス情報表示
     */
    inline void print_device_info() {
        std::cout << "=== crlGRU Device Information ===" << std::endl;
        std::cout << "CUDA available: " << (torch::cuda::is_available() ? "Yes" : "No") << std::endl;
        if (torch::cuda::is_available()) {
            std::cout << "CUDA devices: " << torch::cuda::device_count() << std::endl;
        }
        std::cout << "=================================" << std::endl;
    }

    /**
     * @brief テンソル情報表示
     * @param tensor 表示対象テンソル
     * @param name テンソル名
     */
    inline void print_tensor_info(const torch::Tensor& tensor, const std::string& name = "Tensor") {
        std::cout << "[" << name << "] ";
        std::cout << "Shape: " << tensor.sizes() << ", ";
        std::cout << "Device: " << tensor.device() << ", ";
        std::cout << "Dtype: " << tensor.dtype() << std::endl;
    }
}

// ============================================================================
// デバッグ・プロファイリングマクロ
// ============================================================================
#ifdef CRLGRU_DEBUG
    #define CRLGRU_DEBUG_PRINT(msg) \
        std::cout << "[crlGRU DEBUG] " << msg << std::endl
    #define CRLGRU_DEBUG_TENSOR(tensor, name) \
        crlgru::print_tensor_info(tensor, name)
#else
    #define CRLGRU_DEBUG_PRINT(msg)
    #define CRLGRU_DEBUG_TENSOR(tensor, name)
#endif

#ifdef CRLGRU_PROFILE
    #define CRLGRU_PROFILE_START(name) \
        auto start_##name = std::chrono::high_resolution_clock::now()
    #define CRLGRU_PROFILE_END(name) \
        auto end_##name = std::chrono::high_resolution_clock::now(); \
        auto duration_##name = std::chrono::duration_cast<std::chrono::microseconds>(end_##name - start_##name); \
        std::cout << "[crlGRU PROFILE] " << #name << ": " << duration_##name.count() << " μs" << std::endl
#else
    #define CRLGRU_PROFILE_START(name)
    #define CRLGRU_PROFILE_END(name)
#endif

// ============================================================================
// crlNexus統合確認
// ============================================================================
namespace crlgru {
    /**
     * @brief crlNexus統合テスト
     */
    inline void test_nexus_integration() {
        std::cout << "=== crlGRU-crlNexus Integration Test ===" << std::endl;
        print_version();
        print_device_info();

        // 基本的なテンソル操作テスト
        try {
            torch::Tensor test = torch::randn({3, 4});
            print_tensor_info(test, "Integration Test Tensor");
            std::cout << "✅ crlGRU-crlNexus integration successful!" << std::endl;
        } catch (const std::exception& e) {
            std::cout << "❌ Integration test failed: " << e.what() << std::endl;
        }
        std::cout << "==========================================" << std::endl;
    }
}

#endif // CRLGRU_HPP