#ifndef CRLGRU_CRL_GRU_HPP
#define CRLGRU_CRL_GRU_HPP

/// @file crl_gru.hpp
/// @brief crlGRU ライブラリ統合ヘッダー（ハイブリッドアプローチ版）
/// @version 1.0.0-hybrid

// 共通定義
#include <crlgru/common.hpp>

// ライブラリコンポーネント（宣言のみ）
#include <crlgru/core/fep_gru_cell.hpp>
#include <crlgru/core/fep_gru_network.hpp>
#include <crlgru/core/polar_spatial_attention.hpp>

// ヘッダーオンリーコンポーネント（実装含む）
#include <crlgru/utils/config_types.hpp>
#include <crlgru/utils/spatial_transforms.hpp>
#include <crlgru/utils/math_utils.hpp>
#include <crlgru/optimizers/spsa_optimizer.hpp>

/// @brief crlGRU ライブラリ名前空間
namespace crlgru {

/// @brief ライブラリ初期化（オプション）
/// @return 初期化成功フラグ
inline bool initialize() {
    // 必要に応じて初期化処理
    return true;
}

/// @brief ライブラリ終了処理（オプション）
inline void finalize() {
    // 必要に応じて終了処理
}

/// @brief バージョン情報取得
/// @return バージョン文字列
inline const char* get_version() {
    return VERSION;
}

/// @brief 互換性チェック
/// @param required_major 要求メジャーバージョン
/// @param required_minor 要求マイナーバージョン
/// @return 互換性フラグ
inline bool check_compatibility(int required_major, int required_minor) {
    return (VERSION_MAJOR == required_major) && (VERSION_MINOR >= required_minor);
}

// 便利な型エイリアス
using FEPGRUCellPtr = std::shared_ptr<FEPGRUCell>;
using SPSAOptimizerPtr = std::unique_ptr<SPSAOptimizer<double>>;
using SPSAOptimizerFPtr = std::unique_ptr<SPSAOptimizer<float>>;

// 便利な構築関数
/// @brief FEPGRUCell作成
/// @param config 設定
/// @return FEPGRUCellのスマートポインタ
inline FEPGRUCellPtr make_fep_gru_cell(const FEPGRUCellConfig& config = {}) {
    return std::make_shared<FEPGRUCell>(config);
}

/// @brief SPSA最適化器作成（double版）
/// @param parameters パラメータリスト
/// @param config 設定
/// @return SPSA最適化器のユニークポインタ
inline SPSAOptimizerPtr make_spsa_optimizer_d(
    const std::vector<torch::Tensor>& parameters,
    const SPSAOptimizerConfig& config = {}) {
    
    typename SPSAOptimizer<double>::Config spsa_config;
    spsa_config.a = config.a;
    spsa_config.c = config.c;
    spsa_config.A = config.A;
    spsa_config.alpha = config.alpha;
    spsa_config.gamma = config.gamma;
    spsa_config.param_min = config.param_min;
    spsa_config.param_max = config.param_max;
    spsa_config.use_momentum = config.use_momentum;
    spsa_config.momentum_beta = config.momentum_beta;
    spsa_config.random_seed = config.random_seed;
    spsa_config.tolerance = config.tolerance;
    spsa_config.max_iterations = config.max_iterations;
    spsa_config.learning_rate = config.learning_rate;
    spsa_config.gradient_smoothing = config.gradient_smoothing;
    
    return std::make_unique<SPSAOptimizer<double>>(parameters, spsa_config);
}

/// @brief SPSA最適化器作成（float版）
/// @param parameters パラメータリスト
/// @param config 設定
/// @return SPSA最適化器のユニークポインタ
inline SPSAOptimizerFPtr make_spsa_optimizer_f(
    const std::vector<torch::Tensor>& parameters,
    const SPSAOptimizerConfig& config = {}) {
    
    typename SPSAOptimizer<float>::Config spsa_config;
    spsa_config.a = static_cast<float>(config.a);
    spsa_config.c = static_cast<float>(config.c);
    spsa_config.A = static_cast<float>(config.A);
    spsa_config.alpha = static_cast<float>(config.alpha);
    spsa_config.gamma = static_cast<float>(config.gamma);
    spsa_config.param_min = static_cast<float>(config.param_min);
    spsa_config.param_max = static_cast<float>(config.param_max);
    spsa_config.use_momentum = config.use_momentum;
    spsa_config.momentum_beta = static_cast<float>(config.momentum_beta);
    spsa_config.random_seed = config.random_seed;
    spsa_config.tolerance = static_cast<float>(config.tolerance);
    spsa_config.max_iterations = config.max_iterations;
    spsa_config.learning_rate = static_cast<float>(config.learning_rate);
    spsa_config.gradient_smoothing = static_cast<float>(config.gradient_smoothing);
    
    return std::make_unique<SPSAOptimizer<float>>(parameters, spsa_config);
}

} // namespace crlgru

#endif // CRLGRU_CRL_GRU_HPP
