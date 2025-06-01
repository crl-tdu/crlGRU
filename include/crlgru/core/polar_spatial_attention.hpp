#ifndef CRLGRU_CORE_POLAR_SPATIAL_ATTENTION_HPP
#define CRLGRU_CORE_POLAR_SPATIAL_ATTENTION_HPP

/// @file polar_spatial_attention.hpp
/// @brief PolarSpatialAttention宣言（ライブラリ実装）

#include <crlgru/common.hpp>
#include <crlgru/utils/config_types.hpp>

namespace crlgru {

/// @brief 極座標空間注意機構
/// @details 実装は src/core/attention_evaluator.cpp にあります
class PolarSpatialAttention : public torch::nn::Module {
public:
    /// @brief 設定構造体の型エイリアス
    using AttentionConfig = PolarSpatialAttentionConfig;

private:
    AttentionConfig config_;
    
    // PyTorchモジュール（実装で使用されている名前に合わせる）
    torch::nn::Conv2d distance_attention_{nullptr};
    torch::nn::Conv2d angle_attention_{nullptr};
    torch::nn::Linear fusion_layer_{nullptr};
    torch::nn::Dropout dropout_{nullptr};
    
    // 適応パラメータ
    torch::Tensor adaptive_range_;
    torch::Tensor adaptive_sectors_;
    torch::Tensor attention_history_;

public:
    /// @brief コンストラクタ
    /// @param config 注意機構設定
    explicit PolarSpatialAttention(const AttentionConfig& config);

    /// @brief フォワードパス
    /// @param polar_map 極座標マップ [batch_size, rings, sectors, features]
    /// @return 注意重み付き特徴 [batch_size, features]
    torch::Tensor forward(const torch::Tensor& polar_map);

    /// @brief 注意重み計算
    /// @param features 入力特徴 [batch_size, channels, rings, sectors]
    /// @return (distance_weights, angle_weights) のペア
    std::pair<torch::Tensor, torch::Tensor> compute_attention_weights(const torch::Tensor& features);

    /// @brief 距離ベース重み
    /// @param distances 距離テンソル [batch_size, rings, sectors]
    /// @return 距離重み [batch_size, rings, sectors]
    torch::Tensor compute_distance_weights(const torch::Tensor& distances) const;

    /// @brief 角度ベース重み
    /// @param angles 角度テンソル [batch_size, rings, sectors]
    /// @return 角度重み [batch_size, rings, sectors]
    torch::Tensor compute_angle_weights(const torch::Tensor& angles) const;

    /// @brief 設定取得
    const AttentionConfig& get_config() const { return config_; }

    /// @brief 適応パラメータ更新
    void update_adaptive_parameters();

private:
    /// @brief レイヤー初期化
    void initialize_layers();
    
    /// @brief ガウシアン重み計算
    torch::Tensor gaussian_weights(const torch::Tensor& x, double sigma) const;
    
    /// @brief ソフトマックス重み計算
    torch::Tensor softmax_weights(const torch::Tensor& x, int dim) const;
};

/// @brief メタ評価器
/// @details 実装は src/core/attention_evaluator.cpp にあります
class MetaEvaluator {
public:
    /// @brief 設定構造体の型エイリアス
    using EvaluationConfig = MetaEvaluatorConfig;

private:
    EvaluationConfig config_;
    
    // 評価関数（実装に合わせる）
    std::vector<std::function<double(const torch::Tensor&)>> objective_functions_;
    torch::Tensor weight_history_;  // 重み履歴管理

public:
    /// @brief コンストラクタ
    /// @param config 評価設定
    explicit MetaEvaluator(const EvaluationConfig& config);

    /// @brief 統合評価
    /// @param predicted_states 予測状態
    /// @param current_state 現在の状態
    /// @param environment_state 環境状態
    /// @return 統合評価値
    double evaluate(const torch::Tensor& predicted_states,
                   const torch::Tensor& current_state,
                   const torch::Tensor& environment_state);

    /// @brief 評価関数追加
    /// @param objective 評価関数
    void add_objective(std::function<double(const torch::Tensor&)> objective);

    /// @brief 重み適応
    /// @param recent_performance 最近の性能履歴
    void adapt_weights(const std::vector<double>& recent_performance);

    /// @brief 設定取得
    const EvaluationConfig& get_config() const { return config_; }

private:
    // （実装で実際に使用されているメソッドのみ）
};

} // namespace crlgru

#endif // CRLGRU_CORE_POLAR_SPATIAL_ATTENTION_HPP
