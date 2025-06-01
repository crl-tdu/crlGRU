#ifndef CRLGRU_CORE_FEP_GRU_CELL_HPP
#define CRLGRU_CORE_FEP_GRU_CELL_HPP

/// @file fep_gru_cell.hpp
/// @brief FEPGRUCell宣言（ライブラリ実装）

#include <crlgru/common.hpp>
#include <crlgru/utils/config_types.hpp>
#include <unordered_map>

namespace crlgru {

/// @brief 自由エネルギー原理に基づくGRUセル
/// @details 実装は src/core/fep_gru_cell.cpp にあります
class FEPGRUCell : public torch::nn::Module {
public:
    /// @brief 設定構造体の型エイリアス
    using Config = FEPGRUCellConfig;

private:
    Config config_;
    
    // 標準GRUパラメータ
    torch::nn::Linear input_to_hidden_{nullptr};
    torch::nn::Linear hidden_to_hidden_{nullptr};
    torch::nn::Linear input_to_reset_{nullptr};
    torch::nn::Linear hidden_to_reset_{nullptr};
    torch::nn::Linear input_to_update_{nullptr};
    torch::nn::Linear hidden_to_update_{nullptr};
    
    // FEP特有レイヤー
    torch::nn::Linear prediction_head_{nullptr};
    torch::nn::Linear variance_head_{nullptr};
    torch::nn::Linear meta_evaluation_head_{nullptr};
    
    // 内部状態
    torch::Tensor som_weights_;
    torch::Tensor som_activation_history_;
    torch::Tensor parameter_history_;
    torch::Tensor hidden_state_;
    torch::Tensor prediction_error_;
    torch::Tensor free_energy_;
    
    // ピア情報
    std::unordered_map<int, std::unordered_map<std::string, torch::Tensor>> peer_parameters_;
    std::unordered_map<int, double> peer_performance_;
    std::unordered_map<int, double> peer_trust_;

public:
    /// @brief コンストラクタ
    /// @param config FEP-GRUセル設定
    explicit FEPGRUCell(const Config& config);

    /// @brief フォワードパス
    /// @param input 入力テンソル [batch_size, input_size]
    /// @param hidden 隠れ状態 [batch_size, hidden_size]
    /// @return (新しい隠れ状態, 予測, 自由エネルギー)
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(
        const torch::Tensor& input,
        const torch::Tensor& hidden
    );

    /// @brief 自由エネルギー計算
    /// @param prediction 予測値
    /// @param target 目標値  
    /// @param variance 分散（オプション）
    /// @return 自由エネルギー
    torch::Tensor compute_free_energy(
        const torch::Tensor& prediction,
        const torch::Tensor& target,
        const torch::Tensor& variance = {}
    ) const;

    /// @brief SOM特徴抽出
    /// @return SOM特徴ベクトル
    torch::Tensor extract_som_features() const;

    /// @brief ピアからのパラメータ更新
    /// @param peer_id ピアID
    /// @param peer_params ピアパラメータ
    /// @param peer_performance ピア性能
    void update_parameters_from_peer(
        int peer_id,
        const std::unordered_map<std::string, torch::Tensor>& peer_params,
        double peer_performance
    );

    /// @brief 階層的模倣更新
    /// @param best_peer_id 最適ピアID
    void hierarchical_imitation_update(int best_peer_id);

    /// @brief メタ評価取得
    /// @return メタ評価値
    double get_meta_evaluation() const;

    /// @brief 自由エネルギー取得
    /// @return 自由エネルギーテンソル
    torch::Tensor get_free_energy() const { return free_energy_; }

    /// @brief 設定取得
    const Config& get_config() const { return config_; }

    /// @brief 状態リセット
    void reset_states();

private:
    /// @brief SOM更新
    void update_som(const torch::Tensor& input);
    
    /// @brief メタ評価計算
    torch::Tensor compute_meta_evaluation(const torch::Tensor& state);
};

} // namespace crlgru

#endif // CRLGRU_CORE_FEP_GRU_CELL_HPP
