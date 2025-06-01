#ifndef CRLGRU_CORE_FEP_GRU_NETWORK_HPP
#define CRLGRU_CORE_FEP_GRU_NETWORK_HPP

/// @file fep_gru_network.hpp
/// @brief FEPGRUNetwork宣言（ライブラリ実装）

#include <crlgru/common.hpp>
#include <crlgru/utils/config_types.hpp>
#include <crlgru/core/fep_gru_cell.hpp>
#include <unordered_map>

namespace crlgru {

/// @brief 自由エネルギー原理に基づくGRUネットワーク
/// @details 実装は src/core/fep_gru_network.cpp にあります
class FEPGRUNetwork : public torch::nn::Module {
public:
    /// @brief 設定構造体の型エイリアス
    using NetworkConfig = FEPGRUNetworkConfig;

private:
    NetworkConfig config_;
    
    // 多層GRUセル
    std::vector<std::shared_ptr<FEPGRUCell>> layers_;
    
    // ドロップアウト層
    torch::nn::Dropout dropout_{nullptr};
    
    // エージェント管理
    std::unordered_map<int, double> agent_performance_;
    std::unordered_map<int, std::vector<torch::Tensor>> agent_state_history_;
    
    // 名前変更の一時対応
    std::unordered_map<int, double>& agent_performance_history_ = agent_performance_;
    std::unordered_map<int, std::vector<torch::Tensor>>& agent_states_ = agent_state_history_;
    
    // 集合自由エネルギー
    torch::Tensor collective_free_energy_;
    std::vector<torch::Tensor> layer_outputs_;
    std::vector<torch::Tensor> layer_predictions_;
    std::vector<torch::Tensor> layer_free_energies_;

public:
    /// @brief コンストラクタ
    /// @param config ネットワーク設定
    explicit FEPGRUNetwork(const NetworkConfig& config);

    /// @brief フォワードパス
    /// @param sequence 入力シーケンス [batch_size, seq_len, input_size]
    /// @return (出力, 予測, 自由エネルギー)
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(
        const torch::Tensor& sequence
    );

    /// @brief エージェント間パラメータ共有
    /// @param agent_ids 共有対象エージェントID
    void share_parameters_with_agents(const std::vector<int>& agent_ids);

    /// @brief 集合自由エネルギー計算
    /// @return 集合自由エネルギー
    torch::Tensor compute_collective_free_energy() const;

    /// @brief 多層SOM特徴抽出
    /// @return 各層のSOM特徴ベクトル
    std::vector<torch::Tensor> extract_multi_layer_som_features() const;

    /// @brief エージェント登録
    /// @param agent_id エージェントID
    /// @param initial_performance 初期性能
    void register_agent(int agent_id, double initial_performance);

    /// @brief エージェント性能更新
    /// @param agent_id エージェントID
    /// @param performance 性能値
    void update_agent_performance(int agent_id, double performance);

    /// @brief エージェント削除
    /// @param agent_id エージェントID
    void remove_agent(int agent_id);

    /// @brief 設定取得
    const NetworkConfig& get_config() const { return config_; }

    /// @brief レイヤー数取得
    size_t get_num_layers() const { return layers_.size(); }

    /// @brief 特定レイヤー取得
    /// @param layer_idx レイヤーインデックス
    std::shared_ptr<FEPGRUCell> get_layer(size_t layer_idx) const;

private:
    /// @brief レイヤー初期化
    void initialize_layers();
    
    /// @brief 階層的処理
    torch::Tensor process_layer(
        size_t layer_idx,
        const torch::Tensor& input,
        const torch::Tensor& hidden
    );
};

} // namespace crlgru

#endif // CRLGRU_CORE_FEP_GRU_NETWORK_HPP
