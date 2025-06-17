/**
 * @file embodied_fep_gru_cell.hpp
 * @brief 身体性FEP-GRUセル（基本版実装）
 * @author 五十嵐研究室
 * @date 2025年6月
 *
 * 物理制約を考慮した身体性AIのためのFEP-GRUセル
 */

#ifndef CRLGRU_EMBODIED_FEP_GRU_CELL_HPP
#define CRLGRU_EMBODIED_FEP_GRU_CELL_HPP

#include <torch/torch.h>
#include <memory>
#include <vector>
#include "fep_gru_cell.hpp"
#include "../utils/config_types.hpp"

namespace crlgru {
namespace core {

/// @brief 身体性FEP-GRUセル基本版
/// @details 物理制約と身体性を考慮したFEP-GRUセルの拡張
class EmbodiedFEPGRUCell : public FEPGRUCell {
public:
    /// @brief 身体性設定構造体
    struct EmbodiedConfig {
        // 物理パラメータ
        double mass = 1.0;                     ///< 質量 [kg]
        double inertia = 0.1;                  ///< 慣性モーメント [kg⋅m²]
        double max_force = 10.0;               ///< 最大制御力 [N]
        double friction_coefficient = 0.1;     ///< 摩擦係数
        
        // センサーパラメータ
        double sensor_noise_variance = 0.01;   ///< センサーノイズ分散
        double measurement_delay = 0.033;      ///< 測定遅延 [s] (30FPS)
        bool enable_kalman_filter = true;     ///< カルマンフィルタ有効
        
        // 身体性統合パラメータ
        double physical_constraint_weight = 0.5;  ///< 物理制約重み
        double embodiment_learning_rate = 0.01;   ///< 身体性学習率
        int state_history_length = 10;           ///< 状態履歴長
        
        // 極座標統合パラメータ
        int polar_feature_dim = 108;             ///< 極座標特徴次元
        double polar_map_range = 10.0;           ///< 極座標マップ範囲
        int num_distance_rings = 8;              ///< 距離リング数
        int num_angle_sectors = 16;              ///< 角度セクター数
    };

    /// @brief 物理状態構造体
    struct PhysicalState {
        torch::Tensor position;        ///< 位置 [x, y]
        torch::Tensor velocity;        ///< 速度 [vx, vy]
        torch::Tensor acceleration;    ///< 加速度 [ax, ay]
        torch::Tensor orientation;     ///< 向き [theta]
        torch::Tensor angular_velocity; ///< 角速度 [omega]
        double timestamp = 0.0;        ///< タイムスタンプ
    };

private:
    EmbodiedConfig embodied_config_;
    
    // 物理制約レイヤー
    torch::nn::Linear physical_constraint_layer_{nullptr};
    torch::nn::Linear force_prediction_layer_{nullptr};
    
    // センサーモデル
    torch::nn::Linear sensor_noise_model_{nullptr};
    torch::nn::GRU kalman_filter_gru_{nullptr};
    
    // 身体性統合モジュール
    torch::nn::Linear embodiment_integration_{nullptr};
    torch::nn::Linear polar_feature_processor_{nullptr};
    
    // 状態管理
    PhysicalState current_physical_state_;
    std::vector<PhysicalState> state_history_;
    torch::Tensor kalman_state_;
    
    // 内部バッファ（最適化用）
    torch::Tensor constraint_buffer_;
    torch::Tensor force_buffer_;
    torch::Tensor sensor_buffer_;
    torch::Tensor polar_buffer_;

public:
    /// @brief コンストラクタ
    /// @param config 基本FEP-GRU設定
    /// @param embodied_config 身体性設定
    explicit EmbodiedFEPGRUCell(const Config& config);
    explicit EmbodiedFEPGRUCell(const Config& config, 
                               const EmbodiedConfig& embodied_config);

    /// @brief 身体性フォワードパス
    /// @param input 入力テンソル [batch_size, input_size]
    /// @param hidden 隠れ状態 [batch_size, hidden_size]
    /// @param physical_state 現在の物理状態
    /// @param neighbor_positions 近隣エージェント位置 [num_neighbors, 2]
    /// @return (新しい隠れ状態, 予測, 自由エネルギー, 制御力, 更新された物理状態)
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, PhysicalState> 
    forward_embodied(const torch::Tensor& input,
                    const torch::Tensor& hidden,
                    const PhysicalState& physical_state,
                    const torch::Tensor& neighbor_positions = torch::Tensor{});

    /// @brief 物理制約適用
    /// @param raw_force 生の制御力
    /// @param physical_state 現在の物理状態
    /// @return 制約された制御力
    torch::Tensor apply_physical_constraints(const torch::Tensor& raw_force,
                                            const PhysicalState& physical_state);

    /// @brief センサーノイズシミュレーション
    /// @param true_state 真の状態
    /// @return ノイズを含む観測
    torch::Tensor simulate_sensor_observation(const torch::Tensor& true_state);

    /// @brief 極座標特徴変換
    /// @param agent_position 自身の位置
    /// @param neighbor_positions 近隣位置
    /// @return 極座標特徴 [polar_feature_dim]
    torch::Tensor convert_to_polar_features(const torch::Tensor& agent_position,
                                           const torch::Tensor& neighbor_positions);

    /// @brief 物理状態更新
    /// @param control_force 制御力
    /// @param dt 時間ステップ
    void update_physical_state(const torch::Tensor& control_force, double dt = 0.033);

    /// @brief カルマンフィルタ更新
    /// @param observation 観測値
    /// @return フィルタ済み状態
    torch::Tensor update_kalman_filter(const torch::Tensor& observation);

    /// @brief 身体性学習の更新
    /// @param prediction_error 予測誤差
    /// @param physical_error 物理誤差
    void update_embodiment_learning(const torch::Tensor& prediction_error,
                                  const torch::Tensor& physical_error);

    /// @brief 現在の物理状態取得
    /// @return 物理状態
    const PhysicalState& get_physical_state() const { return current_physical_state_; }

    /// @brief 物理状態設定
    /// @param state 新しい物理状態
    void set_physical_state(const PhysicalState& state);

    /// @brief 状態履歴取得
    /// @return 状態履歴
    const std::vector<PhysicalState>& get_state_history() const { return state_history_; }

    /// @brief リセット（親クラスのリセットも呼び出し）
    void reset_states() override;

private:
    /// @brief モジュール初期化
    void initialize_embodied_modules();

    /// @brief バッファ初期化
    void initialize_buffers();

    /// @brief 状態履歴更新
    void update_state_history(const PhysicalState& new_state);

    /// @brief 物理パラメータ検証
    void validate_physical_parameters() const;
};

/// @brief 身体性FEP-GRUセルのファクトリ関数
/// @param config 基本設定
/// @param embodied_config 身体性設定
/// @return 身体性FEP-GRUセルの共有ポインタ
std::shared_ptr<EmbodiedFEPGRUCell> create_embodied_fep_gru_cell(
    const FEPGRUCell::Config& config
);

std::shared_ptr<EmbodiedFEPGRUCell> create_embodied_fep_gru_cell(
    const FEPGRUCell::Config& config,
    const EmbodiedFEPGRUCell::EmbodiedConfig& embodied_config
);

} // namespace core
} // namespace crlgru

#endif // CRLGRU_EMBODIED_FEP_GRU_CELL_HPP