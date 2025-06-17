/**
 * @file embodied_fep_gru_cell.cpp
 * @brief 身体性FEP-GRUセル実装
 * @author 五十嵐研究室
 * @date 2025年6月
 */

#include "crlgru/core/embodied_fep_gru_cell.hpp"
#include "crlgru/utils/math_utils.hpp"
#include "crlgru/utils/spatial_transforms.hpp"
#include <stdexcept>
#include <algorithm>
#include <cmath>

namespace crlgru {
namespace core {

EmbodiedFEPGRUCell::EmbodiedFEPGRUCell(const Config& config)
    : FEPGRUCell(config), embodied_config_{} {
    
    validate_physical_parameters();
    initialize_embodied_modules();
    initialize_buffers();
    
    // 物理状態初期化
    current_physical_state_.position = torch::zeros({2});
    current_physical_state_.velocity = torch::zeros({2});
    current_physical_state_.acceleration = torch::zeros({2});
    current_physical_state_.orientation = torch::zeros({1});
    current_physical_state_.angular_velocity = torch::zeros({1});
    current_physical_state_.timestamp = 0.0;
    
    // カルマンフィルタ状態初期化
    kalman_state_ = torch::zeros({1, config.hidden_size});
}

EmbodiedFEPGRUCell::EmbodiedFEPGRUCell(const Config& config, 
                                       const EmbodiedConfig& embodied_config)
    : FEPGRUCell(config), embodied_config_(embodied_config) {
    
    validate_physical_parameters();
    initialize_embodied_modules();
    initialize_buffers();
    
    // 物理状態初期化
    current_physical_state_.position = torch::zeros({2});
    current_physical_state_.velocity = torch::zeros({2});
    current_physical_state_.acceleration = torch::zeros({2});
    current_physical_state_.orientation = torch::zeros({1});
    current_physical_state_.angular_velocity = torch::zeros({1});
    current_physical_state_.timestamp = 0.0;
    
    // カルマンフィルタ状態初期化
    kalman_state_ = torch::zeros({1, config.hidden_size});
}

void EmbodiedFEPGRUCell::initialize_embodied_modules() {
    auto& config = get_config();
    
    // 物理制約レイヤー (hidden_size -> 4: [fx, fy, valid_force_magnitude, constraint_violation])
    physical_constraint_layer_ = register_module(
        "physical_constraint_layer",
        torch::nn::Linear(config.hidden_size, 4)
    );
    
    // 制御力予測レイヤー (hidden_size -> 2: [fx, fy])
    force_prediction_layer_ = register_module(
        "force_prediction_layer", 
        torch::nn::Linear(config.hidden_size, 2)
    );
    
    // センサーノイズモデル (物理状態次元6 -> hidden_size/2)
    sensor_noise_model_ = register_module(
        "sensor_noise_model",
        torch::nn::Linear(6, config.hidden_size / 2)
    );
    
    // カルマンフィルタGRU
    kalman_filter_gru_ = register_module(
        "kalman_filter_gru",
        torch::nn::GRU(torch::nn::GRUOptions(6, config.hidden_size / 4).batch_first(true))
    );
    
    // 身体性統合レイヤー
    embodiment_integration_ = register_module(
        "embodiment_integration",
        torch::nn::Linear(config.hidden_size + embodied_config_.polar_feature_dim, config.hidden_size)
    );
    
    // 極座標特徴処理レイヤー
    polar_feature_processor_ = register_module(
        "polar_feature_processor",
        torch::nn::Linear(embodied_config_.polar_feature_dim, embodied_config_.polar_feature_dim / 2)
    );
}

void EmbodiedFEPGRUCell::initialize_buffers() {
    auto& config = get_config();
    
    constraint_buffer_ = torch::zeros({1, 4});
    force_buffer_ = torch::zeros({1, 2});
    sensor_buffer_ = torch::zeros({1, config.hidden_size / 2});
    polar_buffer_ = torch::zeros({1, embodied_config_.polar_feature_dim});
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, EmbodiedFEPGRUCell::PhysicalState>
EmbodiedFEPGRUCell::forward_embodied(const torch::Tensor& input,
                                    const torch::Tensor& hidden,
                                    const PhysicalState& physical_state,
                                    const torch::Tensor& neighbor_positions) {
    
    // 1. 基本FEP-GRUフォワードパス
    auto [new_hidden, prediction, free_energy] = forward(input, hidden);
    
    // 2. 制御力予測
    auto raw_force = force_prediction_layer_->forward(new_hidden);
    
    // 3. 物理制約適用
    auto constrained_force = apply_physical_constraints(raw_force, physical_state);
    
    // 4. センサー観測シミュレーション
    auto physical_state_tensor = torch::cat({
        physical_state.position,
        physical_state.velocity, 
        physical_state.acceleration
    });
    auto sensor_observation = simulate_sensor_observation(physical_state_tensor);
    
    // 5. カルマンフィルタ更新
    auto filtered_state = update_kalman_filter(sensor_observation);
    
    // 6. 極座標特徴変換
    torch::Tensor polar_features;
    if (neighbor_positions.defined() && neighbor_positions.numel() > 0) {
        polar_features = convert_to_polar_features(physical_state.position, neighbor_positions);
    } else {
        polar_features = torch::zeros({embodied_config_.polar_feature_dim});
    }
    
    // 7. 身体性統合
    auto combined_features = torch::cat({new_hidden.squeeze(0), polar_features});
    auto embodied_hidden = embodiment_integration_->forward(combined_features.unsqueeze(0));
    
    // 8. 物理状態更新
    PhysicalState updated_state = physical_state;
    auto force_2d = constrained_force.squeeze(0);
    
    // 簡単な物理シミュレーション
    double dt = 0.033; // 30FPS
    auto acceleration = force_2d / embodied_config_.mass;
    updated_state.acceleration = acceleration;
    updated_state.velocity = physical_state.velocity + acceleration * dt;
    updated_state.position = physical_state.position + updated_state.velocity * dt;
    updated_state.timestamp = physical_state.timestamp + dt;
    
    // 摩擦適用
    updated_state.velocity *= (1.0 - embodied_config_.friction_coefficient * dt);
    
    // 状態履歴更新
    update_state_history(updated_state);
    current_physical_state_ = updated_state;
    
    // 身体性自由エネルギー計算（物理制約違反ペナルティ）
    auto constraint_info = physical_constraint_layer_->forward(embodied_hidden);
    auto constraint_violation = constraint_info.select(-1, 3).abs().mean();
    auto embodied_free_energy = free_energy + 
                               embodied_config_.physical_constraint_weight * constraint_violation;
    
    return std::make_tuple(embodied_hidden, prediction, embodied_free_energy, 
                          constrained_force, updated_state);
}

torch::Tensor EmbodiedFEPGRUCell::apply_physical_constraints(const torch::Tensor& raw_force,
                                                           const PhysicalState& physical_state) {
    
    // 最大力制限
    auto force_magnitude = torch::norm(raw_force, 2, -1, true);
    auto scale_factor = torch::min(
        torch::ones_like(force_magnitude),
        embodied_config_.max_force / (force_magnitude + 1e-8)
    );
    
    auto constrained_force = raw_force * scale_factor;
    
    // 速度依存摩擦
    auto velocity_magnitude = torch::norm(physical_state.velocity);
    auto friction_force = -physical_state.velocity * embodied_config_.friction_coefficient * velocity_magnitude;
    
    // 摩擦を考慮した最終制御力
    constrained_force += friction_force.unsqueeze(0);
    
    return constrained_force;
}

torch::Tensor EmbodiedFEPGRUCell::simulate_sensor_observation(const torch::Tensor& true_state) {
    // センサーノイズ追加
    auto noise = torch::randn_like(true_state) * std::sqrt(embodied_config_.sensor_noise_variance);
    auto noisy_observation = true_state + noise;
    
    // 測定遅延シミュレーション（簡易版：前の状態との線形補間）
    if (!state_history_.empty()) {
        auto delay_factor = embodied_config_.measurement_delay / 0.033; // フレーム数での遅延
        if (delay_factor > 0.5) {
            auto prev_state_tensor = torch::cat({
                state_history_.back().position,
                state_history_.back().velocity,
                state_history_.back().acceleration
            });
            noisy_observation = (1.0 - delay_factor) * noisy_observation + delay_factor * prev_state_tensor;
        }
    }
    
    return noisy_observation;
}

torch::Tensor EmbodiedFEPGRUCell::convert_to_polar_features(const torch::Tensor& agent_position,
                                                          const torch::Tensor& neighbor_positions) {
    
    if (!neighbor_positions.defined() || neighbor_positions.numel() == 0) {
        return torch::zeros({embodied_config_.polar_feature_dim});
    }
    
    // 極座標変換を使用
    auto polar_map = utils::cartesian_to_polar_map(
        neighbor_positions.unsqueeze(0), // [1, num_neighbors, 2]
        agent_position.unsqueeze(0),     // [1, 2]
        embodied_config_.num_distance_rings,
        embodied_config_.num_angle_sectors,
        embodied_config_.polar_map_range
    );
    
    // 極座標マップをフラット化して特徴ベクトルに変換
    auto flattened_map = polar_map.reshape({-1});
    
    // 指定次元にリサイズ
    if (flattened_map.size(0) != embodied_config_.polar_feature_dim) {
        if (flattened_map.size(0) > embodied_config_.polar_feature_dim) {
            // トランケート
            flattened_map = flattened_map.slice(0, 0, embodied_config_.polar_feature_dim);
        } else {
            // パディング
            auto padding_size = embodied_config_.polar_feature_dim - flattened_map.size(0);
            auto padding = torch::zeros({padding_size});
            flattened_map = torch::cat({flattened_map, padding});
        }
    }
    
    // 特徴処理レイヤーを通す
    auto processed_features = polar_feature_processor_->forward(flattened_map.unsqueeze(0));
    
    // 最終的な極座標特徴（パディングで目標次元に調整）
    auto final_features = torch::zeros({embodied_config_.polar_feature_dim});
    auto copy_size = std::min(processed_features.size(-1), (int64_t)embodied_config_.polar_feature_dim);
    final_features.slice(0, 0, copy_size).copy_(processed_features.squeeze(0).slice(0, 0, copy_size));
    
    return final_features;
}

void EmbodiedFEPGRUCell::update_physical_state(const torch::Tensor& control_force, double dt) {
    // 運動方程式による状態更新
    auto acceleration = control_force / embodied_config_.mass;
    
    current_physical_state_.acceleration = acceleration;
    current_physical_state_.velocity += acceleration * dt;
    current_physical_state_.position += current_physical_state_.velocity * dt;
    current_physical_state_.timestamp += dt;
    
    // 摩擦適用
    current_physical_state_.velocity *= (1.0 - embodied_config_.friction_coefficient * dt);
    
    update_state_history(current_physical_state_);
}

torch::Tensor EmbodiedFEPGRUCell::update_kalman_filter(const torch::Tensor& observation) {
    if (!embodied_config_.enable_kalman_filter) {
        return observation;
    }
    
    // GRUベースの簡易カルマンフィルタ
    auto obs_input = observation.unsqueeze(0).unsqueeze(0); // [1, 1, obs_dim]
    auto [output, new_hidden] = kalman_filter_gru_->forward(obs_input, kalman_state_);
    kalman_state_ = new_hidden;
    
    return output.squeeze(0).squeeze(0);
}

void EmbodiedFEPGRUCell::update_embodiment_learning(const torch::Tensor& prediction_error,
                                                  const torch::Tensor& physical_error) {
    // 身体性学習の重み更新（簡易版）
    auto combined_error = prediction_error.mean() + 
                         embodied_config_.physical_constraint_weight * physical_error.mean();
    
    // パラメータの勾配を調整
    for (auto& param : parameters()) {
        if (param.grad().defined()) {
            auto scale = 1.0 + embodied_config_.embodiment_learning_rate * combined_error.item<double>();
            param.grad().mul_(scale);
        }
    }
}

void EmbodiedFEPGRUCell::set_physical_state(const PhysicalState& state) {
    current_physical_state_ = state;
    update_state_history(state);
}

void EmbodiedFEPGRUCell::reset_states() {
    // 親クラスのリセット
    FEPGRUCell::reset_states();
    
    // 身体性状態のリセット
    current_physical_state_.position = torch::zeros({2});
    current_physical_state_.velocity = torch::zeros({2});
    current_physical_state_.acceleration = torch::zeros({2});
    current_physical_state_.orientation = torch::zeros({1});
    current_physical_state_.angular_velocity = torch::zeros({1});
    current_physical_state_.timestamp = 0.0;
    
    state_history_.clear();
    kalman_state_ = torch::zeros({1, get_config().hidden_size});
}

void EmbodiedFEPGRUCell::update_state_history(const PhysicalState& new_state) {
    state_history_.push_back(new_state);
    
    // 履歴長制限
    if (static_cast<int>(state_history_.size()) > embodied_config_.state_history_length) {
        state_history_.erase(state_history_.begin());
    }
}

void EmbodiedFEPGRUCell::validate_physical_parameters() const {
    if (embodied_config_.mass <= 0.0) {
        throw std::invalid_argument("Mass must be positive");
    }
    if (embodied_config_.inertia <= 0.0) {
        throw std::invalid_argument("Inertia must be positive");
    }
    if (embodied_config_.max_force <= 0.0) {
        throw std::invalid_argument("Maximum force must be positive");
    }
    if (embodied_config_.friction_coefficient < 0.0 || embodied_config_.friction_coefficient > 1.0) {
        throw std::invalid_argument("Friction coefficient must be in [0, 1]");
    }
    if (embodied_config_.sensor_noise_variance < 0.0) {
        throw std::invalid_argument("Sensor noise variance must be non-negative");
    }
}

std::shared_ptr<EmbodiedFEPGRUCell> create_embodied_fep_gru_cell(
    const FEPGRUCell::Config& config) {
    return std::make_shared<EmbodiedFEPGRUCell>(config);
}

std::shared_ptr<EmbodiedFEPGRUCell> create_embodied_fep_gru_cell(
    const FEPGRUCell::Config& config,
    const EmbodiedFEPGRUCell::EmbodiedConfig& embodied_config) {
    return std::make_shared<EmbodiedFEPGRUCell>(config, embodied_config);
}

} // namespace core
} // namespace crlgru