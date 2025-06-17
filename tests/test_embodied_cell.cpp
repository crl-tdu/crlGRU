/**
 * @file test_embodied_cell.cpp
 * @brief 身体性FEP-GRUセルのテスト
 * @author 五十嵐研究室
 * @date 2025年6月
 */

#include <iostream>
#include <cassert>
#include <cmath>
#include <torch/torch.h>
#include "crlgru/core/embodied_fep_gru_cell.hpp"

using namespace crlgru::core;

void test_embodied_cell_construction() {
    std::cout << "Test: Embodied cell construction - ";
    
    FEPGRUCell::Config config;
    config.input_size = 32;
    config.hidden_size = 64;
    
    EmbodiedFEPGRUCell::EmbodiedConfig embodied_config;
    embodied_config.mass = 2.0;
    embodied_config.max_force = 15.0;
    
    try {
        auto cell = std::make_shared<EmbodiedFEPGRUCell>(config, embodied_config);
        assert(cell != nullptr);
        std::cout << "PASSED" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "FAILED: " << e.what() << std::endl;
    }
}

void test_embodied_forward_pass() {
    std::cout << "Test: Embodied forward pass - ";
    
    FEPGRUCell::Config config;
    config.input_size = 32;
    config.hidden_size = 64;
    
    EmbodiedFEPGRUCell::EmbodiedConfig embodied_config;
    
    auto cell = std::make_shared<EmbodiedFEPGRUCell>(config, embodied_config);
    
    auto input = torch::randn({1, 32});
    auto hidden = torch::zeros({1, 64});
    
    EmbodiedFEPGRUCell::PhysicalState state;
    state.position = torch::tensor({0.0, 0.0});
    state.velocity = torch::tensor({1.0, 0.5});
    state.acceleration = torch::zeros({2});
    state.orientation = torch::zeros({1});
    state.angular_velocity = torch::zeros({1});
    
    auto neighbor_positions = torch::tensor({{2.0, 1.0}, {-1.0, 3.0}});
    
    try {
        auto [new_hidden, prediction, free_energy, control_force, updated_state] = 
            cell->forward_embodied(input, hidden, state, neighbor_positions);
        
        assert(new_hidden.size(0) == 1);
        assert(new_hidden.size(1) == 64);
        assert(control_force.size(0) == 1);
        assert(control_force.size(1) == 2);
        assert(free_energy.numel() > 0);
        
        std::cout << "PASSED" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "FAILED: " << e.what() << std::endl;
    }
}

void test_physical_constraints() {
    std::cout << "Test: Physical constraints - ";
    
    FEPGRUCell::Config config;
    config.input_size = 16;
    config.hidden_size = 32;
    
    EmbodiedFEPGRUCell::EmbodiedConfig embodied_config;
    embodied_config.max_force = 5.0;
    embodied_config.friction_coefficient = 0.1;
    
    auto cell = std::make_shared<EmbodiedFEPGRUCell>(config, embodied_config);
    
    // 過大な力をテスト
    auto large_force = torch::tensor({{10.0, 8.0}});
    
    EmbodiedFEPGRUCell::PhysicalState state;
    state.position = torch::zeros({2});
    state.velocity = torch::tensor({2.0, 1.0});
    state.acceleration = torch::zeros({2});
    state.orientation = torch::zeros({1});
    state.angular_velocity = torch::zeros({1});
    
    try {
        auto constrained_force = cell->apply_physical_constraints(large_force, state);
        auto force_magnitude = torch::norm(constrained_force).item<double>();
        
        // 最大力制限がかかっているかチェック
        assert(force_magnitude <= embodied_config.max_force + 1e-3);
        
        std::cout << "PASSED (Force limited to " << force_magnitude << ")" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "FAILED: " << e.what() << std::endl;
    }
}

void test_sensor_simulation() {
    std::cout << "Test: Sensor simulation - ";
    
    FEPGRUCell::Config config;
    config.input_size = 16;
    config.hidden_size = 32;
    
    EmbodiedFEPGRUCell::EmbodiedConfig embodied_config;
    embodied_config.sensor_noise_variance = 0.01;
    embodied_config.measurement_delay = 0.05;
    
    auto cell = std::make_shared<EmbodiedFEPGRUCell>(config, embodied_config);
    
    auto true_state = torch::tensor({1.0, 2.0, 0.5, -0.3, 0.1, 0.2});
    
    try {
        auto observation1 = cell->simulate_sensor_observation(true_state);
        auto observation2 = cell->simulate_sensor_observation(true_state);
        
        // ノイズがあるので2回の観測は異なるはず
        auto diff = torch::abs(observation1 - observation2).sum().item<double>();
        assert(diff > 1e-6);
        
        // 但し真値からそれほど離れていないはず
        auto error = torch::abs(observation1 - true_state).mean().item<double>();
        assert(error < 1.0); // 合理的な範囲
        
        std::cout << "PASSED (Mean error: " << error << ")" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "FAILED: " << e.what() << std::endl;
    }
}

void test_polar_features() {
    std::cout << "Test: Polar features - ";
    
    FEPGRUCell::Config config;
    config.input_size = 16;
    config.hidden_size = 32;
    
    EmbodiedFEPGRUCell::EmbodiedConfig embodied_config;
    embodied_config.polar_feature_dim = 128;
    embodied_config.num_distance_rings = 8;
    embodied_config.num_angle_sectors = 16;
    
    auto cell = std::make_shared<EmbodiedFEPGRUCell>(config, embodied_config);
    
    auto agent_pos = torch::tensor({0.0, 0.0});
    auto neighbor_positions = torch::tensor({
        {2.0, 0.0},   // 右
        {0.0, 3.0},   // 上
        {-1.0, -1.0}, // 左下
        {1.5, 1.5}    // 右上
    });
    
    try {
        auto features = cell->convert_to_polar_features(agent_pos, neighbor_positions);
        
        assert(features.size(0) == embodied_config.polar_feature_dim);
        assert(features.sum().item<double>() != 0.0); // 何らかの特徴があるはず
        
        // 空の近隣でもエラーにならないことを確認
        auto empty_features = cell->convert_to_polar_features(agent_pos, torch::Tensor{});
        assert(empty_features.size(0) == embodied_config.polar_feature_dim);
        
        std::cout << "PASSED" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "FAILED: " << e.what() << std::endl;
    }
}

void test_kalman_filter() {
    std::cout << "Test: Kalman filter - ";
    
    FEPGRUCell::Config config;
    config.input_size = 16;
    config.hidden_size = 32;
    
    EmbodiedFEPGRUCell::EmbodiedConfig embodied_config;
    embodied_config.enable_kalman_filter = true;
    
    auto cell = std::make_shared<EmbodiedFEPGRUCell>(config, embodied_config);
    
    auto observation = torch::randn({6}); // 6次元状態
    
    try {
        auto filtered1 = cell->update_kalman_filter(observation);
        auto filtered2 = cell->update_kalman_filter(observation + torch::randn({6}) * 0.1);
        
        assert(filtered1.size(0) == 6);
        assert(filtered2.size(0) == 6);
        
        // フィルタ無効でも動作することを確認
        embodied_config.enable_kalman_filter = false;
        auto cell_no_filter = std::make_shared<EmbodiedFEPGRUCell>(config, embodied_config);
        auto unfiltered = cell_no_filter->update_kalman_filter(observation);
        
        // フィルタ無効の場合は入力そのまま
        auto diff = torch::abs(unfiltered - observation).sum().item<double>();
        assert(diff < 1e-6);
        
        std::cout << "PASSED" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "FAILED: " << e.what() << std::endl;
    }
}

void test_state_management() {
    std::cout << "Test: State management - ";
    
    FEPGRUCell::Config config;
    config.input_size = 16;
    config.hidden_size = 32;
    
    EmbodiedFEPGRUCell::EmbodiedConfig embodied_config;
    embodied_config.state_history_length = 3;
    
    auto cell = std::make_shared<EmbodiedFEPGRUCell>(config, embodied_config);
    
    try {
        // 初期状態確認
        auto initial_state = cell->get_physical_state();
        assert(cell->get_state_history().size() == 0);
        
        // 状態設定
        EmbodiedFEPGRUCell::PhysicalState state1;
        state1.position = torch::tensor({1.0, 2.0});
        state1.velocity = torch::tensor({0.5, -0.3});
        state1.timestamp = 1.0;
        
        cell->set_physical_state(state1);
        assert(cell->get_state_history().size() == 1);
        
        // 複数の状態を追加
        for (int i = 0; i < 5; i++) {
            EmbodiedFEPGRUCell::PhysicalState state;
            state.position = torch::tensor({(double)i, (double)i});
            state.timestamp = i + 2.0;
            cell->set_physical_state(state);
        }
        
        // 履歴長制限の確認
        assert(static_cast<int>(cell->get_state_history().size()) == embodied_config.state_history_length);
        
        // リセット
        cell->reset_states();
        assert(cell->get_state_history().size() == 0);
        
        std::cout << "PASSED" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "FAILED: " << e.what() << std::endl;
    }
}

void test_parameter_validation() {
    std::cout << "Test: Parameter validation - ";
    
    FEPGRUCell::Config config;
    config.input_size = 16;
    config.hidden_size = 32;
    
    EmbodiedFEPGRUCell::EmbodiedConfig bad_config;
    
    int error_count = 0;
    
    // 負の質量
    bad_config.mass = -1.0;
    try {
        auto cell = std::make_shared<EmbodiedFEPGRUCell>(config, bad_config);
    } catch (const std::invalid_argument&) {
        error_count++;
    }
    bad_config.mass = 1.0; // 修正
    
    // 負の慣性
    bad_config.inertia = -0.1;
    try {
        auto cell = std::make_shared<EmbodiedFEPGRUCell>(config, bad_config);
    } catch (const std::invalid_argument&) {
        error_count++;
    }
    bad_config.inertia = 0.1; // 修正
    
    // 不正な摩擦係数
    bad_config.friction_coefficient = 1.5;
    try {
        auto cell = std::make_shared<EmbodiedFEPGRUCell>(config, bad_config);
    } catch (const std::invalid_argument&) {
        error_count++;
    }
    bad_config.friction_coefficient = 0.1; // 修正
    
    // 負のノイズ分散
    bad_config.sensor_noise_variance = -0.01;
    try {
        auto cell = std::make_shared<EmbodiedFEPGRUCell>(config, bad_config);
    } catch (const std::invalid_argument&) {
        error_count++;
    }
    
    if (error_count == 4) {
        std::cout << "PASSED (All invalid parameters caught)" << std::endl;
    } else {
        std::cout << "FAILED (Only " << error_count << "/4 errors caught)" << std::endl;
    }
}

int main() {
    std::cout << "=== Embodied FEP-GRU Cell Tests ===" << std::endl;
    
    test_embodied_cell_construction();
    test_embodied_forward_pass();
    test_physical_constraints();
    test_sensor_simulation();
    test_polar_features();
    test_kalman_filter();
    test_state_management();
    test_parameter_validation();
    
    std::cout << std::endl << "=== Embodied Tests Complete ===" << std::endl;
    
    return 0;
}