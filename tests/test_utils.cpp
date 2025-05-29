#include "crlgru/crl_gru.hpp"
#include <iostream>
#include <cassert>
#include <cmath>
#include <fstream>

bool test_cartesian_to_polar_conversion() {
    try {
        // Test data: 3 agents in 2D space
        auto positions = torch::tensor({{{1.0, 0.0}, {0.0, 1.0}, {-1.0, -1.0}}});
        auto self_position = torch::tensor({{0.0, 0.0}});
        
        auto polar_map = crlgru::utils::cartesian_to_polar_map(
            positions, self_position, 4, 8, 5.0);
        
        // Check output shape
        assert(polar_map.size(0) == 1); // batch size
        assert(polar_map.size(1) == 4); // num rings
        assert(polar_map.size(2) == 8); // num sectors
        
        // Should have 3 agents total in the map
        auto total_agents = polar_map.sum().item<double>();
        assert(std::abs(total_agents - 3.0) < 1e-6);
        
        std::cout << "✓ Cartesian to polar conversion test passed" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "✗ Cartesian to polar conversion test failed: " << e.what() << std::endl;
        return false;
    }
}

bool test_mutual_information_computation() {
    try {
        // Test with correlated and uncorrelated states
        auto state1 = torch::randn({100});
        auto state2 = torch::randn({100});
        auto correlated_state2 = 0.8 * state1 + 0.2 * torch::randn({100});
        
        auto mi_uncorrelated = crlgru::utils::compute_mutual_information(state1, state2);
        auto mi_correlated = crlgru::utils::compute_mutual_information(state1, correlated_state2);
        
        // Correlated states should have higher mutual information
        assert(mi_correlated > mi_uncorrelated);
        assert(mi_uncorrelated >= 0.0);
        assert(mi_correlated >= 0.0);
        
        std::cout << "✓ Mutual information computation test passed (uncorr: " 
                  << mi_uncorrelated << ", corr: " << mi_correlated << ")" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "✗ Mutual information computation test failed: " << e.what() << std::endl;
        return false;
    }
}

bool test_gaussian_kernel_application() {
    try {
        // Create a test image with a sharp peak
        auto test_image = torch::zeros({32, 32});
        test_image[16][16] = 1.0; // Sharp peak in center
        
        auto smoothed_image = crlgru::utils::apply_gaussian_kernel(test_image, 2.0, 7);
        
        // Check output shape
        assert(smoothed_image.size(0) == 32);
        assert(smoothed_image.size(1) == 32);
        
        // Should be smooth (center value less than 1, but neighboring values > 0)
        assert(smoothed_image[16][16].item<double>() < 1.0);
        assert(smoothed_image[16][16].item<double>() > 0.5);
        assert(smoothed_image[15][16].item<double>() > 0.0);
        assert(smoothed_image[17][16].item<double>() > 0.0);
        
        // Total energy should be approximately preserved
        auto original_sum = test_image.sum().item<double>();
        auto smoothed_sum = smoothed_image.sum().item<double>();
        assert(std::abs(original_sum - smoothed_sum) < 0.1);
        
        std::cout << "✓ Gaussian kernel application test passed" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "✗ Gaussian kernel application test failed: " << e.what() << std::endl;
        return false;
    }
}

bool test_trust_metric_computation() {
    try {
        // Test different performance scenarios
        std::vector<double> high_performance = {0.9, 0.85, 0.92, 0.88, 0.90};
        std::vector<double> low_performance = {0.2, 0.3, 0.1, 0.25, 0.15};
        std::vector<double> variable_performance = {0.1, 0.9, 0.2, 0.8, 0.3};
        
        double close_distance = 2.0;
        double far_distance = 8.0;
        double max_distance = 10.0;
        
        auto trust_high_close = crlgru::utils::compute_trust_metric(
            high_performance, close_distance, max_distance);
        auto trust_high_far = crlgru::utils::compute_trust_metric(
            high_performance, far_distance, max_distance);
        auto trust_low_close = crlgru::utils::compute_trust_metric(
            low_performance, close_distance, max_distance);
        auto trust_variable_close = crlgru::utils::compute_trust_metric(
            variable_performance, close_distance, max_distance);
        
        // Trust should be in [0, 1] range
        assert(trust_high_close >= 0.0 && trust_high_close <= 1.0);
        assert(trust_high_far >= 0.0 && trust_high_far <= 1.0);
        assert(trust_low_close >= 0.0 && trust_low_close <= 1.0);
        assert(trust_variable_close >= 0.0 && trust_variable_close <= 1.0);
        
        // High performance + close distance should have highest trust
        assert(trust_high_close > trust_high_far);
        assert(trust_high_close > trust_low_close);
        assert(trust_high_close > trust_variable_close);
        
        std::cout << "✓ Trust metric computation test passed" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "✗ Trust metric computation test failed: " << e.what() << std::endl;
        return false;
    }
}

bool test_parameter_save_load() {
    try {
        // Create test parameters
        std::unordered_map<std::string, torch::Tensor> original_params;
        original_params["weights"] = torch::randn({64, 32});
        original_params["bias"] = torch::randn({32});
        original_params["embeddings"] = torch::randn({100, 16});
        original_params["small_tensor"] = torch::tensor({1.0, 2.0, 3.0});
        
        std::string filename = "test_params_utils.bin";
        
        // Save parameters
        crlgru::utils::save_parameters(filename, original_params);
        
        // Load parameters
        auto loaded_params = crlgru::utils::load_parameters(filename);
        
        // Check that all parameters were loaded correctly
        assert(loaded_params.size() == original_params.size());
        
        for (const auto& param_pair : original_params) {
            const std::string& param_name = param_pair.first;
            const torch::Tensor& original_tensor = param_pair.second;
            
            auto loaded_it = loaded_params.find(param_name);
            assert(loaded_it != loaded_params.end());
            
            const torch::Tensor& loaded_tensor = loaded_it->second;
            
            // Check shape
            assert(original_tensor.sizes() == loaded_tensor.sizes());
            
            // Check values (should be exactly equal)
            assert(torch::allclose(original_tensor, loaded_tensor, 1e-6));
        }
        
        // Clean up test file
        std::remove(filename.c_str());
        
        std::cout << "✓ Parameter save/load test passed" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "✗ Parameter save/load test failed: " << e.what() << std::endl;
        return false;
    }
}

bool test_polar_spatial_attention() {
    try {
        crlgru::PolarSpatialAttention::AttentionConfig config;
        config.input_channels = 32;
        config.attention_dim = 16;
        config.num_distance_rings = 4;
        config.num_angle_sectors = 8;
        
        auto attention = std::make_shared<crlgru::PolarSpatialAttention>(config);
        
        auto polar_map = torch::randn({2, 32, 4, 8}); // [batch, channels, rings, sectors]
        auto attended_map = attention->forward(polar_map);
        
        // Check output shape (should be same as input)
        assert(attended_map.size(0) == 2);
        assert(attended_map.size(1) == 32);
        assert(attended_map.size(2) == 4);
        assert(attended_map.size(3) == 8);
        
        // Check that attention weights are computed
        auto [distance_weights, angle_weights] = attention->compute_attention_weights(polar_map);
        
        assert(distance_weights.size(0) == 2); // batch size
        assert(distance_weights.size(1) == 4); // rings
        assert(angle_weights.size(0) == 2);   // batch size
        assert(angle_weights.size(1) == 8);   // sectors
        
        // Attention weights should sum to 1 across spatial dimensions
        auto distance_sums = distance_weights.sum(1);
        auto angle_sums = angle_weights.sum(1);
        
        for (int b = 0; b < 2; ++b) {
            assert(std::abs(distance_sums[b].item<double>() - 1.0) < 1e-5);
            assert(std::abs(angle_sums[b].item<double>() - 1.0) < 1e-5);
        }
        
        std::cout << "✓ Polar spatial attention test passed" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "✗ Polar spatial attention test failed: " << e.what() << std::endl;
        return false;
    }
}

bool test_meta_evaluator() {
    try {
        crlgru::MetaEvaluator::EvaluationConfig config;
        config.objective_weights = {1.0, 0.8, 0.6, 0.4}; // goal, collision, cohesion, alignment
        config.adaptive_weights = true;
        
        auto evaluator = std::make_shared<crlgru::MetaEvaluator>(config);
        
        // Create test predicted states: [batch, time, features]
        // Features: [pos_x, pos_y, goal_x, goal_y, vel_x, vel_y, ...]
        auto predicted_states = torch::randn({1, 10, 8});
        auto current_state = torch::randn({1, 8});
        auto environment_state = torch::randn({1, 10});
        
        auto score = evaluator->evaluate(predicted_states, current_state, environment_state);
        
        // Score should be finite
        assert(std::isfinite(score));
        
        // Test adding custom objective
        evaluator->add_objective([](const torch::Tensor& states) -> double {
            return states.mean().item<double>(); // Simple custom objective
        });
        
        auto score_with_custom = evaluator->evaluate(predicted_states, current_state, environment_state);
        assert(std::isfinite(score_with_custom));
        
        // Test adaptive weights
        std::vector<double> performance_history = {0.7, 0.8, 0.6, 0.9, 0.75};
        evaluator->adapt_weights(performance_history);
        
        auto score_after_adaptation = evaluator->evaluate(predicted_states, current_state, environment_state);
        assert(std::isfinite(score_after_adaptation));
        
        std::cout << "✓ Meta evaluator test passed" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "✗ Meta evaluator test failed: " << e.what() << std::endl;
        return false;
    }
}

bool test_edge_cases() {
    try {
        // Test empty performance history
        std::vector<double> empty_performance;
        auto trust_empty = crlgru::utils::compute_trust_metric(empty_performance, 1.0, 10.0);
        assert(trust_empty == 0.0);
        
        // Test zero distance
        std::vector<double> good_performance = {0.8, 0.9};
        auto trust_zero_dist = crlgru::utils::compute_trust_metric(good_performance, 0.0, 10.0);
        assert(trust_zero_dist > 0.0);
        
        // Test maximum distance
        auto trust_max_dist = crlgru::utils::compute_trust_metric(good_performance, 10.0, 10.0);
        assert(trust_max_dist >= 0.0);
        
        // Test single value tensors
        auto single_state1 = torch::tensor({1.0});
        auto single_state2 = torch::tensor({2.0});
        auto mi_single = crlgru::utils::compute_mutual_information(single_state1, single_state2);
        assert(std::isfinite(mi_single));
        
        std::cout << "✓ Edge cases test passed" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "✗ Edge cases test failed: " << e.what() << std::endl;
        return false;
    }
}

int main() {
    std::cout << "=== Utility Functions Tests ===" << std::endl;
    
    int passed = 0;
    int total = 8;
    
    if (test_cartesian_to_polar_conversion()) passed++;
    if (test_mutual_information_computation()) passed++;
    if (test_gaussian_kernel_application()) passed++;
    if (test_trust_metric_computation()) passed++;
    if (test_parameter_save_load()) passed++;
    if (test_polar_spatial_attention()) passed++;
    if (test_meta_evaluator()) passed++;
    if (test_edge_cases()) passed++;
    
    std::cout << "\n=== Test Results ===" << std::endl;
    std::cout << "Passed: " << passed << "/" << total << std::endl;
    
    if (passed == total) {
        std::cout << "All tests passed! ✓" << std::endl;
        return 0;
    } else {
        std::cout << "Some tests failed! ✗" << std::endl;
        return 1;
    }
}
