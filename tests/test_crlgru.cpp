#include <iostream>
#include <cassert>
#include <vector>
#include <memory>
#include <functional>
#include <random>

// crlGRU library include
#include <crlgru/crl_gru.hpp>

class TestRunner {
private:
    int total_tests_ = 0;
    int passed_tests_ = 0;
    std::string current_section_;

public:
    void start_section(const std::string& section_name) {
        current_section_ = section_name;
        std::cout << "\n=== " << section_name << " ===" << std::endl;
    }

    void run_test(const std::string& test_name, std::function<bool()> test_func) {
        total_tests_++;
        std::cout << "Test " << total_tests_ << ": " << test_name << " - ";
        
        try {
            if (test_func()) {
                std::cout << "PASSED" << std::endl;
                passed_tests_++;
            } else {
                std::cout << "FAILED" << std::endl;
            }
        } catch (const std::exception& e) {
            std::cout << "FAILED (Exception: " << e.what() << ")" << std::endl;
        }
    }

    void print_summary() {
        std::cout << "\n=== Test Summary ===" << std::endl;
        std::cout << "Total tests: " << total_tests_ << std::endl;
        std::cout << "Passed: " << passed_tests_ << std::endl;
        std::cout << "Failed: " << (total_tests_ - passed_tests_) << std::endl;
        std::cout << "Success rate: " << (100.0 * passed_tests_ / total_tests_) << "%" << std::endl;
    }

    bool all_passed() const {
        return passed_tests_ == total_tests_;
    }
};

// Global variables for test instances
std::shared_ptr<crlgru::FEPGRUCell> g_cell;
std::shared_ptr<crlgru::FEPGRUNetwork> g_network;
std::shared_ptr<crlgru::SPSAOptimizer> g_optimizer;
std::shared_ptr<crlgru::PolarSpatialAttention> g_attention;
std::shared_ptr<crlgru::MetaEvaluator> g_evaluator;

// Test functions for each component
namespace crlgru_tests {

    bool test_fep_gru_cell_construction() {
        try {
            crlgru::FEPGRUCell::Config config;
            config.input_size = 10;
            config.hidden_size = 64;
            config.enable_som_extraction = true;
            g_cell = std::make_shared<crlgru::FEPGRUCell>(config);
            return g_cell != nullptr;
        } catch (const std::exception& e) {
            std::cerr << "Exception: " << e.what() << std::endl;
            return false;
        }
    }

    bool test_fep_gru_cell_forward() {
        try {
            if (!g_cell) return false;
            
            auto input = torch::randn({1, 10});
            auto hidden = torch::zeros({1, 64});
            auto [new_hidden, prediction, free_energy] = g_cell->forward(input, hidden);
            
            return new_hidden.size(1) == 64 && prediction.defined() && free_energy.defined();
        } catch (const std::exception& e) {
            std::cerr << "Exception: " << e.what() << std::endl;
            return false;
        }
    }

    bool test_fep_gru_cell_free_energy() {
        try {
            if (!g_cell) return false;
            
            auto prediction = torch::randn({1, 10});
            auto target = torch::randn({1, 10});
            auto variance = torch::ones({1, 10}) * 0.1;
            auto fe = g_cell->compute_free_energy(prediction, target, variance);
            return fe.item<double>() >= 0.0;
        } catch (const std::exception& e) {
            std::cerr << "Exception: " << e.what() << std::endl;
            return false;
        }
    }

    bool test_fep_gru_cell_som_features() {
        try {
            if (!g_cell) return false;
            
            auto som_features = g_cell->extract_som_features();
            return som_features.defined() && som_features.numel() > 0;
        } catch (const std::exception& e) {
            std::cerr << "Exception: " << e.what() << std::endl;
            return false;
        }
    }

    bool test_fep_gru_network_construction() {
        try {
            crlgru::FEPGRUNetwork::NetworkConfig config;
            config.layer_sizes = {64, 128, 64};
            g_network = std::make_shared<crlgru::FEPGRUNetwork>(config);
            return g_network != nullptr;
        } catch (const std::exception& e) {
            std::cerr << "Exception: " << e.what() << std::endl;
            return false;
        }
    }

    bool test_fep_gru_network_agent_management() {
        try {
            if (!g_network) return false;
            
            g_network->register_agent(0, 0.8);
            g_network->register_agent(1, 0.6);
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Exception: " << e.what() << std::endl;
            return false;
        }
    }

    bool test_fep_gru_network_collective_energy() {
        try {
            if (!g_network) return false;
            
            auto collective_energy = g_network->compute_collective_free_energy();
            return collective_energy.item<double>() >= 0.0;
        } catch (const std::exception& e) {
            std::cerr << "Exception: " << e.what() << std::endl;
            return false;
        }
    }

    bool test_spsa_optimizer_configuration() {
        try {
            crlgru::SPSAOptimizer::SPSAConfig config;
            config.learning_rate = 0.01;
            config.perturbation_magnitude = 0.1;
            g_optimizer = std::make_shared<crlgru::SPSAOptimizer>(config);
            return g_optimizer != nullptr;
        } catch (const std::exception& e) {
            std::cerr << "Exception: " << e.what() << std::endl;
            return false;
        }
    }

    bool test_spsa_optimizer_gradient_estimation() {
        try {
            if (!g_optimizer) return false;
            
            auto objective_function = [](const torch::Tensor& params) -> double {
                return params.pow(2).sum().item<double>();
            };
            auto params = torch::randn({10});
            auto gradient = g_optimizer->estimate_gradient(params, objective_function);
            
            return gradient.defined() && gradient.numel() == params.numel();
        } catch (const std::exception& e) {
            std::cerr << "Exception: " << e.what() << std::endl;
            return false;
        }
    }

    bool test_spsa_optimizer_optimization() {
        try {
            if (!g_optimizer) return false;
            
            auto objective_function = [](const torch::Tensor& params) -> double {
                return params.pow(2).sum().item<double>();
            };
            auto params = torch::randn({10});
            auto optimized_params = g_optimizer->optimize(params, objective_function);
            
            return optimized_params.defined();
        } catch (const std::exception& e) {
            std::cerr << "Exception: " << e.what() << std::endl;
            return false;
        }
    }

    bool test_utils_polar_coordinate_transformation() {
        try {
            auto positions = torch::randn({5, 2});  // 5 agents, 2D positions
            auto self_position = torch::randn({2}); // 2D self position (not zeros to avoid edge cases)
            auto polar_map = crlgru::utils::cartesian_to_polar_map(
                positions, self_position, 8, 16, 10.0);
            
            return polar_map.defined() && polar_map.size(0) == 8 && polar_map.size(1) == 16;
        } catch (const std::exception& e) {
            std::cerr << "Exception: " << e.what() << std::endl;
            return false;
        }
    }

    bool test_utils_mutual_information() {
        try {
            auto state1 = torch::randn({64});
            auto state2 = torch::randn({64});
            auto mutual_info = crlgru::utils::compute_mutual_information(state1, state2);
            
            return mutual_info >= 0.0;
        } catch (const std::exception& e) {
            std::cerr << "Exception: " << e.what() << std::endl;
            return false;
        }
    }

    bool test_utils_trust_metric() {
        try {
            std::vector<double> performance_history = {0.8, 0.7, 0.9, 0.6};
            double distance = 2.5;
            double max_distance = 10.0;
            auto trust_score = crlgru::utils::compute_trust_metric(
                performance_history, distance, max_distance);
            
            return trust_score >= 0.0 && trust_score <= 1.0;
        } catch (const std::exception& e) {
            std::cerr << "Exception: " << e.what() << std::endl;
            return false;
        }
    }

    bool test_utils_gaussian_kernel() {
        try {
            auto input = torch::randn({32, 32});
            auto smoothed = crlgru::utils::apply_gaussian_kernel(input, 1.0, 5);
            
            return smoothed.defined() && smoothed.sizes() == input.sizes();
        } catch (const std::exception& e) {
            std::cerr << "Exception: " << e.what() << std::endl;
            return false;
        }
    }

    bool test_integration_multi_agent_simulation() {
        try {
            const int steps = 10;
            
            // Create network with proper input configuration
            auto network_config = crlgru::FEPGRUNetwork::NetworkConfig();
            network_config.layer_sizes = {64, 128, 64}; // input_size=64, hidden=128, output=64
            auto brain = std::make_shared<crlgru::FEPGRUNetwork>(network_config);
            
            auto attention_config = crlgru::PolarSpatialAttention::AttentionConfig();
            attention_config.num_distance_rings = 8;
            attention_config.num_angle_sectors = 16;
            auto attention = std::make_shared<crlgru::PolarSpatialAttention>(attention_config);
            
            // Simulate steps with corrected input dimensions
            for (int step = 0; step < steps; ++step) {
                // Input sequence: [sequence_length, input_size] = [10, 64]
                auto input_sequence = torch::randn({10, 64});
                auto [output, prediction, free_energy] = brain->forward(input_sequence);
                
                // Verify outputs
                if (!output.defined() || !prediction.defined() || !free_energy.defined()) {
                    return false;
                }
            }
            
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Exception: " << e.what() << std::endl;
            return false;
        }
    }

    bool test_integration_hierarchical_coordination() {
        try {
            auto attention_config = crlgru::PolarSpatialAttention::AttentionConfig();
            attention_config.num_distance_rings = 8;
            attention_config.num_angle_sectors = 16;
            auto attention = std::make_shared<crlgru::PolarSpatialAttention>(attention_config);
            
            for (int step = 0; step < 20; ++step) {
                auto positions = torch::randn({5, 2});
                auto self_pos = torch::randn({2}); // Use non-zero values to avoid edge cases
                auto polar_map = crlgru::utils::cartesian_to_polar_map(
                    positions, self_pos, 8, 16, 10.0);
                auto attended_map = attention->forward(polar_map.unsqueeze(0).unsqueeze(0));
                
                if (!attended_map.defined()) {
                    return false;
                }
            }
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Exception: " << e.what() << std::endl;
            return false;
        }
    }

    bool test_integration_predictive_coding() {
        try {
            if (!g_network) {
                // Create network if not already created with proper input size
                crlgru::FEPGRUNetwork::NetworkConfig config;
                config.layer_sizes = {64, 128, 64}; // input_size=64, hidden=128, output=64
                g_network = std::make_shared<crlgru::FEPGRUNetwork>(config);
            }
            
            // Use correct input dimensions [sequence_length, input_size]
            auto sequence = torch::randn({64, 64});  // 64 time steps, input_size=64
            auto [output, prediction, free_energy] = g_network->forward(sequence);
            
            return output.defined() && prediction.defined() && free_energy.defined();
        } catch (const std::exception& e) {
            std::cerr << "Exception: " << e.what() << std::endl;
            return false;
        }
    }

    bool test_integration_meta_evaluation() {
        try {
            crlgru::MetaEvaluator::EvaluationConfig eval_config;
            eval_config.objective_weights = {1.0, 1.0, 1.0, 1.0};
            auto evaluator = std::make_shared<crlgru::MetaEvaluator>(eval_config);
            
            auto predicted_states = torch::randn({10, 64});
            auto current_state = torch::randn({64});
            auto environment_state = torch::randn({64});
            auto evaluation_score = evaluator->evaluate(
                predicted_states, current_state, environment_state);
            
            return evaluation_score != 0.0;
        } catch (const std::exception& e) {
            std::cerr << "Exception: " << e.what() << std::endl;
            return false;
        }
    }
}

int main() {
    std::cout << "=== crlGRU Unified Test Suite ===" << std::endl;
    std::cout << "Testing all components in a single executable" << std::endl;
    
    // Initialize PyTorch
    torch::manual_seed(42);
    
    TestRunner runner;
    
    // FEP-GRU Cell Tests
    runner.start_section("FEP-GRU Cell Tests");
    runner.run_test("Cell construction", crlgru_tests::test_fep_gru_cell_construction);
    runner.run_test("Forward pass", crlgru_tests::test_fep_gru_cell_forward);
    runner.run_test("Free energy computation", crlgru_tests::test_fep_gru_cell_free_energy);
    runner.run_test("SOM feature extraction", crlgru_tests::test_fep_gru_cell_som_features);
    
    // FEP-GRU Network Tests
    runner.start_section("FEP-GRU Network Tests");
    runner.run_test("Network construction", crlgru_tests::test_fep_gru_network_construction);
    runner.run_test("Agent management", crlgru_tests::test_fep_gru_network_agent_management);
    runner.run_test("Collective energy computation", crlgru_tests::test_fep_gru_network_collective_energy);
    
    // SPSA Optimizer Tests
    runner.start_section("SPSA Optimizer Tests");
    runner.run_test("SPSA configuration", crlgru_tests::test_spsa_optimizer_configuration);
    runner.run_test("Gradient estimation", crlgru_tests::test_spsa_optimizer_gradient_estimation);
    runner.run_test("Parameter optimization", crlgru_tests::test_spsa_optimizer_optimization);
    
    // Utility Functions Tests
    runner.start_section("Utility Functions Tests");
    runner.run_test("Polar coordinate transformation", crlgru_tests::test_utils_polar_coordinate_transformation);
    runner.run_test("Mutual information computation", crlgru_tests::test_utils_mutual_information);
    runner.run_test("Trust metric computation", crlgru_tests::test_utils_trust_metric);
    runner.run_test("Gaussian kernel application", crlgru_tests::test_utils_gaussian_kernel);
    
    // Integration Tests
    runner.start_section("Integration Tests");
    runner.run_test("Multi-agent simulation", crlgru_tests::test_integration_multi_agent_simulation);
    runner.run_test("Hierarchical coordination", crlgru_tests::test_integration_hierarchical_coordination);
    runner.run_test("Predictive coding integration", crlgru_tests::test_integration_predictive_coding);
    runner.run_test("Multi-objective evaluation", crlgru_tests::test_integration_meta_evaluation);
    
    // Print final results
    runner.print_summary();
    
    if (runner.all_passed()) {
        std::cout << "\n🎉 All tests passed! crlGRU library is functioning correctly." << std::endl;
        return 0;
    } else {
        std::cout << "\n❌ Some tests failed. Please check the implementation." << std::endl;
        return 1;
    }
}
