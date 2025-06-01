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
std::shared_ptr<crlgru::SPSAOptimizer<double>> g_optimizer;
std::shared_ptr<crlgru::PolarSpatialAttention> g_attention;
std::shared_ptr<crlgru::MetaEvaluator> g_evaluator;

// Test functions for each component
namespace crlgru_tests {

    bool test_fep_gru_cell_construction() {
        try {
            crlgru::FEPGRUCellConfig config;
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
            crlgru::FEPGRUNetworkConfig config;
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
            auto params = torch::randn({10}, torch::requires_grad(true));
            std::vector<torch::Tensor> param_list = {params};
            
            typename crlgru::SPSAOptimizer<double>::Config config;
            config.a = 0.16;
            config.c = 0.16;
            config.learning_rate = 0.01;
            
            g_optimizer = std::make_shared<crlgru::SPSAOptimizer<double>>(param_list, config);
            return g_optimizer != nullptr;
        } catch (const std::exception& e) {
            std::cerr << "Exception: " << e.what() << std::endl;
            return false;
        }
    }

    bool test_spsa_optimizer_step() {
        try {
            if (!g_optimizer) return false;
            
            auto objective_function = []() -> double {
                return 1.0; // Simple objective
            };
            
            g_optimizer->step(objective_function, 1);
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Exception: " << e.what() << std::endl;
            return false;
        }
    }

    bool test_spsa_optimizer_optimization() {
        try {
            if (!g_optimizer) return false;
            
            auto objective_function = []() -> double {
                return 2.0; // Simple objective
            };
            
            auto final_loss = g_optimizer->optimize(objective_function);
            return final_loss >= 0.0;
        } catch (const std::exception& e) {
            std::cerr << "Exception: " << e.what() << std::endl;
            return false;
        }
    }

    bool test_utils_polar_coordinate_transformation() {
        try {
            auto positions = torch::tensor({{{1.0, 1.0}, {-1.0, -1.0}}});
            auto self_position = torch::tensor({{0.0, 0.0}});
            auto polar_map = crlgru::utils::cartesian_to_polar_map(
                positions, self_position, 8, 16, 10.0);
            
            return polar_map.defined() && polar_map.size(1) == 8 && polar_map.size(2) == 16;
        } catch (const std::exception& e) {
            std::cerr << "Exception: " << e.what() << std::endl;
            return false;
        }
    }

    bool test_utils_safe_normalize() {
        try {
            auto tensor = torch::tensor({3.0, 4.0});
            auto normalized = crlgru::utils::safe_normalize(tensor);
            auto norm = torch::norm(normalized).item<double>();
            
            return std::abs(norm - 1.0) < 1e-6;
        } catch (const std::exception& e) {
            std::cerr << "Exception: " << e.what() << std::endl;
            return false;
        }
    }

    bool test_utils_stable_softmax() {
        try {
            auto logits = torch::tensor({1.0, 2.0, 3.0});
            auto softmax = crlgru::utils::stable_softmax(logits, 0);
            auto sum = torch::sum(softmax).item<double>();
            
            return std::abs(sum - 1.0) < 1e-6;
        } catch (const std::exception& e) {
            std::cerr << "Exception: " << e.what() << std::endl;
            return false;
        }
    }

    bool test_utils_tensor_mean() {
        try {
            auto tensor = torch::tensor({1.0, 2.0, 3.0, 4.0, 5.0});
            auto mean = crlgru::utils::tensor_mean(tensor);
            
            return std::abs(mean - 3.0) < 1e-6;
        } catch (const std::exception& e) {
            std::cerr << "Exception: " << e.what() << std::endl;
            return false;
        }
    }

    bool test_polar_spatial_attention_construction() {
        try {
            crlgru::PolarSpatialAttentionConfig config;
            config.input_channels = 64;
            config.num_distance_rings = 8;
            config.num_angle_sectors = 16;
            
            g_attention = std::make_shared<crlgru::PolarSpatialAttention>(config);
            return g_attention != nullptr;
        } catch (const std::exception& e) {
            std::cerr << "Exception: " << e.what() << std::endl;
            return false;
        }
    }

    bool test_polar_spatial_attention_forward() {
        try {
            if (!g_attention) return false;
            
            // [batch_size, rings, sectors, features]
            auto polar_map = torch::randn({1, 8, 16, 64});
            auto result = g_attention->forward(polar_map);
            
            return result.defined() && result.size(1) == 64;
        } catch (const std::exception& e) {
            std::cerr << "Exception: " << e.what() << std::endl;
            return false;
        }
    }

    bool test_integration_multi_agent_simulation() {
        try {
            // Create network with proper configuration
            crlgru::FEPGRUNetworkConfig network_config;
            network_config.layer_sizes = {64, 128, 64};
            auto brain = std::make_shared<crlgru::FEPGRUNetwork>(network_config);
            
            // Simulate steps
            for (int step = 0; step < 5; ++step) {
                auto input_sequence = torch::randn({10, 64}); // [seq_len, input_size]
                auto [output, prediction, free_energy] = brain->forward(input_sequence);
                
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

    bool test_integration_predictive_coding() {
        try {
            if (!g_network) {
                crlgru::FEPGRUNetworkConfig config;
                config.layer_sizes = {64, 128, 64};
                g_network = std::make_shared<crlgru::FEPGRUNetwork>(config);
            }
            
            auto sequence = torch::randn({64, 64}); // [seq_len, input_size]
            auto [output, prediction, free_energy] = g_network->forward(sequence);
            
            return output.defined() && prediction.defined() && free_energy.defined();
        } catch (const std::exception& e) {
            std::cerr << "Exception: " << e.what() << std::endl;
            return false;
        }
    }

    bool test_meta_evaluator_construction() {
        try {
            crlgru::MetaEvaluatorConfig eval_config;
            eval_config.metrics = {"prediction_accuracy", "free_energy"};
            eval_config.adaptive_weights = true;
            
            g_evaluator = std::make_shared<crlgru::MetaEvaluator>(eval_config);
            return g_evaluator != nullptr;
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
    runner.run_test("SPSA step", crlgru_tests::test_spsa_optimizer_step);
    runner.run_test("Parameter optimization", crlgru_tests::test_spsa_optimizer_optimization);
    
    // Utility Functions Tests
    runner.start_section("Utility Functions Tests");
    runner.run_test("Polar coordinate transformation", crlgru_tests::test_utils_polar_coordinate_transformation);
    runner.run_test("Safe normalize", crlgru_tests::test_utils_safe_normalize);
    runner.run_test("Stable softmax", crlgru_tests::test_utils_stable_softmax);
    runner.run_test("Tensor mean", crlgru_tests::test_utils_tensor_mean);
    
    // Spatial Attention Tests
    runner.start_section("Spatial Attention Tests");
    runner.run_test("Attention construction", crlgru_tests::test_polar_spatial_attention_construction);
    runner.run_test("Attention forward", crlgru_tests::test_polar_spatial_attention_forward);
    
    // Meta Evaluator Tests
    runner.start_section("Meta Evaluator Tests");
    runner.run_test("Evaluator construction", crlgru_tests::test_meta_evaluator_construction);
    
    // Integration Tests
    runner.start_section("Integration Tests");
    runner.run_test("Multi-agent simulation", crlgru_tests::test_integration_multi_agent_simulation);
    runner.run_test("Predictive coding integration", crlgru_tests::test_integration_predictive_coding);
    
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
