#include "crlgru/crl_gru.hpp"
#include <iostream>
#include <cassert>
#include <cmath>

bool test_fep_gru_cell_creation() {
    try {
        crlgru::FEPGRUCell::Config config;
        config.input_size = 32;
        config.hidden_size = 64;
        config.enable_som_extraction = true;
        
        auto cell = std::make_shared<crlgru::FEPGRUCell>(config);
        
        std::cout << "✓ FEP-GRU cell creation test passed" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "✗ FEP-GRU cell creation test failed: " << e.what() << std::endl;
        return false;
    }
}

bool test_fep_gru_cell_forward() {
    try {
        crlgru::FEPGRUCell::Config config;
        config.input_size = 32;
        config.hidden_size = 64;
        
        auto cell = std::make_shared<crlgru::FEPGRUCell>(config);
        
        auto input = torch::randn({1, 32});
        auto [hidden, prediction, free_energy] = cell->forward(input);
        
        // Check output shapes
        assert(hidden.size(0) == 1 && hidden.size(1) == 64);
        assert(prediction.size(0) == 1 && prediction.size(1) == 32);
        assert(free_energy.size(0) == 1);
        
        // Check that values are reasonable
        assert(hidden.isfinite().all().item<bool>());
        assert(prediction.isfinite().all().item<bool>());
        assert(free_energy.isfinite().all().item<bool>());
        
        std::cout << "✓ FEP-GRU cell forward test passed" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "✗ FEP-GRU cell forward test failed: " << e.what() << std::endl;
        return false;
    }
}

bool test_fep_gru_cell_som_extraction() {
    try {
        crlgru::FEPGRUCell::Config config;
        config.input_size = 32;
        config.hidden_size = 64;
        config.enable_som_extraction = true;
        config.som_grid_size = 8;
        
        auto cell = std::make_shared<crlgru::FEPGRUCell>(config);
        
        // Run a few forward passes to populate internal state
        for (int i = 0; i < 5; ++i) {
            auto input = torch::randn({1, 32});
            cell->forward(input);
        }
        
        auto som_features = cell->extract_som_features();
        assert(som_features.defined());
        assert(som_features.numel() > 0);
        
        auto som_weights = cell->get_som_weights();
        assert(som_weights.defined());
        assert(som_weights.size(0) == 8 && som_weights.size(1) == 8);
        
        std::cout << "✓ FEP-GRU cell SOM extraction test passed" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "✗ FEP-GRU cell SOM extraction test failed: " << e.what() << std::endl;
        return false;
    }
}

bool test_fep_gru_cell_imitation() {
    try {
        crlgru::FEPGRUCell::Config config;
        config.input_size = 32;
        config.hidden_size = 64;
        config.enable_hierarchical_imitation = true;
        
        auto cell1 = std::make_shared<crlgru::FEPGRUCell>(config);
        auto cell2 = std::make_shared<crlgru::FEPGRUCell>(config);
        
        // Get parameters from cell1
        std::unordered_map<std::string, torch::Tensor> params1;
        for (auto& param_pair : cell1->named_parameters()) {
            params1[param_pair.key()] = param_pair.value();
        }
        
        // Update cell2 with cell1's parameters
        cell2->update_parameters_from_peer(1, params1, 0.8);
        
        std::cout << "✓ FEP-GRU cell imitation test passed" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "✗ FEP-GRU cell imitation test failed: " << e.what() << std::endl;
        return false;
    }
}

bool test_free_energy_computation() {
    try {
        crlgru::FEPGRUCell::Config config;
        config.input_size = 32;
        config.hidden_size = 64;
        config.free_energy_weight = 1.0;
        config.variational_beta = 0.5;
        
        auto cell = std::make_shared<crlgru::FEPGRUCell>(config);
        
        auto prediction = torch::randn({1, 32});
        auto target = torch::randn({1, 32});
        auto variance = torch::ones({1, 32}) * 0.1;
        
        auto free_energy = cell->compute_free_energy(prediction, target, variance);
        
        assert(free_energy.defined());
        assert(free_energy.numel() == 1);
        assert(free_energy.isfinite().item<bool>());
        
        std::cout << "✓ Free energy computation test passed" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "✗ Free energy computation test failed: " << e.what() << std::endl;
        return false;
    }
}

int main() {
    std::cout << "=== FEP-GRU Cell Tests ===" << std::endl;
    
    int passed = 0;
    int total = 5;
    
    if (test_fep_gru_cell_creation()) passed++;
    if (test_fep_gru_cell_forward()) passed++;
    if (test_fep_gru_cell_som_extraction()) passed++;
    if (test_fep_gru_cell_imitation()) passed++;
    if (test_free_energy_computation()) passed++;
    
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
