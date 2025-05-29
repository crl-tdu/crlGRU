#include "crlgru/crl_gru.hpp"
#include <iostream>
#include <cassert>
#include <cmath>

bool test_fep_gru_network_creation() {
    try {
        crlgru::FEPGRUNetwork::NetworkConfig config;
        config.layer_sizes = {32, 64, 32};
        config.cell_config.input_size = 32;
        config.cell_config.hidden_size = 64;
        
        auto network = std::make_shared<crlgru::FEPGRUNetwork>(config);
        
        std::cout << "✓ FEP-GRU network creation test passed" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "✗ FEP-GRU network creation test failed: " << e.what() << std::endl;
        return false;
    }
}

bool test_fep_gru_network_forward() {
    try {
        crlgru::FEPGRUNetwork::NetworkConfig config;
        config.layer_sizes = {32, 64, 32};
        config.cell_config.input_size = 32;
        config.cell_config.hidden_size = 64;
        config.sequence_length = 10;
        
        auto network = std::make_shared<crlgru::FEPGRUNetwork>(config);
        
        auto sequence = torch::randn({1, 10, 32}); // [batch, time, features]
        auto [hidden_states, predictions, free_energies] = network->forward(sequence);
        
        // Check output shapes
        assert(hidden_states.size(0) == 1);  // batch size
        assert(hidden_states.size(1) == 10); // sequence length
        assert(hidden_states.size(2) == 32); // final layer size
        
        assert(predictions.size(0) == 1);
        assert(predictions.size(1) == 10);
        // Final layer predicts to its input size (which is the previous layer's hidden size)
        // For layer_sizes = {32, 64, 32}, final layer input_size = 64
        assert(predictions.size(2) == 64); // Not 32, but 64!
        
        assert(free_energies.size(0) == 1);
        assert(free_energies.size(1) == 10);
        
        // Check that values are reasonable
        assert(hidden_states.isfinite().all().item<bool>());
        assert(predictions.isfinite().all().item<bool>());
        assert(free_energies.isfinite().all().item<bool>());
        
        std::cout << "✓ FEP-GRU network forward test passed" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "✗ FEP-GRU network forward test failed: " << e.what() << std::endl;
        return false;
    }
}

bool test_fep_gru_network_agent_management() {
    try {
        crlgru::FEPGRUNetwork::NetworkConfig config;
        config.layer_sizes = {32, 64, 32};
        config.cell_config.input_size = 32;
        config.cell_config.hidden_size = 64;
        
        auto network = std::make_shared<crlgru::FEPGRUNetwork>(config);
        
        // Register agents
        network->register_agent(1, 0.5);
        network->register_agent(2, 0.7);
        network->register_agent(3, 0.3);
        
        // Update performance
        network->update_agent_performance(1, 0.8);
        network->update_agent_performance(2, 0.6);
        
        // Remove agent
        network->remove_agent(3);
        
        std::cout << "✓ FEP-GRU network agent management test passed" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "✗ FEP-GRU network agent management test failed: " << e.what() << std::endl;
        return false;
    }
}

bool test_fep_gru_network_parameter_sharing() {
    try {
        crlgru::FEPGRUNetwork::NetworkConfig config;
        config.layer_sizes = {32, 64, 32};
        config.cell_config.input_size = 32;
        config.cell_config.hidden_size = 64;
        config.cell_config.enable_hierarchical_imitation = true;
        
        auto network = std::make_shared<crlgru::FEPGRUNetwork>(config);
        
        // Register agents with different performance levels
        network->register_agent(1, 0.9);  // High performance
        network->register_agent(2, 0.5);  // Medium performance
        network->register_agent(3, 0.2);  // Low performance
        
        // Share parameters with agents
        std::vector<int> agent_ids = {1, 2, 3};
        network->share_parameters_with_agents(agent_ids);
        
        std::cout << "✓ FEP-GRU network parameter sharing test passed" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "✗ FEP-GRU network parameter sharing test failed: " << e.what() << std::endl;
        return false;
    }
}

bool test_collective_free_energy() {
    try {
        crlgru::FEPGRUNetwork::NetworkConfig config;
        config.layer_sizes = {32, 64, 32};
        config.cell_config.input_size = 32;
        config.cell_config.hidden_size = 64;
        
        auto network = std::make_shared<crlgru::FEPGRUNetwork>(config);
        
        // Run forward pass to populate internal states
        auto sequence = torch::randn({1, 10, 32});
        network->forward(sequence);
        
        auto collective_fe = network->compute_collective_free_energy();
        
        assert(collective_fe.defined());
        assert(collective_fe.numel() == 1);
        assert(collective_fe.isfinite().item<bool>());
        
        std::cout << "✓ Collective free energy test passed" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "✗ Collective free energy test failed: " << e.what() << std::endl;
        return false;
    }
}

bool test_multi_layer_som_features() {
    try {
        crlgru::FEPGRUNetwork::NetworkConfig config;
        config.layer_sizes = {32, 64, 32};
        config.cell_config.input_size = 32;
        config.cell_config.hidden_size = 64;
        config.cell_config.enable_som_extraction = true;
        
        auto network = std::make_shared<crlgru::FEPGRUNetwork>(config);
        
        // Run forward pass to populate internal states
        auto sequence = torch::randn({1, 10, 32});
        network->forward(sequence);
        
        auto som_features = network->extract_multi_layer_som_features();
        
        // Should have features from multiple layers
        assert(som_features.size() > 0);
        
        for (const auto& features : som_features) {
            assert(features.defined());
            assert(features.numel() > 0);
        }
        
        std::cout << "✓ Multi-layer SOM features test passed" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "✗ Multi-layer SOM features test failed: " << e.what() << std::endl;
        return false;
    }
}

int main() {
    std::cout << "=== FEP-GRU Network Tests ===" << std::endl;
    
    int passed = 0;
    int total = 6;
    
    if (test_fep_gru_network_creation()) passed++;
    if (test_fep_gru_network_forward()) passed++;
    if (test_fep_gru_network_agent_management()) passed++;
    if (test_fep_gru_network_parameter_sharing()) passed++;
    if (test_collective_free_energy()) passed++;
    if (test_multi_layer_som_features()) passed++;
    
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
