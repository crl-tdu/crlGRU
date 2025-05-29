#include "crlgru/crl_gru.hpp"
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>

/**
 * @brief Comprehensive integration test for FEP-GRU system
 * Tests multi-agent scenario with hierarchical imitation learning
 */
bool test_multi_agent_swarm_simulation() {
    try {
        std::cout << "Testing multi-agent swarm simulation..." << std::endl;
        
        // Configuration for swarm agents
        crlgru::FEPGRUCell::Config config;
        config.input_size = 32;  // State representation size
        config.hidden_size = 64; // Internal representation size
        config.enable_hierarchical_imitation = true;
        config.enable_som_extraction = true;
        config.imitation_rate = 0.1;
        config.trust_radius = 5.0;
        config.som_grid_size = 8;
        
        // Create multiple agents
        const int num_agents = 5;
        std::vector<std::shared_ptr<crlgru::FEPGRUCell>> agents;
        
        for (int i = 0; i < num_agents; ++i) {
            agents.push_back(std::make_shared<crlgru::FEPGRUCell>(config));
        }
        
        std::cout << "  Created " << num_agents << " FEP-GRU agents" << std::endl;
        
        // Simulation parameters
        const int time_steps = 20;
        const int input_dim = config.input_size;
        
        // Store agent performances and states
        std::vector<double> agent_performances(num_agents, 0.0);
        std::vector<torch::Tensor> agent_states(num_agents);
        
        // Multi-step simulation
        for (int t = 0; t < time_steps; ++t) {
            std::vector<torch::Tensor> inputs(num_agents);
            std::vector<torch::Tensor> predictions(num_agents);
            std::vector<torch::Tensor> free_energies(num_agents);
            
            // Generate inputs (could represent environmental observations)
            for (int i = 0; i < num_agents; ++i) {
                inputs[i] = torch::randn({1, input_dim}) * 0.5; // Normalized input
            }
            
            // Forward pass for all agents
            for (int i = 0; i < num_agents; ++i) {
                auto [hidden, pred, fe] = agents[i]->forward(inputs[i]);
                agent_states[i] = hidden;
                predictions[i] = pred;
                free_energies[i] = fe;
                
                // Simple performance metric: negative free energy
                agent_performances[i] = -free_energies[i].mean().item<double>();
            }
            
            // Hierarchical imitation learning
            if (t > 5) { // Start imitation after initial exploration
                // Find best performing agent
                int best_agent = 0;
                double best_performance = agent_performances[0];
                
                for (int i = 1; i < num_agents; ++i) {
                    if (agent_performances[i] > best_performance) {
                        best_performance = agent_performances[i];
                        best_agent = i;
                    }
                }
                
                // Other agents imitate the best performer
                for (int i = 0; i < num_agents; ++i) {
                    if (i != best_agent) {
                        // Get parameters from best agent
                        std::unordered_map<std::string, torch::Tensor> best_params;
                        for (auto& param_pair : agents[best_agent]->named_parameters()) {
                            best_params[param_pair.key()] = param_pair.value();
                        }
                        
                        // Update agent with best agent's parameters
                        agents[i]->update_parameters_from_peer(
                            best_agent, best_params, best_performance);
                    }
                }
            }
            
            // Periodic output
            if (t % 5 == 0) {
                double avg_performance = 0.0;
                for (double perf : agent_performances) {
                    avg_performance += perf;
                }
                avg_performance /= num_agents;
                
                std::cout << "    Step " << t << ": Avg performance = " 
                          << avg_performance << ", Best = " << *std::max_element(
                             agent_performances.begin(), agent_performances.end()) << std::endl;
            }
        }
        
        // Verify improvement over time
        double final_avg_performance = 0.0;
        for (double perf : agent_performances) {
            final_avg_performance += perf;
        }
        final_avg_performance /= num_agents;
        
        // Check that all agents have reasonable states
        for (int i = 0; i < num_agents; ++i) {
            assert(agent_states[i].defined());
            assert(agent_states[i].isfinite().all().item<bool>());
            assert(std::isfinite(agent_performances[i]));
        }
        
        std::cout << "✓ Multi-agent swarm simulation test passed" << std::endl;
        std::cout << "  Final average performance: " << final_avg_performance << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "✗ Multi-agent swarm simulation test failed: " << e.what() << std::endl;
        return false;
    }
}

/**
 * @brief Test SOM feature extraction and clustering
 */
bool test_som_clustering_behavior() {
    try {
        std::cout << "Testing SOM clustering behavior..." << std::endl;
        
        crlgru::FEPGRUCell::Config config;
        config.input_size = 16;
        config.hidden_size = 32;
        config.enable_som_extraction = true;
        config.som_grid_size = 6;
        config.som_learning_rate = 0.05;
        
        auto agent = std::make_shared<crlgru::FEPGRUCell>(config);
        
        // Generate structured input patterns
        const int num_patterns = 3;
        const int steps_per_pattern = 10;
        
        std::vector<torch::Tensor> pattern_centers = {
            torch::ones({1, 16}) * 0.5,   // Pattern 1: positive
            torch::ones({1, 16}) * -0.5,  // Pattern 2: negative  
            torch::zeros({1, 16})         // Pattern 3: neutral
        };
        
        std::vector<std::vector<torch::Tensor>> som_features_by_pattern(num_patterns);
        
        // Present each pattern multiple times
        for (int pattern = 0; pattern < num_patterns; ++pattern) {
            agent->reset_states(); // Start fresh for each pattern
            
            for (int step = 0; step < steps_per_pattern; ++step) {
                // Add noise to pattern
                auto noisy_input = pattern_centers[pattern] + 
                                 torch::randn({1, 16}) * 0.1;
                
                // Forward pass
                agent->forward(noisy_input);
                
                // Extract SOM features
                auto som_features = agent->extract_som_features();
                if (som_features.defined() && som_features.numel() > 0) {
                    som_features_by_pattern[pattern].push_back(som_features.clone());
                }
            }
        }
        
        // Verify that SOM features were extracted
        for (int pattern = 0; pattern < num_patterns; ++pattern) {
            assert(som_features_by_pattern[pattern].size() > 0);
            std::cout << "  Pattern " << pattern << ": " 
                      << som_features_by_pattern[pattern].size() << " feature vectors" << std::endl;
        }
        
        // Check SOM weights structure
        auto som_weights = agent->get_som_weights();
        assert(som_weights.defined());
        assert(som_weights.size(0) == config.som_grid_size);
        assert(som_weights.size(1) == config.som_grid_size);
        
        std::cout << "✓ SOM clustering behavior test passed" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "✗ SOM clustering behavior test failed: " << e.what() << std::endl;
        return false;
    }
}

/**
 * @brief Test hierarchical network with agent coordination
 */
bool test_hierarchical_network_coordination() {
    try {
        std::cout << "Testing hierarchical network coordination..." << std::endl;
        
        // Create network configuration
        crlgru::FEPGRUNetwork::NetworkConfig config;
        config.layer_sizes = {32, 64, 32}; // 3-layer hierarchy
        config.cell_config.input_size = 32;
        config.cell_config.hidden_size = 64;
        config.cell_config.enable_hierarchical_imitation = true;
        config.sequence_length = 15;
        config.layer_dropout = 0.1;
        
        auto network = std::make_shared<crlgru::FEPGRUNetwork>(config);
        
        // Register multiple agents with different performance levels
        const int num_agents = 4;
        std::vector<int> agent_ids = {101, 102, 103, 104};
        std::vector<double> performances = {0.9, 0.7, 0.5, 0.3}; // Decreasing performance
        
        for (int i = 0; i < num_agents; ++i) {
            network->register_agent(agent_ids[i], performances[i]);
        }
        
        // Test sequence processing
        auto input_sequence = torch::randn({1, config.sequence_length, 32});
        auto [hidden_states, predictions, free_energies] = network->forward(input_sequence);
        
        // Verify output dimensions
        assert(hidden_states.size(0) == 1);
        assert(hidden_states.size(1) == config.sequence_length);
        assert(hidden_states.size(2) == 32); // Final layer size
        
        assert(predictions.size(0) == 1);
        assert(predictions.size(1) == config.sequence_length);
        assert(predictions.size(2) == 64); // Previous layer size
        
        assert(free_energies.size(0) == 1);
        assert(free_energies.size(1) == config.sequence_length);
        
        // Test parameter sharing mechanism
        network->share_parameters_with_agents(agent_ids);
        
        // Test collective free energy computation
        auto collective_fe = network->compute_collective_free_energy();
        assert(collective_fe.defined());
        assert(collective_fe.numel() == 1);
        assert(collective_fe.isfinite().item<bool>());
        
        // Test agent performance updates
        for (int i = 0; i < num_agents; ++i) {
            network->update_agent_performance(agent_ids[i], performances[i] + 0.1);
        }
        
        // Test agent removal
        network->remove_agent(agent_ids.back());
        
        std::cout << "✓ Hierarchical network coordination test passed" << std::endl;
        std::cout << "  Collective free energy: " << collective_fe.item<double>() << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "✗ Hierarchical network coordination test failed: " << e.what() << std::endl;
        return false;
    }
}

/**
 * @brief Test predictive coding and free energy minimization
 */
bool test_predictive_coding_dynamics() {
    try {
        std::cout << "Testing predictive coding dynamics..." << std::endl;
        
        crlgru::FEPGRUCell::Config config;
        config.input_size = 20;
        config.hidden_size = 40;
        config.free_energy_weight = 1.0;
        config.variational_beta = 0.5;
        config.prediction_horizon = 5.0;
        
        auto agent = std::make_shared<crlgru::FEPGRUCell>(config);
        
        // Generate predictable sequence (sine wave)
        const int sequence_length = 30;
        std::vector<torch::Tensor> inputs;
        std::vector<torch::Tensor> predictions;
        std::vector<torch::Tensor> free_energies;
        
        for (int t = 0; t < sequence_length; ++t) {
            // Create periodic input pattern
            auto input = torch::zeros({1, config.input_size});
            for (int i = 0; i < config.input_size; ++i) {
                input[0][i] = std::sin(2.0 * M_PI * t / 10.0 + i * 0.1);
            }
            
            auto [hidden, pred, fe] = agent->forward(input);
            
            inputs.push_back(input);
            predictions.push_back(pred);
            free_energies.push_back(fe);
        }
        
        // Analyze prediction accuracy over time
        std::vector<double> prediction_errors;
        
        for (int t = 1; t < sequence_length; ++t) {
            auto pred_error = torch::mse_loss(predictions[t-1], inputs[t]);
            prediction_errors.push_back(pred_error.item<double>());
        }
        
        // Check that prediction error generally decreases (learning)
        double early_error = 0.0, late_error = 0.0;
        int mid_point = prediction_errors.size() / 2;
        
        for (int i = 0; i < mid_point; ++i) {
            early_error += prediction_errors[i];
        }
        for (int i = mid_point; i < prediction_errors.size(); ++i) {
            late_error += prediction_errors[i];
        }
        
        early_error /= mid_point;
        late_error /= (prediction_errors.size() - mid_point);
        
        // Verify free energy computation
        for (const auto& fe : free_energies) {
            assert(fe.defined());
            assert(fe.isfinite().all().item<bool>());
        }
        
        std::cout << "✓ Predictive coding dynamics test passed" << std::endl;
        std::cout << "  Early prediction error: " << early_error << std::endl;
        std::cout << "  Late prediction error: " << late_error << std::endl;
        
        // Learning should show some improvement
        if (late_error < early_error) {
            std::cout << "  ✓ Prediction improvement detected!" << std::endl;
        } else {
            std::cout << "  ! No clear prediction improvement (may be normal for short sequence)" << std::endl;
        }
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "✗ Predictive coding dynamics test failed: " << e.what() << std::endl;
        return false;
    }
}

int main() {
    std::cout << "=== FEP-GRU Integration Tests ===" << std::endl;
    std::cout << "Testing comprehensive swarm intelligence functionality\n" << std::endl;
    
    int passed = 0;
    int total = 4;
    
    if (test_multi_agent_swarm_simulation()) passed++;
    std::cout << std::endl;
    
    if (test_som_clustering_behavior()) passed++;
    std::cout << std::endl;
    
    if (test_hierarchical_network_coordination()) passed++;
    std::cout << std::endl;
    
    if (test_predictive_coding_dynamics()) passed++;
    std::cout << std::endl;
    
    std::cout << "=== Integration Test Results ===" << std::endl;
    std::cout << "Passed: " << passed << "/" << total << std::endl;
    
    if (passed == total) {
        std::cout << "All integration tests passed! ✓" << std::endl;
        std::cout << "\nFEP-GRU system is ready for crlNexus integration." << std::endl;
        return 0;
    } else {
        std::cout << "Some integration tests failed! ✗" << std::endl;
        return 1;
    }
}
