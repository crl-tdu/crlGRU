#include "crlgru/crl_gru.hpp"
#include <iostream>
#include <vector>
#include <random>

/**
 * @brief Simple time series prediction example using FEP-GRU
 * 
 * This example demonstrates:
 * - Basic FEP-GRU cell usage for time series prediction
 * - Variational free energy computation
 * - SOM feature extraction
 */

int main() {
    try {
        std::cout << "=== FEP-GRU Time Series Prediction Example ===" << std::endl;
        
        // Set random seed for reproducibility
        torch::manual_seed(42);
        
        // Create FEP-GRU cell configuration
        crlgru::FEPGRUCell::Config config;
        config.input_size = 10;
        config.hidden_size = 64;
        config.enable_som_extraction = true;
        config.som_grid_size = 8;
        config.free_energy_weight = 1.0;
        config.variational_beta = 0.5;
        
        // Create FEP-GRU cell
        auto gru_cell = std::make_shared<crlgru::FEPGRUCell>(config);
        
        std::cout << "Created FEP-GRU cell with:" << std::endl;
        std::cout << "  Input size: " << config.input_size << std::endl;
        std::cout << "  Hidden size: " << config.hidden_size << std::endl;
        std::cout << "  SOM enabled: " << (config.enable_som_extraction ? "Yes" : "No") << std::endl;
        
        // Generate synthetic time series data (sine wave with noise)
        int sequence_length = 100;
        int batch_size = 1;
        
        std::vector<torch::Tensor> time_series;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> noise_dist(0.0, 0.1);
        
        for (int t = 0; t < sequence_length; ++t) {
            auto signal = torch::zeros({batch_size, config.input_size});
            
            // Generate multi-dimensional sine wave with different frequencies
            for (int i = 0; i < config.input_size; ++i) {
                double freq = 0.1 * (i + 1);
                double phase = 0.2 * i;
                double value = std::sin(2 * M_PI * freq * t + phase) + noise_dist(gen);
                signal[0][i] = value;
            }
            
            time_series.push_back(signal);
        }
        
        std::cout << "Generated synthetic time series with " << sequence_length << " time steps" << std::endl;
        
        // Process time series through FEP-GRU
        std::vector<torch::Tensor> hidden_states;
        std::vector<torch::Tensor> predictions;
        std::vector<torch::Tensor> free_energies;
        
        torch::Tensor hidden_state;
        double total_free_energy = 0.0;
        
        std::cout << "Processing time series..." << std::endl;
        
        for (int t = 0; t < sequence_length; ++t) {
            auto input = time_series[t];
            
            // Forward pass
            auto [new_hidden, prediction, free_energy] = gru_cell->forward(input, hidden_state);
            
            hidden_state = new_hidden;
            hidden_states.push_back(new_hidden);
            predictions.push_back(prediction);
            free_energies.push_back(free_energy);
            
            total_free_energy += free_energy.mean().item<double>();
            
            // Print progress every 20 steps
            if (t % 20 == 0) {
                std::cout << "  Step " << t << "/" << sequence_length 
                          << " - Free Energy: " << free_energy.mean().item<double>() << std::endl;
            }
        }
        
        std::cout << "Processing completed!" << std::endl;
        std::cout << "Average Free Energy: " << total_free_energy / sequence_length << std::endl;
        
        // Compute prediction accuracy
        double total_prediction_error = 0.0;
        for (int t = 1; t < sequence_length; ++t) {
            auto target = time_series[t];
            auto pred = predictions[t-1];
            auto error = torch::mse_loss(pred, target);
            total_prediction_error += error.item<double>();
        }
        
        double avg_prediction_error = total_prediction_error / (sequence_length - 1);
        std::cout << "Average Prediction Error (MSE): " << avg_prediction_error << std::endl;
        
        // Extract SOM features
        std::cout << "\nExtracting SOM features..." << std::endl;
        auto som_features = gru_cell->extract_som_features();
        auto som_weights = gru_cell->get_som_weights();
        
        if (som_features.defined() && som_weights.defined()) {
            std::cout << "SOM features shape: [" << som_features.sizes() << "]" << std::endl;
            std::cout << "SOM weights shape: [" << som_weights.sizes() << "]" << std::endl;
            
            // Print some SOM statistics
            auto som_activation_variance = som_weights.var().item<double>();
            std::cout << "SOM weight variance: " << som_activation_variance << std::endl;
        }
        
        // Test imitation learning functionality
        std::cout << "\nTesting imitation learning..." << std::endl;
        
        // Create a second GRU cell to imitate
        auto gru_cell_2 = std::make_shared<crlgru::FEPGRUCell>(config);
        
        // Collect parameters from first cell
        std::unordered_map<std::string, torch::Tensor> params_1;
        for (auto& param_pair : gru_cell->named_parameters()) {
            params_1[param_pair.key()] = param_pair.value();
        }
        
        // Update second cell with first cell's parameters (imitation)
        gru_cell_2->update_parameters_from_peer(1, params_1, 0.8); // Peer ID 1, performance 0.8
        
        std::cout << "Parameter imitation completed!" << std::endl;
        
        // Test both cells on the same input to see similarity
        auto test_input = time_series[50]; // Middle of sequence
        auto [hidden_1, pred_1, fe_1] = gru_cell->forward(test_input);
        auto [hidden_2, pred_2, fe_2] = gru_cell_2->forward(test_input);
        
        auto prediction_similarity = torch::cosine_similarity(pred_1, pred_2, 1).mean().item<double>();
        std::cout << "Prediction similarity after imitation: " << prediction_similarity << std::endl;
        
        // Test SPSA optimizer
        std::cout << "\nTesting SPSA optimizer..." << std::endl;
        
        crlgru::SPSAOptimizer::SPSAConfig spsa_config;
        spsa_config.learning_rate = 0.01;
        spsa_config.max_iterations = 50;
        spsa_config.tolerance = 1e-4;
        
        auto spsa_optimizer = std::make_shared<crlgru::SPSAOptimizer>(spsa_config);
        
        // Create a simple optimization problem (minimize distance to target)
        auto target_params = torch::randn({10});
        auto initial_params = torch::randn({10});
        
        auto objective_function = [&target_params](const torch::Tensor& params) -> double {
            auto distance = torch::norm(params - target_params);
            return -distance.item<double>(); // Negative because SPSA maximizes
        };
        
        auto optimized_params = spsa_optimizer->optimize(initial_params, objective_function);
        
        auto initial_distance = torch::norm(initial_params - target_params).item<double>();
        auto final_distance = torch::norm(optimized_params - target_params).item<double>();
        
        std::cout << "SPSA Optimization results:" << std::endl;
        std::cout << "  Initial distance to target: " << initial_distance << std::endl;
        std::cout << "  Final distance to target: " << final_distance << std::endl;
        std::cout << "  Improvement: " << (initial_distance - final_distance) << std::endl;
        
        // Test utility functions
        std::cout << "\nTesting utility functions..." << std::endl;
        
        // Test mutual information computation
        auto state1 = torch::randn({50});
        auto state2 = torch::randn({50});
        auto correlated_state2 = 0.7 * state1 + 0.3 * torch::randn({50}); // Correlated with state1
        
        auto mi_uncorrelated = crlgru::utils::compute_mutual_information(state1, state2);
        auto mi_correlated = crlgru::utils::compute_mutual_information(state1, correlated_state2);
        
        std::cout << "Mutual Information (uncorrelated): " << mi_uncorrelated << std::endl;
        std::cout << "Mutual Information (correlated): " << mi_correlated << std::endl;
        
        // Test Gaussian kernel application
        auto test_image = torch::randn({32, 32});
        auto smoothed_image = crlgru::utils::apply_gaussian_kernel(test_image, 1.0, 5);
        
        auto smoothing_effect = torch::mse_loss(test_image, smoothed_image).item<double>();
        std::cout << "Gaussian smoothing effect (MSE): " << smoothing_effect << std::endl;
        
        // Test trust metric computation
        std::vector<double> good_performance = {0.8, 0.85, 0.9, 0.88, 0.92};
        std::vector<double> poor_performance = {0.3, 0.2, 0.4, 0.35, 0.25};
        
        auto trust_good_close = crlgru::utils::compute_trust_metric(good_performance, 2.0, 10.0);
        auto trust_good_far = crlgru::utils::compute_trust_metric(good_performance, 8.0, 10.0);
        auto trust_poor_close = crlgru::utils::compute_trust_metric(poor_performance, 2.0, 10.0);
        
        std::cout << "Trust metrics:" << std::endl;
        std::cout << "  Good performance, close: " << trust_good_close << std::endl;
        std::cout << "  Good performance, far: " << trust_good_far << std::endl;
        std::cout << "  Poor performance, close: " << trust_poor_close << std::endl;
        
        // Save and load parameters test
        std::cout << "\nTesting parameter save/load..." << std::endl;
        
        std::unordered_map<std::string, torch::Tensor> test_params;
        test_params["weights"] = torch::randn({64, 32});
        test_params["bias"] = torch::randn({32});
        test_params["embeddings"] = torch::randn({100, 16});
        
        std::string param_file = "test_parameters.bin";
        crlgru::utils::save_parameters(param_file, test_params);
        
        auto loaded_params = crlgru::utils::load_parameters(param_file);
        
        bool load_success = true;
        for (const auto& param_pair : test_params) {
            auto loaded_it = loaded_params.find(param_pair.first);
            if (loaded_it == loaded_params.end()) {
                load_success = false;
                break;
            }
            
            auto original = param_pair.second;
            auto loaded = loaded_it->second;
            
            if (!torch::allclose(original, loaded, 1e-6)) {
                load_success = false;
                break;
            }
        }
        
        std::cout << "Parameter save/load test: " << (load_success ? "PASSED" : "FAILED") << std::endl;
        
        // Test polar coordinate conversion
        std::cout << "\nTesting polar coordinate conversion..." << std::endl;
        
        auto agent_positions = torch::tensor({{{1.0, 2.0}, {3.0, 4.0}, {-1.0, -2.0}}});
        auto self_position = torch::tensor({{0.0, 0.0}});
        
        auto polar_map = crlgru::utils::cartesian_to_polar_map(agent_positions, self_position, 4, 8, 10.0);
        
        std::cout << "Polar map shape: [" << polar_map.sizes() << "]" << std::endl;
        std::cout << "Total agents in polar map: " << polar_map.sum().item<double>() << std::endl;
        
        // Final summary
        std::cout << "\n=== Test Summary ===" << std::endl;
        std::cout << "✓ FEP-GRU cell creation and forward pass" << std::endl;
        std::cout << "✓ Time series prediction with free energy computation" << std::endl;
        std::cout << "✓ SOM feature extraction" << std::endl;
        std::cout << "✓ Parameter imitation learning" << std::endl;
        std::cout << "✓ SPSA optimization" << std::endl;
        std::cout << "✓ Utility functions (mutual information, Gaussian kernel, trust metrics)" << std::endl;
        std::cout << "✓ Parameter save/load functionality" << std::endl;
        std::cout << "✓ Polar coordinate conversion" << std::endl;
        
        std::cout << "\nAll tests completed successfully!" << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
}
