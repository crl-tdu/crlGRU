#include "crlgru/crl_gru.hpp"
#include <algorithm>

namespace crlgru {

FEPGRUNetwork::FEPGRUNetwork(const NetworkConfig& config) : config_(config) {
    // Create multi-layer FEP-GRU cells
    for (size_t i = 0; i < config_.layer_sizes.size(); ++i) {
        auto cell_config = config_.cell_config;
        
        if (i == 0) {
            // First layer uses the configured input size
            cell_config.input_size = config_.cell_config.input_size;
        } else {
            // Subsequent layers use previous layer's hidden size as input
            cell_config.input_size = config_.layer_sizes[i-1];
        }
        
        cell_config.hidden_size = config_.layer_sizes[i];
        
        auto cell = std::make_shared<FEPGRUCell>(cell_config);
        layers_.push_back(cell);
        register_module("layer_" + std::to_string(i), cell);
    }
    
    // Dropout layer
    if (config_.layer_dropout > 0.0) {
        dropout_ = register_module("dropout", torch::nn::Dropout(config_.layer_dropout));
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
FEPGRUNetwork::forward(const torch::Tensor& sequence) {
    // sequence shape: [batch_size, sequence_length, input_size]
    // auto batch_size = sequence.size(0);  // Commented out as unused
    auto seq_length = sequence.size(1);
    
    std::vector<torch::Tensor> all_hidden_states;
    std::vector<torch::Tensor> all_predictions;
    std::vector<torch::Tensor> all_free_energies;
    
    torch::Tensor current_input = sequence;
    
    // Process through each layer
    for (size_t layer_idx = 0; layer_idx < layers_.size(); ++layer_idx) {
        auto& layer = layers_[layer_idx];
        
        std::vector<torch::Tensor> layer_hidden_states;
        std::vector<torch::Tensor> layer_predictions;
        std::vector<torch::Tensor> layer_free_energies;
        
        torch::Tensor hidden_state;
        
        // Process sequence through current layer
        for (int t = 0; t < seq_length; ++t) {
            auto input_t = current_input.select(1, t); // [batch_size, input_size]
            
            auto [new_hidden, prediction, free_energy] = layer->forward(input_t, hidden_state);
            
            hidden_state = new_hidden;
            layer_hidden_states.push_back(new_hidden);
            layer_predictions.push_back(prediction);
            layer_free_energies.push_back(free_energy);
        }
        
        // Stack temporal outputs
        auto layer_output = torch::stack(layer_hidden_states, 1); // [batch_size, seq_length, hidden_size]
        auto layer_pred = torch::stack(layer_predictions, 1);
        auto layer_fe = torch::stack(layer_free_energies, 1);
        
        all_hidden_states.push_back(layer_output);
        all_predictions.push_back(layer_pred);
        all_free_energies.push_back(layer_fe);
        
        // Apply dropout if configured
        if (dropout_  && layer_idx < layers_.size() - 1) {
            current_input = dropout_->forward(layer_output);
        } else {
            current_input = layer_output;
        }
    }
    
    // Return outputs from the final layer
    auto final_hidden = all_hidden_states.back();
    auto final_prediction = all_predictions.back();
    auto final_free_energy = all_free_energies.back();
    
    return std::make_tuple(final_hidden, final_prediction, final_free_energy);
}

void FEPGRUNetwork::share_parameters_with_agents(const std::vector<int>& agent_ids) {
    // Collect current network parameters
    std::unordered_map<std::string, torch::Tensor> current_params;
    for (auto& param_pair : named_parameters()) {
        current_params[param_pair.key()] = param_pair.value().clone();
    }
    
    // Share with each layer's cells
    for (size_t layer_idx = 0; layer_idx < layers_.size(); ++layer_idx) {
        auto& layer = layers_[layer_idx];
        
        // Find best performing agent for this layer
        int best_agent_id = -1;
        double best_performance = -1.0;
        
        for (int agent_id : agent_ids) {
            auto perf_it = agent_performance_history_.find(agent_id);
            if (perf_it != agent_performance_history_.end() && 
                perf_it->second > best_performance) {
                best_performance = perf_it->second;
                best_agent_id = agent_id;
            }
        }
        
        if (best_agent_id != -1) {
            // Get layer-specific parameters
            std::unordered_map<std::string, torch::Tensor> layer_params;
            std::string layer_prefix = "layer_" + std::to_string(layer_idx) + ".";
            
            for (const auto& param_pair : current_params) {
                if (param_pair.first.find(layer_prefix) == 0) {
                    std::string param_name = param_pair.first.substr(layer_prefix.length());
                    layer_params[param_name] = param_pair.second;
                }
            }
            
            // Update layer with best agent's parameters
            layer->update_parameters_from_peer(best_agent_id, layer_params, best_performance);
        }
    }
}

torch::Tensor FEPGRUNetwork::compute_collective_free_energy() const {
    torch::Tensor collective_fe = torch::zeros({1});
    
    for (const auto& layer : layers_) {
        auto layer_fe = layer->get_free_energy();
        if (layer_fe.defined() && layer_fe.numel() > 0) {
            collective_fe = collective_fe + layer_fe.mean();
        }
    }
    
    return collective_fe / static_cast<double>(layers_.size());
}

std::vector<torch::Tensor> FEPGRUNetwork::extract_multi_layer_som_features() const {
    std::vector<torch::Tensor> multi_layer_features;
    
    for (const auto& layer : layers_) {
        auto som_features = layer->extract_som_features();
        if (som_features.defined() && som_features.numel() > 0) {
            multi_layer_features.push_back(som_features);
        }
    }
    
    return multi_layer_features;
}

void FEPGRUNetwork::register_agent(int agent_id, double initial_performance) {
    agent_performance_history_[agent_id] = initial_performance;
    agent_states_[agent_id] = std::vector<torch::Tensor>();
}

void FEPGRUNetwork::update_agent_performance(int agent_id, double performance) {
    agent_performance_history_[agent_id] = performance;
}

void FEPGRUNetwork::remove_agent(int agent_id) {
    agent_performance_history_.erase(agent_id);
    agent_states_.erase(agent_id);
}

} // namespace crlgru
