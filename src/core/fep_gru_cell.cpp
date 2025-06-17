#include "crlgru/core/fep_gru_cell.hpp"
#include <cmath>
#include <random>
#include <algorithm>

namespace crlgru {
namespace core {

FEPGRUCell::FEPGRUCell(const Config& config) : config_(config) {
    // Initialize standard GRU layers
    input_to_hidden_ = register_module("input_to_hidden", 
        torch::nn::Linear(config_.input_size, config_.hidden_size));
    hidden_to_hidden_ = register_module("hidden_to_hidden",
        torch::nn::Linear(config_.hidden_size, config_.hidden_size));
    
    input_to_reset_ = register_module("input_to_reset",
        torch::nn::Linear(config_.input_size, config_.hidden_size));
    hidden_to_reset_ = register_module("hidden_to_reset",
        torch::nn::Linear(config_.hidden_size, config_.hidden_size));
    
    input_to_update_ = register_module("input_to_update",
        torch::nn::Linear(config_.input_size, config_.hidden_size));
    hidden_to_update_ = register_module("hidden_to_update",
        torch::nn::Linear(config_.hidden_size, config_.hidden_size));
    
    // FEP specific layers
    prediction_head_ = register_module("prediction_head",
        torch::nn::Linear(config_.hidden_size, config_.input_size));
    variance_head_ = register_module("variance_head",
        torch::nn::Linear(config_.hidden_size, config_.input_size));
    meta_evaluation_head_ = register_module("meta_evaluation_head",
        torch::nn::Linear(config_.hidden_size, 1));
    
    // Initialize SOM weights if enabled
    if (config_.enable_som_extraction) {
        // SOM features: hidden_size + 2 (prediction_error + free_energy)
        int som_feature_size = config_.hidden_size + 2;
        som_weights_ = torch::randn({config_.som_grid_size, config_.som_grid_size, som_feature_size});
        som_activation_history_ = torch::zeros({config_.som_grid_size, config_.som_grid_size});
    }
    
    reset_states();
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> 
FEPGRUCell::forward(const torch::Tensor& input, const torch::Tensor& hidden) {
    torch::Tensor h_prev = hidden.defined() ? hidden : hidden_state_;
    
    if (!h_prev.defined() || h_prev.numel() == 0) {
        h_prev = torch::zeros({input.size(0), config_.hidden_size}, input.options());
    }
    
    // Standard GRU computation
    auto reset_gate = torch::sigmoid(input_to_reset_->forward(input) + 
                                   hidden_to_reset_->forward(h_prev));
    auto update_gate = torch::sigmoid(input_to_update_->forward(input) + 
                                    hidden_to_update_->forward(h_prev));
    
    auto reset_hidden = reset_gate * h_prev;
    auto new_gate = torch::tanh(input_to_hidden_->forward(input) + 
                               hidden_to_hidden_->forward(reset_hidden));
    
    auto new_hidden = (1 - update_gate) * new_gate + update_gate * h_prev;
    
    // Predictive coding: generate prediction and variance
    auto prediction = prediction_head_->forward(new_hidden);
    auto log_variance = variance_head_->forward(new_hidden);
    auto variance = torch::exp(log_variance);
    
    // Compute prediction error
    prediction_error_ = torch::mean(torch::pow(input - prediction, 2), -1, true);
    
    // Compute variational free energy
    free_energy_ = compute_free_energy(prediction, input, variance);
    
    // Update internal state
    hidden_state_ = new_hidden;
    
    // Update SOM if enabled
    if (config_.enable_som_extraction) {
        update_som(new_hidden);
    }
    
    return std::make_tuple(new_hidden, prediction, free_energy_);
}

torch::Tensor FEPGRUCell::compute_free_energy(const torch::Tensor& prediction, 
                                            const torch::Tensor& target,
                                            const torch::Tensor& variance) const {
    // Variational free energy: F = KL[q(x)||p(x|y)] + H[p(y)]
    // Simplified as: prediction error + complexity penalty
    
    auto prediction_error = torch::mean(torch::pow(target - prediction, 2) / variance, -1);
    auto complexity_penalty = 0.5 * torch::mean(torch::log(variance), -1);
    auto prior_penalty = config_.variational_beta * torch::mean(torch::pow(prediction, 2), -1);
    
    return config_.free_energy_weight * (prediction_error + complexity_penalty + prior_penalty);
}

void FEPGRUCell::update_parameters_from_peer(int peer_id, 
                                           const std::unordered_map<std::string, torch::Tensor>& peer_params,
                                           double peer_performance) {
    // Store peer information
    peer_performance_[peer_id] = peer_performance;
    
    // Update trust based on performance and distance (if available)
    double trust = std::min(1.0, peer_performance / (peer_performance + 1.0));
    peer_trust_[peer_id] = trust;
    
    // Real-time parameter blending based on trust and imitation rate
    if (trust > 0.5 && config_.enable_hierarchical_imitation) {
        auto learning_rate = config_.imitation_rate * trust;
        
        // Blend parameters with peer
        for (auto& param_pair : named_parameters()) {
            auto param_name = param_pair.key();
            auto& param_tensor = param_pair.value();
            
            auto peer_it = peer_params.find(param_name);
            if (peer_it != peer_params.end()) {
                auto peer_param = peer_it->second;
                if (param_tensor.sizes() == peer_param.sizes()) {
                    // Use safe tensor operation that preserves computational graph
                    auto updated_param = (1.0 - learning_rate) * param_tensor + 
                                        learning_rate * peer_param;
                    param_tensor.copy_(updated_param);
                }
            }
        }
    }
}

void FEPGRUCell::hierarchical_imitation_update(int best_peer_id) {
    if (peer_trust_.find(best_peer_id) == peer_trust_.end()) {
        return; // No information about this peer
    }
    
    double trust = peer_trust_[best_peer_id];
    double performance_ratio = peer_performance_[best_peer_id] / 
                              (get_meta_evaluation() + 1e-8);
    
    if (performance_ratio > 1.1 && trust > 0.7) { // Peer performs significantly better and is trustworthy
        // Level 1: Prediction result imitation (highest level)

        // Level 2: Exploration strategy imitation (medium level)

        // Level 3: Internal model distillation (lowest level)

        // Apply hierarchical updates
        // (Implementation would depend on specific peer state access)
        // This is a framework for the hierarchical imitation mechanism
    }
}

torch::Tensor FEPGRUCell::extract_som_features() const {
    if (!config_.enable_som_extraction || !hidden_state_.defined()) {
        return torch::Tensor();
    }

    // Extract relevant features for SOM clustering
    auto hidden_mean = hidden_state_.mean(0, true);  // [1, hidden_size]
    
    // Ensure prediction_error_ and free_energy_ have compatible dimensions
    torch::Tensor pred_error_feature;
    torch::Tensor free_energy_feature;
    
    if (prediction_error_.defined() && prediction_error_.numel() > 0) {
        // Reduce prediction error to single value and expand to match hidden dimensions
        auto pred_error_scalar = prediction_error_.mean();
        pred_error_feature = pred_error_scalar.view({1, 1}).expand({1, 1});
    } else {
        pred_error_feature = torch::zeros({1, 1});
    }
    
    if (free_energy_.defined() && free_energy_.numel() > 0) {
        // Reduce free energy to single value and expand to match hidden dimensions
        auto free_energy_scalar = free_energy_.mean();
        free_energy_feature = free_energy_scalar.view({1, 1}).expand({1, 1});
    } else {
        free_energy_feature = torch::zeros({1, 1});
    }

    // Combine features: [hidden_size + 2] (hidden state + pred_error + free_energy)
    auto features = torch::cat({
        hidden_mean,           // [1, hidden_size]
        pred_error_feature,    // [1, 1]
        free_energy_feature    // [1, 1]
    }, -1);

    return features;
}

void FEPGRUCell::update_som(const torch::Tensor&) {
    if (!config_.enable_som_extraction || !som_weights_.defined()) {
        return;
    }

    auto features = extract_som_features();
    if (!features.defined() || features.numel() == 0) {
        return;
    }

    // Find best matching unit (BMU)
    auto expanded_features = features.unsqueeze(0).unsqueeze(0);
    auto distances = torch::sum(torch::pow(som_weights_ - expanded_features, 2), -1);
    auto flat_distances = distances.view(-1);
    auto bmu_idx = torch::argmin(flat_distances);

    int bmu_i = bmu_idx.item<int>() / config_.som_grid_size;
    int bmu_j = bmu_idx.item<int>() % config_.som_grid_size;

    // Update SOM weights using Gaussian neighborhood
    double sigma = config_.som_grid_size / 4.0; // Neighborhood radius

    for (int i = 0; i < config_.som_grid_size; ++i) {
        for (int j = 0; j < config_.som_grid_size; ++j) {
            double dist_sq = (i - bmu_i) * (i - bmu_i) + (j - bmu_j) * (j - bmu_j);
            double influence = std::exp(-dist_sq / (2.0 * sigma * sigma));
            double learning_rate = config_.som_learning_rate * influence;

            som_weights_[i][j] = som_weights_[i][j] +
                               learning_rate * (features.squeeze() - som_weights_[i][j]);
        }
    }

    // Update activation history
    som_activation_history_[bmu_i][bmu_j] += 1.0;
}

double FEPGRUCell::get_meta_evaluation() {
    if (!hidden_state_.defined() || !meta_evaluation_head_) {
        return 0.0;
    }
    
    // Use evaluation mode and no grad
    torch::NoGradGuard no_grad;
    bool was_training = meta_evaluation_head_->is_training();
    meta_evaluation_head_->eval();
    
    auto input_tensor = hidden_state_.mean(0, true).detach();
    auto meta_score = meta_evaluation_head_->forward(input_tensor);
    
    // Restore training mode
    meta_evaluation_head_->train(was_training);
    
    return torch::sigmoid(meta_score).item<double>();
}

void FEPGRUCell::reset_states() {
    hidden_state_ = torch::Tensor();
    prediction_error_ = torch::Tensor();
    free_energy_ = torch::Tensor();
    
    // Clear peer information
    peer_parameters_.clear();
    peer_performance_.clear();
    peer_trust_.clear();
    
    // Reset SOM activation history
    if (config_.enable_som_extraction && som_activation_history_.defined()) {
        som_activation_history_.zero_();
    }
}

} // namespace core
} // namespace crlgru
