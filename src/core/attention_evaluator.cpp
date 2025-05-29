#include "crlgru/crl_gru.hpp"
#include <cmath>

namespace crlgru {

PolarSpatialAttention::PolarSpatialAttention(const AttentionConfig& config) : config_(config) {
    // Distance attention: focuses on radial patterns
    distance_attention_ = register_module("distance_attention",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(config_.input_channels, config_.attention_dim, 3)
            .padding(1).bias(true)));
    
    // Angle attention: focuses on angular patterns
    angle_attention_ = register_module("angle_attention", 
        torch::nn::Conv2d(torch::nn::Conv2dOptions(config_.input_channels, config_.attention_dim, 3)
            .padding(1).bias(true)));
    
    // Fusion layer to combine distance and angle attention
    fusion_layer_ = register_module("fusion_layer",
        torch::nn::Linear(config_.input_channels * 2, config_.input_channels));
    
    if (config_.attention_dropout > 0.0) {
        dropout_ = register_module("dropout", torch::nn::Dropout(config_.attention_dropout));
    }
}

torch::Tensor PolarSpatialAttention::forward(const torch::Tensor& polar_map) {
    // polar_map shape: [batch_size, channels, num_rings, num_sectors]
    auto batch_size = polar_map.size(0);
    auto channels = polar_map.size(1);
    auto num_rings = polar_map.size(2);
    auto num_sectors = polar_map.size(3);
    
    // Compute distance and angle attention weights
    auto [distance_weights, angle_weights] = compute_attention_weights(polar_map);
    
    // Apply attention to the polar map
    auto attended_distance = polar_map * distance_weights.unsqueeze(1);
    auto attended_angle = polar_map * angle_weights.unsqueeze(1);
    
    // Global average pooling
    auto distance_pooled = torch::adaptive_avg_pool2d(attended_distance, {1, 1});
    auto angle_pooled = torch::adaptive_avg_pool2d(attended_angle, {1, 1});
    
    // Flatten and concatenate
    auto distance_flat = distance_pooled.view({batch_size, -1});
    auto angle_flat = angle_pooled.view({batch_size, -1});
    auto combined = torch::cat({distance_flat, angle_flat}, -1);
    
    // Apply dropout if configured
    if (config_.attention_dropout > 0.0) {
        combined = dropout_->forward(combined);
    }
    
    // Fusion layer
    auto attended_features = fusion_layer_->forward(combined);
    
    // Reshape back to original spatial dimensions and broadcast
    auto attention_map = torch::sigmoid(attended_features).unsqueeze(-1).unsqueeze(-1);
    attention_map = attention_map.expand({batch_size, channels, num_rings, num_sectors});
    
    return polar_map * attention_map;
}

std::pair<torch::Tensor, torch::Tensor> 
PolarSpatialAttention::compute_attention_weights(const torch::Tensor& features) {
    // Distance attention: emphasize radial patterns
    auto distance_features = distance_attention_->forward(features);
    auto distance_weights = torch::softmax(distance_features.mean(1, true), 2); // Average over channels, softmax over rings
    
    // Angle attention: emphasize angular patterns  
    auto angle_features = angle_attention_->forward(features);
    auto angle_weights = torch::softmax(angle_features.mean(1, true), 3); // Average over channels, softmax over sectors
    
    return std::make_pair(distance_weights.squeeze(1), angle_weights.squeeze(1));
}

MetaEvaluator::MetaEvaluator(const EvaluationConfig& config) : config_(config) {
    // Initialize weight history
    weight_history_ = torch::ones({static_cast<int>(config_.objective_weights.size()), 1});
    for (size_t i = 0; i < config_.objective_weights.size(); ++i) {
        weight_history_[i][0] = config_.objective_weights[i];
    }
    
    // Initialize default objective functions
    // Goal achievement objective
    objective_functions_.push_back([](const torch::Tensor& predicted_states) -> double {
        // Assume last dimensions are position [x, y] and goal [goal_x, goal_y]
        if (predicted_states.size(-1) < 4) return 0.0;
        
        auto final_pos = predicted_states.select(-2, -1).narrow(-1, 0, 2); // Final position
        auto goal_pos = predicted_states.select(-2, -1).narrow(-1, 2, 2);  // Goal position
        auto distance = torch::norm(final_pos - goal_pos, 2, -1);
        return -distance.mean().item<double>(); // Negative distance (closer is better)
    });
    
    // Collision avoidance objective
    objective_functions_.push_back([](const torch::Tensor& predicted_states) -> double {
        // Simple collision penalty based on minimum distance to others
        if (predicted_states.size(-1) < 6) return 0.0;
        
        auto pos = predicted_states.narrow(-1, 0, 2);
        auto min_distance = torch::norm(pos.unsqueeze(-2) - pos.unsqueeze(-3), 2, -1);
        min_distance = min_distance + torch::eye(min_distance.size(-1)) * 1000.0; // Ignore self-distance
        auto min_result = torch::min(min_distance, -1);
        auto min_dist_value = std::get<0>(min_result).min().item<double>();
        
        return std::max(0.0, min_dist_value - 1.0); // Penalty if too close
    });
    
    // Cohesion objective
    objective_functions_.push_back([](const torch::Tensor& predicted_states) -> double {
        auto pos = predicted_states.narrow(-1, 0, 2);
        auto centroid = pos.mean(-2, true);
        auto cohesion = -torch::norm(pos - centroid, 2, -1).mean().item<double>();
        return cohesion;
    });
    
    // Alignment objective
    objective_functions_.push_back([](const torch::Tensor& predicted_states) -> double {
        if (predicted_states.size(-1) < 4) return 0.0;
        
        auto vel = predicted_states.narrow(-1, 2, 2);
        auto mean_vel = vel.mean(-2, true);
        auto alignment = torch::cosine_similarity(vel, mean_vel, -1).mean().item<double>();
        return alignment;
    });
}

double MetaEvaluator::evaluate(const torch::Tensor& predicted_states, 
                             const torch::Tensor& /* current_state */,
                             const torch::Tensor& /* environment_state */) {
    double total_score = 0.0;
    auto current_weights = weight_history_.select(-1, -1); // Latest weights
    
    for (size_t i = 0; i < objective_functions_.size() && i < static_cast<size_t>(current_weights.size(0)); ++i) {
        double objective_score = objective_functions_[i](predicted_states);
        double weight = current_weights[i].item<double>();
        total_score += weight * objective_score;
    }
    
    return total_score;
}

void MetaEvaluator::add_objective(std::function<double(const torch::Tensor&)> objective) {
    objective_functions_.push_back(objective);
    
    // Extend weight history with default weight
    auto new_weights = torch::cat({weight_history_, torch::ones({1, weight_history_.size(1)})}, 0);
    weight_history_ = new_weights;
}

void MetaEvaluator::adapt_weights(const std::vector<double>& recent_performance) {
    if (!config_.adaptive_weights || recent_performance.empty()) {
        return;
    }
    
    // Simple adaptive weight adjustment based on recent performance
    auto current_weights = weight_history_.select(-1, -1);
    auto new_weights = current_weights.clone();
    
    // Increase weights for objectives that correlate with good performance
    double avg_performance = 0.0;
    for (double perf : recent_performance) {
        avg_performance += perf;
    }
    avg_performance /= recent_performance.size();
    
    // Adjust weights based on performance trend
    for (int i = 0; i < new_weights.size(0); ++i) {
        double adjustment = (avg_performance > 0.5) ? 1.05 : 0.95; // Simple heuristic
        new_weights[i] = current_weights[i] * adjustment;
    }
    
    // Normalize weights
    new_weights = new_weights / new_weights.sum();
    
    // Update weight history
    weight_history_ = torch::cat({weight_history_, new_weights.unsqueeze(-1)}, -1);
    
    // Keep only recent history
    int max_history = 100;
    if (weight_history_.size(-1) > max_history) {
        weight_history_ = weight_history_.narrow(-1, weight_history_.size(-1) - max_history, max_history);
    }
}

} // namespace crlgru
