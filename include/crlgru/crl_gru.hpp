#ifndef CRL_GRU_HPP
#define CRL_GRU_HPP

#include <torch/torch.h>
#include <vector>
#include <memory>
#include <unordered_map>
#include <functional>

namespace crl {
namespace gru {

/**
 * @brief Free Energy Principle based GRU cell for embodied swarm intelligence
 * 
 * This implementation extends standard GRU with:
 * - Real-time parameter modification for mutual imitation
 * - Internal state extraction for SOM (Self-Organizing Map)
 * - Variational free energy computation
 * - Predictive coding mechanisms
 * - Meta-evaluation function integration
 */
class FEPGRUCell : public torch::nn::Module {
public:
    struct Config {
        int input_size = 64;
        int hidden_size = 128;
        bool bias = true;
        double dropout = 0.0;
        
        // FEP specific parameters
        double free_energy_weight = 1.0;
        double prediction_horizon = 10.0;
        double variational_beta = 1.0;
        
        // Imitation learning parameters
        double imitation_rate = 0.1;
        double trust_radius = 5.0;
        bool enable_hierarchical_imitation = true;
        
        // SOM integration
        bool enable_som_extraction = true;
        int som_grid_size = 16;
        double som_learning_rate = 0.01;
    };

private:
    Config config_;
    
    // Standard GRU parameters
    torch::nn::Linear input_to_hidden_{nullptr};
    torch::nn::Linear hidden_to_hidden_{nullptr};
    torch::nn::Linear input_to_reset_{nullptr};
    torch::nn::Linear hidden_to_reset_{nullptr};
    torch::nn::Linear input_to_update_{nullptr};
    torch::nn::Linear hidden_to_update_{nullptr};
    
    // FEP specific layers
    torch::nn::Linear prediction_head_{nullptr};
    torch::nn::Linear variance_head_{nullptr};
    torch::nn::Linear meta_evaluation_head_{nullptr};
    
    // Internal state management
    torch::Tensor hidden_state_;
    torch::Tensor prediction_error_;
    torch::Tensor free_energy_;
    
    // Imitation learning storage
    std::unordered_map<int, torch::Tensor> peer_parameters_;
    std::unordered_map<int, double> peer_performance_;
    std::unordered_map<int, double> peer_trust_;
    
    // SOM related
    torch::Tensor som_weights_;
    torch::Tensor som_activation_history_;

public:
    explicit FEPGRUCell(const Config& config);
    
    /**
     * @brief Forward pass with predictive coding
     * @param input Input tensor [batch_size, input_size]
     * @param hidden Previous hidden state [batch_size, hidden_size]
     * @return Tuple of (new_hidden, prediction, free_energy)
     */
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> 
    forward(const torch::Tensor& input, const torch::Tensor& hidden = {});
    
    /**
     * @brief Compute variational free energy
     */
    torch::Tensor compute_free_energy(const torch::Tensor& prediction, 
                                    const torch::Tensor& target,
                                    const torch::Tensor& variance);
    
    /**
     * @brief Real-time parameter modification for imitation learning
     */
    void update_parameters_from_peer(int peer_id, 
                                   const std::unordered_map<std::string, torch::Tensor>& peer_params,
                                   double peer_performance);
    
    /**
     * @brief Hierarchical imitation at three levels
     */
    void hierarchical_imitation_update(int best_peer_id);
    
    /**
     * @brief Extract internal state for SOM
     */
    torch::Tensor extract_som_features() const;
    
    /**
     * @brief Update SOM based on current internal state
     */
    void update_som(const torch::Tensor& input_pattern);
    
    /**
     * @brief Get current meta-evaluation score
     */
    double get_meta_evaluation() const;
    
    /**
     * @brief Reset internal states
     */
    void reset_states();
    
    // Getters
    const torch::Tensor& get_hidden_state() const { return hidden_state_; }
    const torch::Tensor& get_prediction_error() const { return prediction_error_; }
    const torch::Tensor& get_free_energy() const { return free_energy_; }
    const torch::Tensor& get_som_weights() const { return som_weights_; }
    
    // Configuration
    const Config& get_config() const { return config_; }
    void set_config(const Config& config) { config_ = config; }
};

/**
 * @brief Multi-layer FEP-GRU for complex temporal modeling
 */
class FEPGRUNetwork : public torch::nn::Module {
public:
    struct NetworkConfig {
        std::vector<int> layer_sizes = {64, 128, 64};
        FEPGRUCell::Config cell_config;
        bool bidirectional = false;
        double layer_dropout = 0.1;
        int sequence_length = 50;
    };

private:
    NetworkConfig config_;
    std::vector<std::shared_ptr<FEPGRUCell>> layers_;
    torch::nn::Dropout dropout_{nullptr};
    
    // Multi-agent coordination
    std::unordered_map<int, std::vector<torch::Tensor>> agent_states_;
    std::unordered_map<int, double> agent_performance_history_;

public:
    explicit FEPGRUNetwork(const NetworkConfig& config);
    
    /**
     * @brief Process sequence with multi-layer FEP-GRU
     */
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    forward(const torch::Tensor& sequence);
    
    /**
     * @brief Multi-agent parameter sharing and imitation
     */
    void share_parameters_with_agents(const std::vector<int>& agent_ids);
    
    /**
     * @brief Collective free energy minimization
     */
    torch::Tensor compute_collective_free_energy() const;
    
    /**
     * @brief Get layer-wise SOM features
     */
    std::vector<torch::Tensor> extract_multi_layer_som_features() const;
    
    // Agent management
    void register_agent(int agent_id, double initial_performance = 0.0);
    void update_agent_performance(int agent_id, double performance);
    void remove_agent(int agent_id);
};

/**
 * @brief Spatial-Temporal Attention for polar coordinate maps
 */
class PolarSpatialAttention : public torch::nn::Module {
public:
    struct AttentionConfig {
        int input_channels = 64;
        int attention_dim = 32;
        int num_distance_rings = 8;
        int num_angle_sectors = 16;
        bool adaptive_resolution = true;
        double attention_dropout = 0.1;
    };

private:
    AttentionConfig config_;
    torch::nn::Conv2d distance_attention_{nullptr};
    torch::nn::Conv2d angle_attention_{nullptr};
    torch::nn::Linear fusion_layer_{nullptr};
    torch::nn::Dropout dropout_{nullptr};

public:
    explicit PolarSpatialAttention(const AttentionConfig& config);
    
    /**
     * @brief Apply spatial attention to polar coordinate map
     */
    torch::Tensor forward(const torch::Tensor& polar_map);
    
    /**
     * @brief Compute attention weights for distance and angle
     */
    std::pair<torch::Tensor, torch::Tensor> compute_attention_weights(const torch::Tensor& features);
};

/**
 * @brief Meta-evaluation function for multi-objective optimization
 */
class MetaEvaluator {
public:
    struct EvaluationConfig {
        std::vector<double> objective_weights = {1.0, 1.0, 1.0, 1.0}; // goal, collision, cohesion, alignment
        double temporal_discount = 0.95;
        int evaluation_horizon = 10;
        bool adaptive_weights = true;
    };

private:
    EvaluationConfig config_;
    std::vector<std::function<double(const torch::Tensor&)>> objective_functions_;
    torch::Tensor weight_history_;

public:
    explicit MetaEvaluator(const EvaluationConfig& config);
    
    /**
     * @brief Evaluate predicted future states
     */
    double evaluate(const torch::Tensor& predicted_states, 
                   const torch::Tensor& current_state,
                   const torch::Tensor& environment_state);
    
    /**
     * @brief Add custom objective function
     */
    void add_objective(std::function<double(const torch::Tensor&)> objective);
    
    /**
     * @brief Adapt objective weights based on performance
     */
    void adapt_weights(const std::vector<double>& recent_performance);
};

/**
 * @brief SPSA (Simultaneous Perturbation Stochastic Approximation) optimizer
 */
class SPSAOptimizer {
public:
    struct SPSAConfig {
        double learning_rate = 0.01;
        double perturbation_magnitude = 0.1;
        double gradient_smoothing = 0.9;
        int max_iterations = 1000;
        double tolerance = 1e-6;
    };

private:
    SPSAConfig config_;
    torch::Tensor parameter_history_;
    torch::Tensor gradient_estimate_;
    int iteration_count_;

public:
    explicit SPSAOptimizer(const SPSAConfig& config);
    
    /**
     * @brief Optimize parameters using SPSA
     */
    torch::Tensor optimize(torch::Tensor& parameters, 
                          std::function<double(const torch::Tensor&)> objective_function);
    
    /**
     * @brief Estimate gradient using simultaneous perturbation
     */
    torch::Tensor estimate_gradient(const torch::Tensor& parameters,
                                  std::function<double(const torch::Tensor&)> objective_function);
    
    /**
     * @brief Reset optimizer state
     */
    void reset();
};

/**
 * @brief Utility functions for swarm intelligence research
 */
namespace utils {
    /**
     * @brief Generate polar coordinate map from Cartesian positions
     */
    torch::Tensor cartesian_to_polar_map(const torch::Tensor& positions,
                                        const torch::Tensor& self_position,
                                        int num_rings, int num_sectors,
                                        double max_range);
    
    /**
     * @brief Compute mutual information between agents
     */
    double compute_mutual_information(const torch::Tensor& state1, 
                                    const torch::Tensor& state2);
    
    /**
     * @brief Apply Gaussian kernel for spatial smoothing
     */
    torch::Tensor apply_gaussian_kernel(const torch::Tensor& input, 
                                      double sigma, int kernel_size);
    
    /**
     * @brief Compute trust metric based on performance history
     */
    double compute_trust_metric(const std::vector<double>& performance_history,
                              double distance, double max_distance);
    
    /**
     * @brief Save/Load model parameters for distributed learning
     */
    void save_parameters(const std::string& filename, 
                        const std::unordered_map<std::string, torch::Tensor>& params);
    
    std::unordered_map<std::string, torch::Tensor> 
    load_parameters(const std::string& filename);
} // namespace utils

} // namespace gru
} // namespace crl

#endif // CRL_GRU_HPP
