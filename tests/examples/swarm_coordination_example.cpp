#include "crlgru/crl_gru.hpp"
#include <iostream>
#include <vector>
#include <random>

/**
 * @brief Simple swarm coordination example using FEP-GRU
 * 
 * This example demonstrates:
 * - Multi-agent coordination with FEP-GRU networks
 * - Real-time parameter sharing and imitation learning
 * - Polar coordinate spatial attention
 * - Meta-evaluation based optimization
 */

class SwarmAgent {
public:
    struct State {
        torch::Tensor position;     // [x, y]
        torch::Tensor velocity;     // [vx, vy]
        torch::Tensor orientation;  // [theta]
        torch::Tensor goal;         // [goal_x, goal_y]
    };

private:
    int agent_id_;
    State current_state_;
    std::shared_ptr<crlgru::FEPGRUNetwork> brain_;
    std::shared_ptr<crlgru::PolarSpatialAttention> attention_;
    std::shared_ptr<crlgru::MetaEvaluator> evaluator_;
    std::shared_ptr<crlgru::SPSAOptimizer> optimizer_;
    
    std::vector<torch::Tensor> observation_history_;
    std::vector<State> neighbor_states_;
    double performance_score_;

public:
    SwarmAgent(int id, const torch::Tensor& initial_position, const torch::Tensor& goal) 
        : agent_id_(id), performance_score_(0.0) {
        
        // Initialize state
        current_state_.position = initial_position.clone();
        current_state_.velocity = torch::zeros({2});
        current_state_.orientation = torch::zeros({1});
        current_state_.goal = goal.clone();
        
        // Create FEP-GRU network
        crlgru::FEPGRUNetwork::NetworkConfig network_config;
        network_config.layer_sizes = {64, 128, 64};
        network_config.cell_config.input_size = 64;
        network_config.cell_config.hidden_size = 128;
        network_config.cell_config.enable_hierarchical_imitation = true;
        network_config.cell_config.enable_som_extraction = true;
        
        brain_ = std::make_shared<crlgru::FEPGRUNetwork>(network_config);
        
        // Create spatial attention module
        crlgru::PolarSpatialAttention::AttentionConfig attention_config;
        attention_config.input_channels = 64;
        attention_config.num_distance_rings = 8;
        attention_config.num_angle_sectors = 16;
        
        attention_ = std::make_shared<crlgru::PolarSpatialAttention>(attention_config);
        
        // Create meta-evaluator
        crlgru::MetaEvaluator::EvaluationConfig eval_config;
        eval_config.objective_weights = {1.0, 1.5, 0.8, 0.7}; // goal, collision, cohesion, alignment
        eval_config.adaptive_weights = true;
        
        evaluator_ = std::make_shared<crlgru::MetaEvaluator>(eval_config);
        
        // Create SPSA optimizer
        crlgru::SPSAOptimizer::SPSAConfig spsa_config;
        spsa_config.learning_rate = 0.01;
        spsa_config.max_iterations = 100;
        
        optimizer_ = std::make_shared<crlgru::SPSAOptimizer>(spsa_config);
    }
    
    torch::Tensor observe_environment(const std::vector<SwarmAgent*>& all_agents) {
        // Collect neighbor positions
        std::vector<torch::Tensor> neighbor_positions;
        neighbor_states_.clear();
        
        for (auto* other_agent : all_agents) {
            if (other_agent->agent_id_ != agent_id_) {
                auto distance = torch::norm(other_agent->current_state_.position - current_state_.position);
                if (distance.item<double>() < 20.0) { // Observation range
                    neighbor_positions.push_back(other_agent->current_state_.position);
                    neighbor_states_.push_back(other_agent->current_state_);
                }
            }
        }
        
        // Convert to polar coordinate map
        if (!neighbor_positions.empty()) {
            auto positions_tensor = torch::stack(neighbor_positions).unsqueeze(0); // [1, N, 2]
            auto self_position = current_state_.position.unsqueeze(0); // [1, 2]
            
            auto polar_map = crlgru::utils::cartesian_to_polar_map(
                positions_tensor, self_position, 8, 16, 20.0);
            
            // Expand to match attention input requirements
            auto expanded_map = polar_map.unsqueeze(1).expand({1, 64, 8, 16});
            
            // Apply spatial attention
            auto attended_map = attention_->forward(expanded_map);
            
            // Flatten for GRU input
            auto observation = attended_map.flatten(1); // [1, 64*8*16]
            
            // Add self-state information
            auto self_state = torch::cat({
                current_state_.position,
                current_state_.velocity,
                current_state_.orientation,
                current_state_.goal
            });
            
            // Pad to match input size
            int target_size = 64;
            if (observation.size(1) + self_state.size(0) < target_size) {
                auto padding = torch::zeros({1, target_size - observation.size(1) - self_state.size(0)});
                observation = torch::cat({observation, self_state.unsqueeze(0), padding}, 1);
            } else {
                observation = observation.narrow(1, 0, target_size - self_state.size(0));
                observation = torch::cat({observation, self_state.unsqueeze(0)}, 1);
            }
            
            return observation;
        } else {
            // No neighbors, use self-state only
            auto self_state = torch::cat({
                current_state_.position,
                current_state_.velocity,
                current_state_.orientation,
                current_state_.goal
            });
            
            auto observation = torch::zeros({1, 64});
            observation.narrow(1, 0, self_state.size(0)).copy_(self_state);
            
            return observation;
        }
    }
    
    void update(const std::vector<SwarmAgent*>& all_agents) {
        // Observe environment
        auto observation = observe_environment(all_agents);
        observation_history_.push_back(observation);
        
        // Keep only recent history
        if (observation_history_.size() > 50) {
            observation_history_.erase(observation_history_.begin());
        }
        
        // Create sequence for GRU
        torch::Tensor sequence;
        if (observation_history_.size() >= 10) {
            auto recent_obs = std::vector<torch::Tensor>(
                observation_history_.end() - 10, observation_history_.end());
            sequence = torch::stack(recent_obs, 1); // [1, seq_len, input_size]
        } else {
            // Pad sequence if not enough history
            auto padded_obs = observation_history_;
            while (padded_obs.size() < 10) {
                padded_obs.insert(padded_obs.begin(), torch::zeros_like(observation));
            }
            sequence = torch::stack(padded_obs, 1);
        }
        
        // Forward pass through FEP-GRU network
        auto [hidden_states, predictions, free_energies] = brain_->forward(sequence);
        
        // Extract action from final hidden state
        auto final_hidden = hidden_states.select(1, -1); // Last time step
        auto action = torch::tanh(torch::mm(final_hidden, torch::randn({final_hidden.size(1), 3})));
        
        // Apply action to update state
        auto force = action.narrow(1, 0, 2) * 0.5; // Limit force magnitude
        auto angular_velocity = action.select(1, 2) * 0.1;
        
        // Simple dynamics
        current_state_.velocity = current_state_.velocity + force.squeeze();
        current_state_.velocity = current_state_.velocity * 0.95; // Damping
        current_state_.position = current_state_.position + current_state_.velocity * 0.1;
        current_state_.orientation = current_state_.orientation + angular_velocity;
        
        // Evaluate performance
        auto predicted_future = predictions.select(1, -1); // Final prediction
        performance_score_ = evaluator_->evaluate(predicted_future, 
                                                final_hidden, 
                                                torch::zeros({1, 10}));
        
        // Share parameters with high-performing neighbors
        share_parameters_with_neighbors(all_agents);
    }
    
    void share_parameters_with_neighbors(const std::vector<SwarmAgent*>& all_agents) {
        // Find best performing neighbor
        SwarmAgent* best_neighbor = nullptr;
        double best_performance = performance_score_;
        
        for (auto* other_agent : all_agents) {
            if (other_agent->agent_id_ != agent_id_ && 
                other_agent->performance_score_ > best_performance) {
                auto distance = torch::norm(other_agent->current_state_.position - current_state_.position);
                if (distance.item<double>() < 15.0) { // Communication range
                    best_performance = other_agent->performance_score_;
                    best_neighbor = other_agent;
                }
            }
        }
        
        // Imitate best neighbor's parameters
        if (best_neighbor != nullptr) {
            std::unordered_map<std::string, torch::Tensor> neighbor_params;
            for (auto& param_pair : best_neighbor->brain_->named_parameters()) {
                neighbor_params[param_pair.key()] = param_pair.value();
            }
            
            // Update each layer with neighbor's parameters
            brain_->share_parameters_with_agents({best_neighbor->agent_id_});
        }
    }
    
    // Getters
    int get_id() const { return agent_id_; }
    const State& get_state() const { return current_state_; }
    double get_performance() const { return performance_score_; }
    
    torch::Tensor get_som_features() const {
        auto features = brain_->extract_multi_layer_som_features();
        if (!features.empty()) {
            return torch::cat(features, -1);
        }
        return torch::zeros({1, 64});
    }
};

class SwarmSimulation {
private:
    std::vector<std::unique_ptr<SwarmAgent>> agents_;
    int num_agents_;
    int current_step_;
    int max_steps_;

public:
    SwarmSimulation(int num_agents, int max_steps) 
        : num_agents_(num_agents), current_step_(0), max_steps_(max_steps) {
        
        // Initialize agents with random positions and goals
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> pos_dist(-50.0, 50.0);
        
        for (int i = 0; i < num_agents_; ++i) {
            auto position = torch::tensor({pos_dist(gen), pos_dist(gen)});
            auto goal = torch::tensor({pos_dist(gen), pos_dist(gen)});
            
            agents_.push_back(std::make_unique<SwarmAgent>(i, position, goal));
        }
    }
    
    void run() {
        std::cout << "Starting swarm simulation with " << num_agents_ << " agents..." << std::endl;
        
        for (current_step_ = 0; current_step_ < max_steps_; ++current_step_) {
            // Create raw pointers for agents (safe since we control lifetime)
            std::vector<SwarmAgent*> agent_ptrs;
            for (auto& agent : agents_) {
                agent_ptrs.push_back(agent.get());
            }
            
            // Update all agents
            for (auto& agent : agents_) {
                agent->update(agent_ptrs);
            }
            
            // Print progress
            if (current_step_ % 20 == 0) { // Changed from 100 to 20
                print_status();
            }
        }
        
        std::cout << "Simulation completed!" << std::endl;
        print_final_results();
    }
    
    void print_status() {
        double avg_performance = 0.0;
        double avg_distance_to_goal = 0.0;
        
        for (const auto& agent : agents_) {
            avg_performance += agent->get_performance();
            
            auto distance_to_goal = torch::norm(
                agent->get_state().position - agent->get_state().goal);
            avg_distance_to_goal += distance_to_goal.item<double>();
        }
        
        avg_performance /= num_agents_;
        avg_distance_to_goal /= num_agents_;
        
        std::cout << "Step " << current_step_ << "/" << max_steps_ 
                  << " - Avg Performance: " << avg_performance
                  << " - Avg Distance to Goal: " << avg_distance_to_goal << std::endl;
    }
    
    void print_final_results() {
        std::cout << "\n=== Final Results ===" << std::endl;
        
        for (const auto& agent : agents_) {
            auto distance_to_goal = torch::norm(
                agent->get_state().position - agent->get_state().goal);
            
            std::cout << "Agent " << agent->get_id() 
                      << " - Performance: " << agent->get_performance()
                      << " - Distance to Goal: " << distance_to_goal.item<double>() << std::endl;
        }
        
        // Save SOM visualization data
        save_som_data();
    }
    
    void save_som_data() {
        std::cout << "\nSaving SOM visualization data..." << std::endl;
        
        std::vector<torch::Tensor> all_som_features;
        for (const auto& agent : agents_) {
            all_som_features.push_back(agent->get_som_features());
        }
        
        if (!all_som_features.empty()) {
            auto som_data = torch::stack(all_som_features, 0);
            
            // Save parameters for analysis
            std::unordered_map<std::string, torch::Tensor> som_params;
            som_params["som_features"] = som_data;
            som_params["agent_positions"] = torch::zeros({num_agents_, 2});
            som_params["agent_goals"] = torch::zeros({num_agents_, 2});
            
            for (int i = 0; i < num_agents_; ++i) {
                som_params["agent_positions"][i] = agents_[i]->get_state().position;
                som_params["agent_goals"][i] = agents_[i]->get_state().goal;
            }
            
            crlgru::utils::save_parameters("som_visualization_data.bin", som_params);
            std::cout << "SOM data saved to som_visualization_data.bin" << std::endl;
        }
    }
};

int main() {
    try {
        // Set random seed for reproducibility
        torch::manual_seed(42);
        
        // Create and run simulation
        SwarmSimulation simulation(10, 100); // Reduced from 1000 to 100 steps
        simulation.run();
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
}
