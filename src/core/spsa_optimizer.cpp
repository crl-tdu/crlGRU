#include "crlgru/crl_gru.hpp"
#include <random>
#include <cmath>

namespace crlgru {

SPSAOptimizer::SPSAOptimizer(const SPSAConfig& config) : config_(config), iteration_count_(0) {
    reset();
}

torch::Tensor SPSAOptimizer::optimize(torch::Tensor& parameters, 
                                    std::function<double(const torch::Tensor&)> objective_function) {
    auto current_params = parameters.clone();
    double current_objective = objective_function(current_params);
    double prev_objective = current_objective;
    
    for (int iter = 0; iter < config_.max_iterations; ++iter) {
        prev_objective = current_objective;
        
        // Estimate gradient
        auto gradient = estimate_gradient(current_params, objective_function);
        
        // Update gradient estimate with smoothing
        if (!gradient_estimate_.defined()) {
            gradient_estimate_ = gradient.clone();
        } else {
            gradient_estimate_ = config_.gradient_smoothing * gradient_estimate_ + 
                              (1.0 - config_.gradient_smoothing) * gradient;
        }
        
        // Update parameters (always update in SPSA)
        current_params = current_params + config_.learning_rate * gradient_estimate_;
        current_objective = objective_function(current_params);
        
        // Check convergence
        if (std::abs(current_objective - prev_objective) < config_.tolerance) {
            break;
        }
        
        iteration_count_++;
    }
    
    parameters = current_params;
    return current_params;
}

torch::Tensor SPSAOptimizer::estimate_gradient(const torch::Tensor& parameters,
                                             std::function<double(const torch::Tensor&)> objective_function) {
    // Generate random perturbation vector
    std::random_device rd;
    std::mt19937 gen(rd());
    std::bernoulli_distribution dist(0.5);
    
    auto perturbation = torch::zeros_like(parameters);
    for (int i = 0; i < parameters.numel(); ++i) {
        perturbation.view(-1)[i] = dist(gen) ? 1.0 : -1.0;
    }
    
    // Compute function values at perturbed points
    auto params_plus = parameters + config_.perturbation_magnitude * perturbation;
    auto params_minus = parameters - config_.perturbation_magnitude * perturbation;
    
    double f_plus = objective_function(params_plus);
    double f_minus = objective_function(params_minus);
    
    // Estimate gradient using finite differences (element-wise)
    auto gradient = (f_plus - f_minus) / (2.0 * config_.perturbation_magnitude) / perturbation;
    
    return gradient;
}

void SPSAOptimizer::reset() {
    parameter_history_ = torch::Tensor();
    gradient_estimate_ = torch::Tensor();
    iteration_count_ = 0;
}

} // namespace crlgru
