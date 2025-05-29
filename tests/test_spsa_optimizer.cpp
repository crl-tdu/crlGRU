#include "crlgru/crl_gru.hpp"
#include <iostream>
#include <cassert>
#include <cmath>

bool test_spsa_optimizer_creation() {
    try {
        crlgru::SPSAOptimizer::SPSAConfig config;
        config.learning_rate = 0.01;
        config.perturbation_magnitude = 0.1;
        config.max_iterations = 100;
        
        auto optimizer = std::make_shared<crlgru::SPSAOptimizer>(config);
        
        std::cout << "✓ SPSA optimizer creation test passed" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "✗ SPSA optimizer creation test failed: " << e.what() << std::endl;
        return false;
    }
}

bool test_spsa_gradient_estimation() {
    try {
        crlgru::SPSAOptimizer::SPSAConfig config;
        config.learning_rate = 0.01;
        config.perturbation_magnitude = 0.1;
        config.max_iterations = 10;
        
        auto optimizer = std::make_shared<crlgru::SPSAOptimizer>(config);
        
        // Simple quadratic objective: minimize ||x - target||^2
        auto target = torch::tensor({1.0, 2.0, -1.0});
        auto parameters = torch::tensor({0.0, 0.0, 0.0});
        
        auto objective_function = [&target](const torch::Tensor& params) -> double {
            auto diff = params - target;
            return -torch::sum(diff * diff).item<double>(); // Negative for maximization
        };
        
        auto gradient = optimizer->estimate_gradient(parameters, objective_function);
        
        assert(gradient.defined());
        assert(gradient.numel() == 3);
        assert(gradient.isfinite().all().item<bool>());
        
        std::cout << "✓ SPSA gradient estimation test passed" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "✗ SPSA gradient estimation test failed: " << e.what() << std::endl;
        return false;
    }
}

bool test_spsa_optimization() {
    try {
        crlgru::SPSAOptimizer::SPSAConfig config;
        config.learning_rate = 0.1;
        config.perturbation_magnitude = 0.1;
        config.max_iterations = 50;
        config.tolerance = 1e-3;
        
        auto optimizer = std::make_shared<crlgru::SPSAOptimizer>(config);
        
        // Simple quadratic objective: minimize ||x - target||^2
        auto target = torch::tensor({1.0, 2.0, -1.0});
        auto initial_params = torch::tensor({0.0, 0.0, 0.0});
        
        auto objective_function = [&target](const torch::Tensor& params) -> double {
            auto diff = params - target;
            return -torch::sum(diff * diff).item<double>(); // Negative for maximization
        };
        
        auto initial_distance = torch::norm(initial_params - target).item<double>();
        
        auto optimized_params = optimizer->optimize(initial_params, objective_function);
        
        auto final_distance = torch::norm(optimized_params - target).item<double>();
        
        // Should improve (reduce distance to target)
        assert(final_distance < initial_distance);
        assert(final_distance < 1.0); // Should get reasonably close
        
        std::cout << "✓ SPSA optimization test passed (distance reduced from " 
                  << initial_distance << " to " << final_distance << ")" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "✗ SPSA optimization test failed: " << e.what() << std::endl;
        return false;
    }
}

bool test_spsa_reset() {
    try {
        crlgru::SPSAOptimizer::SPSAConfig config;
        config.learning_rate = 0.01;
        config.perturbation_magnitude = 0.1;
        config.max_iterations = 10;
        
        auto optimizer = std::make_shared<crlgru::SPSAOptimizer>(config);
        
        // Run some optimization to populate internal state
        auto target = torch::tensor({1.0, 2.0});
        auto params = torch::tensor({0.0, 0.0});
        
        auto objective_function = [&target](const torch::Tensor& p) -> double {
            return -torch::sum((p - target) * (p - target)).item<double>();
        };
        
        optimizer->optimize(params, objective_function);
        
        // Reset should work without throwing
        optimizer->reset();
        
        std::cout << "✓ SPSA reset test passed" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "✗ SPSA reset test failed: " << e.what() << std::endl;
        return false;
    }
}

bool test_spsa_convergence() {
    try {
        crlgru::SPSAOptimizer::SPSAConfig config;
        config.learning_rate = 0.05;
        config.perturbation_magnitude = 0.05;
        config.max_iterations = 100;
        config.tolerance = 1e-4;
        config.gradient_smoothing = 0.9;
        
        auto optimizer = std::make_shared<crlgru::SPSAOptimizer>(config);
        
        // Convex quadratic objective
        auto target = torch::tensor({3.0, -2.0, 1.5});
        auto initial_params = torch::randn({3}) * 2.0; // Random start
        
        auto objective_function = [&target](const torch::Tensor& params) -> double {
            auto diff = params - target;
            return -torch::sum(diff * diff).item<double>();
        };
        
        auto optimized_params = optimizer->optimize(initial_params, objective_function);
        
        // Check convergence
        auto final_error = torch::norm(optimized_params - target).item<double>();
        assert(final_error < 0.5); // Should converge reasonably well
        
        std::cout << "✓ SPSA convergence test passed (final error: " 
                  << final_error << ")" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "✗ SPSA convergence test failed: " << e.what() << std::endl;
        return false;
    }
}

bool test_spsa_robustness() {
    try {
        crlgru::SPSAOptimizer::SPSAConfig config;
        config.learning_rate = 0.02;
        config.perturbation_magnitude = 0.1;
        config.max_iterations = 50;
        
        auto optimizer = std::make_shared<crlgru::SPSAOptimizer>(config);
        
        // Noisy objective function
        auto target = torch::tensor({1.0, -1.0});
        auto params = torch::tensor({0.0, 0.0});
        
        auto objective_function = [&target](const torch::Tensor& p) -> double {
            auto noise = torch::randn(1).item<double>() * 0.1;
            auto dist = torch::sum((p - target) * (p - target)).item<double>();
            return -(dist + noise); // Add noise to objective
        };
        
        auto optimized_params = optimizer->optimize(params, objective_function);
        
        // Should still make progress despite noise
        auto final_distance = torch::norm(optimized_params - target).item<double>();
        auto initial_distance = torch::norm(params - target).item<double>();
        
        assert(final_distance < initial_distance);
        
        std::cout << "✓ SPSA robustness test passed" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "✗ SPSA robustness test failed: " << e.what() << std::endl;
        return false;
    }
}

int main() {
    std::cout << "=== SPSA Optimizer Tests ===" << std::endl;
    
    int passed = 0;
    int total = 6;
    
    if (test_spsa_optimizer_creation()) passed++;
    if (test_spsa_gradient_estimation()) passed++;
    if (test_spsa_optimization()) passed++;
    if (test_spsa_reset()) passed++;
    if (test_spsa_convergence()) passed++;
    if (test_spsa_robustness()) passed++;
    
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
