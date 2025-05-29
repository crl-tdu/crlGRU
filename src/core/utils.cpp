#include "crlgru/crl_gru.hpp"
#include <fstream>
#include <cmath>
#include <algorithm>

namespace crlgru {
namespace utils {

torch::Tensor cartesian_to_polar_map(const torch::Tensor& positions,
                                    const torch::Tensor& self_position,
                                    int num_rings, int num_sectors,
                                    double max_range) {
    auto batch_size = positions.size(0);
    auto num_agents = positions.size(1);
    
    // Initialize polar map
    auto polar_map = torch::zeros({batch_size, num_rings, num_sectors});
    
    for (int b = 0; b < batch_size; ++b) {
        for (int a = 0; a < num_agents; ++a) {
            // Get relative position
            auto rel_pos = positions[b][a] - self_position[b];
            auto x = rel_pos[0].item<double>();
            auto y = rel_pos[1].item<double>();
            
            // Convert to polar coordinates
            double distance = std::sqrt(x * x + y * y);
            double angle = std::atan2(y, x);
            
            // Normalize angle to [0, 2*pi]
            if (angle < 0) angle += 2 * M_PI;
            
            // Map to grid indices
            int ring_idx = std::min(static_cast<int>(distance / max_range * num_rings), num_rings - 1);
            int sector_idx = std::min(static_cast<int>(angle / (2 * M_PI) * num_sectors), num_sectors - 1);
            
            if (distance <= max_range) {
                polar_map[b][ring_idx][sector_idx] += 1.0;
            }
        }
    }
    
    return polar_map;
}

double compute_mutual_information(const torch::Tensor& state1, 
                                const torch::Tensor& state2) {
    // Simplified mutual information computation using correlation
    auto normalized_state1 = (state1 - state1.mean()) / (state1.std() + 1e-8);
    auto normalized_state2 = (state2 - state2.mean()) / (state2.std() + 1e-8);
    
    auto correlation = torch::corrcoef(torch::stack({normalized_state1, normalized_state2}))[0][1];
    
    // Convert correlation to approximate mutual information
    double corr_val = correlation.item<double>();
    return -0.5 * std::log(1 - corr_val * corr_val + 1e-8);
}

torch::Tensor apply_gaussian_kernel(const torch::Tensor& input, 
                                  double sigma, int kernel_size) {
    // Create Gaussian kernel
    auto kernel = torch::zeros({kernel_size, kernel_size});
    int center = kernel_size / 2;
    double sigma_sq = sigma * sigma;
    
    for (int i = 0; i < kernel_size; ++i) {
        for (int j = 0; j < kernel_size; ++j) {
            double dx = i - center;
            double dy = j - center;
            double dist_sq = dx * dx + dy * dy;
            kernel[i][j] = std::exp(-dist_sq / (2 * sigma_sq));
        }
    }
    
    // Normalize kernel
    kernel = kernel / kernel.sum();
    
    // Apply convolution (simplified)
    auto result = torch::conv2d(input.unsqueeze(0).unsqueeze(0), 
                               kernel.unsqueeze(0).unsqueeze(0),
                               /*bias=*/torch::Tensor(), 
                               /*stride=*/1, 
                               /*padding=*/kernel_size/2);
    
    return result.squeeze(0).squeeze(0);
}

double compute_trust_metric(const std::vector<double>& performance_history,
                          double distance, double max_distance) {
    if (performance_history.empty()) {
        return 0.0;
    }
    
    // Compute average performance
    double avg_performance = 0.0;
    for (double perf : performance_history) {
        avg_performance += perf;
    }
    avg_performance /= performance_history.size();
    
    // Compute performance stability (inverse of variance)
    double variance = 0.0;
    for (double perf : performance_history) {
        double diff = perf - avg_performance;
        variance += diff * diff;
    }
    variance /= performance_history.size();
    double stability = 1.0 / (1.0 + variance);
    
    // Distance factor (closer is better)
    double distance_factor = 1.0 - (distance / max_distance);
    distance_factor = std::max(0.0, distance_factor);
    
    // Combine factors
    return avg_performance * stability * distance_factor;
}

void save_parameters(const std::string& filename, 
                    const std::unordered_map<std::string, torch::Tensor>& params) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for writing: " + filename);
    }
    
    // Write number of parameters
    size_t num_params = params.size();
    file.write(reinterpret_cast<const char*>(&num_params), sizeof(num_params));
    
    for (const auto& param_pair : params) {
        const std::string& name = param_pair.first;
        const torch::Tensor& tensor = param_pair.second;
        
        // Write parameter name
        size_t name_size = name.size();
        file.write(reinterpret_cast<const char*>(&name_size), sizeof(name_size));
        file.write(name.c_str(), name_size);
        
        // Write tensor metadata
        auto sizes = tensor.sizes();
        size_t num_dims = sizes.size();
        file.write(reinterpret_cast<const char*>(&num_dims), sizeof(num_dims));
        
        for (size_t i = 0; i < num_dims; ++i) {
            int64_t dim_size = sizes[i];
            file.write(reinterpret_cast<const char*>(&dim_size), sizeof(dim_size));
        }
        
        // Write tensor data
        auto cpu_tensor = tensor.cpu();
        auto data_ptr = cpu_tensor.data_ptr<float>();
        size_t data_size = cpu_tensor.numel() * sizeof(float);
        file.write(reinterpret_cast<const char*>(data_ptr), data_size);
    }
    
    file.close();
}

std::unordered_map<std::string, torch::Tensor> 
load_parameters(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for reading: " + filename);
    }
    
    std::unordered_map<std::string, torch::Tensor> params;
    
    // Read number of parameters
    size_t num_params;
    file.read(reinterpret_cast<char*>(&num_params), sizeof(num_params));
    
    for (size_t p = 0; p < num_params; ++p) {
        // Read parameter name
        size_t name_size;
        file.read(reinterpret_cast<char*>(&name_size), sizeof(name_size));
        
        std::string name(name_size, '\0');
        file.read(&name[0], name_size);
        
        // Read tensor metadata
        size_t num_dims;
        file.read(reinterpret_cast<char*>(&num_dims), sizeof(num_dims));
        
        std::vector<int64_t> sizes(num_dims);
        for (size_t i = 0; i < num_dims; ++i) {
            file.read(reinterpret_cast<char*>(&sizes[i]), sizeof(int64_t));
        }
        
        // Create tensor and read data
        auto tensor = torch::zeros(sizes);
        auto data_ptr = tensor.data_ptr<float>();
        size_t data_size = tensor.numel() * sizeof(float);
        file.read(reinterpret_cast<char*>(data_ptr), data_size);
        
        params[name] = tensor;
    }
    
    file.close();
    return params;
}

} // namespace utils
} // namespace crlgru
