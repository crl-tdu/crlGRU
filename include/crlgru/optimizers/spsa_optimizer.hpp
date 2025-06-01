#ifndef CRLGRU_OPTIMIZERS_SPSA_OPTIMIZER_HPP
#define CRLGRU_OPTIMIZERS_SPSA_OPTIMIZER_HPP

/// @file spsa_optimizer.hpp
/// @brief SPSA最適化器のヘッダーオンリー実装

#include <crlgru/common.hpp>
#include <crlgru/utils/config_types.hpp>
#include <random>
#include <cmath>
#include <algorithm>
#include <type_traits>

namespace crlgru {

/// @brief 同時摂動確率近似（SPSA）最適化器
/// @details 勾配フリー最適化アルゴリズムのテンプレート実装
template<typename FloatType = double>
class SPSAOptimizer {
public:
    /// @brief SPSA設定構造体（テンプレート特化版）
    struct Config {
        FloatType a = 0.16;                     ///< ステップサイズ係数
        FloatType c = 0.16;                     ///< 摂動サイズ係数
        FloatType A = 100.0;                    ///< 安定性パラメータ
        FloatType alpha = 0.602;                ///< ステップサイズ減衰指数
        FloatType gamma = 0.101;                ///< 摂動サイズ減衰指数
        
        // 制約
        FloatType param_min = -10.0;            ///< パラメータ下限
        FloatType param_max = 10.0;             ///< パラメータ上限
        bool use_momentum = true;               ///< モメンタム使用フラグ
        FloatType momentum_beta = 0.9;          ///< モメンタム係数
        
        // その他
        uint32_t random_seed = 42;              ///< 乱数シード
        FloatType tolerance = 1e-6;             ///< 収束判定閾値
        int max_iterations = 1000;              ///< 最大イテレーション数
        FloatType learning_rate = 0.01;         ///< 学習率（廃止予定）
        FloatType gradient_smoothing = 0.9;     ///< 勾配平滑化係数
    };

private:
    Config config_;                             ///< 設定
    std::vector<torch::Tensor> parameters_;    ///< 最適化対象パラメータ
    std::vector<torch::Tensor> momentum_buffers_; ///< モメンタムバッファ
    torch::Tensor gradient_estimate_;          ///< 勾配推定値
    mutable std::mt19937 rng_;                 ///< 乱数生成器
    int iteration_count_;                       ///< イテレーション数

public:
    /// @brief コンストラクタ
    explicit SPSAOptimizer(const std::vector<torch::Tensor>& parameters,
                          const Config& config = Config{})
        : config_(config), parameters_(parameters), rng_(config.random_seed), iteration_count_(0) {
        
        if (parameters_.empty()) {
            throw std::invalid_argument("Parameters list cannot be empty");
        }
        
        // パラメータの検証
        for (const auto& param : parameters_) {
            if (!param.defined()) {
                throw std::invalid_argument("All parameters must be defined tensors");
            }
        }
        
        // モメンタムバッファの初期化
        if (config_.use_momentum) {
            momentum_buffers_.reserve(parameters_.size());
            for (const auto& param : parameters_) {
                momentum_buffers_.push_back(torch::zeros_like(param));
            }
        }
        
        reset();
    }

    /// @brief 最適化ステップ実行
    void step(std::function<FloatType()> objective_function, int iteration = -1) {
        if (iteration >= 0) {
            iteration_count_ = iteration;
        } else {
            iteration_count_++;
        }

        auto ak = compute_step_size(iteration_count_);
        auto ck = compute_perturbation_size(iteration_count_);
        auto gradients = estimate_gradient(objective_function, ck);
        update_gradient_estimate(gradients);
        update_parameters(ak);
    }

    /// @brief 完全な最適化実行
    FloatType optimize(std::function<FloatType()> objective_function) {
        FloatType current_objective = objective_function();
        FloatType prev_objective = current_objective;
        
        for (int iter = 0; iter < config_.max_iterations; ++iter) {
            prev_objective = current_objective;
            step(objective_function, iter);
            current_objective = objective_function();
            
            if (std::abs(current_objective - prev_objective) < config_.tolerance) {
                break;
            }
        }
        
        return current_objective;
    }

    /// @brief 勾配推定
    std::vector<torch::Tensor> estimate_gradient(
        std::function<FloatType()> objective_function, FloatType ck) {
        
        std::vector<torch::Tensor> gradients;
        gradients.reserve(parameters_.size());

        for (size_t i = 0; i < parameters_.size(); ++i) {
            auto& param = parameters_[i];
            auto perturbation = generate_bernoulli_perturbation(
                std::vector<int64_t>(param.sizes().begin(), param.sizes().end())
            );
            
            param += ck * perturbation;
            auto loss_plus = objective_function();
            
            param -= 2.0 * ck * perturbation;
            auto loss_minus = objective_function();
            
            param += ck * perturbation;
            
            auto gradient = (loss_plus - loss_minus) / (2.0 * ck) / perturbation;
            gradients.push_back(gradient);
        }

        return gradients;
    }

    /// @brief 状態リセット
    void reset() {
        iteration_count_ = 0;
        gradient_estimate_ = torch::Tensor();
        
        if (config_.use_momentum) {
            for (auto& buffer : momentum_buffers_) {
                buffer.zero_();
            }
        }
    }

    const Config& get_config() const { return config_; }
    int get_iteration_count() const { return iteration_count_; }
    size_t get_num_parameters() const { return parameters_.size(); }

private:
    FloatType compute_step_size(int iteration) const {
        return config_.a / std::pow(iteration + config_.A, config_.alpha);
    }

    FloatType compute_perturbation_size(int iteration) const {
        return config_.c / std::pow(iteration + 1, config_.gamma);
    }

    torch::Tensor generate_bernoulli_perturbation(const std::vector<int64_t>& sizes) {
        torch::ScalarType dtype = std::is_same_v<FloatType, float> ? torch::kFloat : torch::kDouble;
        auto perturbation = torch::empty(sizes, dtype);
        std::bernoulli_distribution bernoulli(0.5);
        
        auto flat = perturbation.flatten();
        auto accessor = flat.template accessor<FloatType, 1>();
        
        for (int64_t i = 0; i < flat.numel(); ++i) {
            accessor[i] = bernoulli(rng_) ? static_cast<FloatType>(1.0) : static_cast<FloatType>(-1.0);
        }
        
        return perturbation.reshape(sizes);
    }

    void update_gradient_estimate(const std::vector<torch::Tensor>& new_gradients) {
        if (new_gradients.empty()) return;
        
        std::vector<torch::Tensor> flat_grads;
        for (const auto& grad : new_gradients) {
            flat_grads.push_back(grad.flatten());
        }
        auto flattened_gradients = torch::cat(flat_grads);
        
        if (!gradient_estimate_.defined()) {
            gradient_estimate_ = flattened_gradients.clone();
        } else {
            gradient_estimate_ = config_.gradient_smoothing * gradient_estimate_ + 
                               (1.0 - config_.gradient_smoothing) * flattened_gradients;
        }
    }

    void update_parameters(FloatType step_size) {
        if (!gradient_estimate_.defined()) return;
        
        auto gradients = unflatten_gradient(gradient_estimate_);
        
        for (size_t i = 0; i < parameters_.size(); ++i) {
            auto& param = parameters_[i];
            const auto& gradient = gradients[i];

            torch::Tensor update;
            
            if (config_.use_momentum && i < momentum_buffers_.size()) {
                auto& momentum = momentum_buffers_[i];
                momentum = config_.momentum_beta * momentum + step_size * gradient;
                update = momentum;
            } else {
                update = step_size * gradient;
            }

            param -= update;
            param.clamp_(config_.param_min, config_.param_max);
        }
    }

    std::vector<torch::Tensor> unflatten_gradient(const torch::Tensor& flat_gradient) {
        std::vector<torch::Tensor> gradients;
        gradients.reserve(parameters_.size());
        
        int64_t start_idx = 0;
        for (const auto& param : parameters_) {
            auto param_size = param.numel();
            auto param_grad = flat_gradient.slice(0, start_idx, start_idx + param_size);
            gradients.push_back(param_grad.reshape(param.sizes()));
            start_idx += param_size;
        }
        
        return gradients;
    }
};

// 型エイリアス
using SPSAOptimizerF = SPSAOptimizer<float>;
using SPSAOptimizerD = SPSAOptimizer<double>;

/// @brief 便利な構築関数
template<typename FloatType = double>
inline std::unique_ptr<SPSAOptimizer<FloatType>> make_spsa_optimizer(
    const std::vector<torch::Tensor>& parameters,
    const typename SPSAOptimizer<FloatType>::Config& config = {}) {
    return std::make_unique<SPSAOptimizer<FloatType>>(parameters, config);
}

} // namespace crlgru

#endif // CRLGRU_OPTIMIZERS_SPSA_OPTIMIZER_HPP
