#ifndef CRLGRU_UTILS_MATH_UTILS_HPP
#define CRLGRU_UTILS_MATH_UTILS_HPP

/// @file math_utils.hpp
/// @brief 数学関数のヘッダーオンリー実装

#include <crlgru/common.h>
#include <cmath>
#include <algorithm>
#include <numeric>

namespace crlgru {
namespace utils {

/// @brief 効率的な相互情報量計算（高速版）
/// @param state1 第1の状態ベクトル
/// @param state2 第2の状態ベクトル
/// @param num_bins ビン数（デフォルト: 64）
/// @return 相互情報量
template<typename Scalar = double>
inline Scalar compute_mutual_information(const torch::Tensor& state1, 
                                        const torch::Tensor& state2,
                                        int /* num_bins */ = 64) {
    // 入力の正規化
    auto normalized_state1 = (state1 - state1.mean()) / (state1.std() + 1e-8);
    auto normalized_state2 = (state2 - state2.mean()) / (state2.std() + 1e-8);
    
    // 相関係数による近似（高速）
    auto correlation = torch::corrcoef(torch::stack({normalized_state1.flatten(), 
                                                   normalized_state2.flatten()}));
    auto corr_value = torch::abs(correlation[0][1]).item<Scalar>();
    
    // 相関から相互情報量への近似変換
    // MI ≈ -0.5 * log(1 - ρ²) where ρ is correlation coefficient
    auto mi = -0.5 * std::log(std::max(1.0 - corr_value * corr_value, 1e-8));
    
    return static_cast<Scalar>(mi);
}

/// @brief エントロピー計算
/// @param distribution 確率分布
/// @param epsilon 数値安定化定数
/// @return エントロピー値
template<typename Scalar = double>
inline Scalar compute_entropy(const torch::Tensor& distribution,
                            Scalar epsilon = 1e-8) {
    auto normalized = distribution / (distribution.sum() + epsilon);
    auto log_p = torch::log(normalized + epsilon);
    return -(normalized * log_p).sum().template item<Scalar>();
}

/// @brief KLダイバージェンス計算
/// @param p 確率分布P
/// @param q 確率分布Q
/// @param epsilon 数値安定化定数
/// @return KL(P||Q)
template<typename Scalar = double>
inline Scalar compute_kl_divergence(const torch::Tensor& p,
                                   const torch::Tensor& q,
                                   Scalar epsilon = 1e-8) {
    auto p_norm = p / (p.sum() + epsilon);
    auto q_norm = q / (q.sum() + epsilon);
    
    auto log_ratio = torch::log((p_norm + epsilon) / (q_norm + epsilon));
    return (p_norm * log_ratio).sum().template item<Scalar>();
}

/// @brief 信頼度メトリック計算
/// @param performance_history 性能履歴
/// @param spatial_distance 空間距離
/// @param max_distance 最大距離
/// @param temporal_decay 時間減衰係数
/// @return 信頼度スコア
template<typename Scalar = double>
inline Scalar compute_trust_metric(const std::vector<Scalar>& performance_history,
                                  Scalar spatial_distance,
                                  Scalar max_distance,
                                  Scalar temporal_decay = 0.95) {
    if (performance_history.empty()) {
        return 0.0;
    }
    
    // 時間加重平均の計算
    Scalar weighted_sum = 0.0;
    Scalar weight_sum = 0.0;
    Scalar current_weight = 1.0;
    
    // 最新から過去へ遡る
    for (auto it = performance_history.rbegin(); it != performance_history.rend(); ++it) {
        weighted_sum += (*it) * current_weight;
        weight_sum += current_weight;
        current_weight *= temporal_decay;
    }
    
    Scalar performance_score = weight_sum > 0 ? weighted_sum / weight_sum : 0.0;
    
    // 空間距離による減衰
    Scalar distance_factor = 1.0 - std::min(spatial_distance / max_distance, 1.0);
    
    return performance_score * distance_factor;
}

/// @brief 信頼度行列構築
/// @param performance_histories 各エージェントの性能履歴
/// @param distance_matrix 距離行列
/// @param max_distance 最大距離
/// @return 信頼度行列
template<typename Scalar = double>
inline torch::Tensor build_trust_matrix(const std::vector<std::vector<Scalar>>& performance_histories,
                                       const torch::Tensor& distance_matrix,
                                       Scalar max_distance) {
    auto num_agents = static_cast<int64_t>(performance_histories.size());
    auto trust_matrix = torch::zeros({num_agents, num_agents});
    
    for (int64_t i = 0; i < num_agents; ++i) {
        for (int64_t j = 0; j < num_agents; ++j) {
            if (i != j) {
                auto distance = distance_matrix[i][j].template item<Scalar>();
                auto trust = compute_trust_metric(performance_histories[j], 
                                                distance, max_distance);
                trust_matrix[i][j] = trust;
            } else {
                trust_matrix[i][j] = 1.0; // 自己信頼度
            }
        }
    }
    
    return trust_matrix;
}

/// @brief ガウシアンカーネル適用
/// @param input 入力テンソル
/// @param sigma 標準偏差
/// @param kernel_size カーネルサイズ（0の場合は自動計算）
/// @return フィルタリング済みテンソル
inline torch::Tensor apply_gaussian_kernel(const torch::Tensor& input,
                                          double sigma,
                                          int kernel_size = 0) {
    // カーネルサイズの自動計算
    if (kernel_size == 0) {
        kernel_size = static_cast<int>(std::ceil(6 * sigma)) | 1; // 奇数にする
    }
    
    // 1Dガウシアンカーネル生成
    auto half_size = kernel_size / 2;
    std::vector<double> kernel_values;
    kernel_values.reserve(kernel_size);
    
    double sum = 0.0;
    for (int i = -half_size; i <= half_size; ++i) {
        double value = std::exp(-(i * i) / (2 * sigma * sigma));
        kernel_values.push_back(value);
        sum += value;
    }
    
    // 正規化
    for (auto& val : kernel_values) {
        val /= sum;
    }
    
    auto kernel = torch::tensor(kernel_values).to(input.device());
    
    // 入力の次元に応じて処理
    if (input.dim() == 1) {
        // 1D畳み込み
        auto padded_input = torch::nn::functional::pad(
            input.unsqueeze(0).unsqueeze(0),
            torch::nn::functional::PadFuncOptions({half_size, half_size}).mode(torch::kReflect)
        );
        auto result = torch::conv1d(padded_input, kernel.view({1, 1, -1}));
        return result.squeeze(0).squeeze(0);
    } else if (input.dim() == 2) {
        // 2D分離可能畳み込み（高速）
        auto temp = torch::nn::functional::conv1d(
            input.unsqueeze(1),
            kernel.view({1, 1, -1}),
            torch::nn::functional::Conv1dFuncOptions().padding(half_size)
        ).squeeze(1);
        
        auto result = torch::nn::functional::conv1d(
            temp.transpose(0, 1).unsqueeze(1),
            kernel.view({1, 1, -1}),
            torch::nn::functional::Conv1dFuncOptions().padding(half_size)
        ).squeeze(1).transpose(0, 1);
        
        return result;
    }
    
    return input; // サポートされていない次元の場合はそのまま返す
}

/// @brief 移動平均フィルタ
/// @param signal 信号
/// @param window_size ウィンドウサイズ
/// @return フィルタリング済み信号
inline torch::Tensor moving_average(const torch::Tensor& signal,
                                   int window_size) {
    if (window_size <= 1) {
        return signal;
    }
    
    auto kernel = torch::ones({window_size}) / window_size;
    auto padding = window_size / 2;
    
    auto padded_signal = torch::nn::functional::pad(
        signal.unsqueeze(0).unsqueeze(0),
        torch::nn::functional::PadFuncOptions({padding, padding}).mode(torch::kReflect)
    );
    
    auto result = torch::conv1d(padded_signal, kernel.view({1, 1, -1}));
    return result.squeeze(0).squeeze(0);
}

/// @brief 効率的なテンソル統計計算
template<typename Scalar = double>
inline Scalar compute_tensor_mean(const torch::Tensor& tensor) {
    return tensor.mean().template item<Scalar>();
}

template<typename Scalar = double>
inline Scalar compute_tensor_std(const torch::Tensor& tensor) {
    return tensor.std().template item<Scalar>();
}

template<typename Scalar = double>
inline Scalar compute_tensor_var(const torch::Tensor& tensor) {
    return tensor.var().template item<Scalar>();
}

/// @brief ゼロ除算安全な正規化
inline torch::Tensor safe_normalize(const torch::Tensor& tensor, 
                                   double eps = 1e-8) {
    auto norm = tensor.norm();
    return tensor / (norm + eps);
}

/// @brief L2正規化
inline torch::Tensor l2_normalize(const torch::Tensor& tensor,
                                 int dim = -1,
                                 double eps = 1e-8) {
    auto norm = tensor.norm(2, dim, true);
    return tensor / (norm + eps);
}

/// @brief ソフトマックス関数（数値安定版）
inline torch::Tensor stable_softmax(const torch::Tensor& logits,
                                   int dim = -1) {
    auto max_vals = std::get<0>(torch::max(logits, dim, true));
    auto shifted = logits - max_vals;
    auto exp_vals = torch::exp(shifted);
    auto sum_exp = torch::sum(exp_vals, dim, true);
    return exp_vals / sum_exp;
}

/// @brief ログソフトマックス関数（数値安定版）
inline torch::Tensor stable_log_softmax(const torch::Tensor& logits,
                                       int dim = -1) {
    auto max_vals = std::get<0>(torch::max(logits, dim, true));
    auto shifted = logits - max_vals;
    auto log_sum_exp = torch::log(torch::sum(torch::exp(shifted), dim, true));
    return shifted - log_sum_exp;
}

/// @brief 重み付き平均計算
inline torch::Tensor weighted_average(const torch::Tensor& values,
                                     const torch::Tensor& weights,
                                     int dim = 0,
                                     double eps = 1e-8) {
    auto normalized_weights = weights / (weights.sum(dim, true) + eps);
    return (values * normalized_weights).sum(dim);
}

/// @brief クランプ付きシグモイド
inline torch::Tensor clamped_sigmoid(const torch::Tensor& x,
                                    double min_val = 1e-6,
                                    double max_val = 1.0 - 1e-6) {
    auto sigmoid_x = torch::sigmoid(x);
    return torch::clamp(sigmoid_x, min_val, max_val);
}

/// @brief スムーズなステップ関数
inline torch::Tensor smooth_step(const torch::Tensor& x,
                                double edge0 = 0.0,
                                double edge1 = 1.0) {
    auto t = torch::clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

} // namespace utils
} // namespace crlgru

#endif // CRLGRU_UTILS_MATH_UTILS_HPP
