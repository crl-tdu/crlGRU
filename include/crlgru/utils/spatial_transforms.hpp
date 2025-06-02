#ifndef CRLGRU_UTILS_SPATIAL_TRANSFORMS_HPP
#define CRLGRU_UTILS_SPATIAL_TRANSFORMS_HPP

/// @file spatial_transforms.hpp
/// @brief 空間変換関数のヘッダーオンリー実装

#include <crlgru/common.h>
#include <cmath>
#include <algorithm>

namespace crlgru {
namespace utils {

/// @brief カルテシアン座標から極座標マップへの変換（最適化版）
/// @param positions エージェント位置 [batch_size, num_agents, 2]
/// @param self_position 自己位置 [batch_size, 2]
/// @param num_rings 距離リング数
/// @param num_sectors 角度セクター数
/// @param max_range 最大範囲
/// @return 極座標マップ [batch_size, num_rings, num_sectors]
inline torch::Tensor cartesian_to_polar_map(const torch::Tensor& positions,
                                           const torch::Tensor& self_position,
                                           int num_rings, 
                                           int num_sectors,
                                           double max_range) {
    auto batch_size = positions.size(0);
    auto num_agents = positions.size(1);
    
    // 効率的な実装：テンソル演算を最大限活用
    auto polar_map = torch::zeros({batch_size, num_rings, num_sectors}, 
                                  positions.options());
    
    // ベクトル化された相対位置計算
    auto rel_positions = positions - self_position.unsqueeze(1); // broadcast
    auto x = rel_positions.select(2, 0); // x coordinates
    auto y = rel_positions.select(2, 1); // y coordinates
    
    // 距離と角度をベクトル化して計算
    auto distances = torch::sqrt(x * x + y * y);
    auto angles = torch::atan2(y, x);
    
    // 角度を [0, 2π] に正規化
    angles = torch::where(angles < 0, angles + 2 * M_PI, angles);
    
    // インデックス計算（ベクトル化）
    auto ring_indices = torch::clamp(
        (distances / max_range * num_rings).to(torch::kLong),
        0, num_rings - 1
    );
    auto sector_indices = torch::clamp(
        (angles / (2 * M_PI) * num_sectors).to(torch::kLong),
        0, num_sectors - 1
    );
    
    // 範囲内のマスク
    auto valid_mask = distances <= max_range;
    
    // 効率的なビン集計
    for (int64_t b = 0; b < batch_size; ++b) {
        for (int64_t a = 0; a < num_agents; ++a) {
            if (valid_mask[b][a].item<bool>()) {
                auto ring_idx = ring_indices[b][a].item<int64_t>();
                auto sector_idx = sector_indices[b][a].item<int64_t>();
                polar_map[b][ring_idx][sector_idx] += 1.0;
            }
        }
    }
    
    return polar_map;
}

/// @brief 極座標マップからカルテシアン座標への逆変換
/// @param polar_map 極座標マップ [batch_size, channels, rings, sectors]
/// @param self_position 自己位置 [batch_size, 2]
/// @param max_range 最大範囲
/// @return 推定位置 [batch_size, num_detected, 2]
inline torch::Tensor polar_to_cartesian(const torch::Tensor& polar_map,
                                       const torch::Tensor& self_position,
                                       double max_range) {
    auto batch_size = polar_map.size(0);
    auto num_rings = polar_map.size(-2);
    auto num_sectors = polar_map.size(-1);
    
    std::vector<torch::Tensor> positions_list;
    
    for (int64_t b = 0; b < batch_size; ++b) {
        std::vector<torch::Tensor> batch_positions;
        
        for (int r = 0; r < num_rings; ++r) {
            for (int s = 0; s < num_sectors; ++s) {
                auto value = polar_map[b].select(1, r).select(1, s);
                if (torch::any(value > 0.5).item<bool>()) {
                    // 極座標からカルテシアン座標に変換
                    double distance = (r + 0.5) * max_range / num_rings;
                    double angle = (s + 0.5) * 2 * M_PI / num_sectors;
                    
                    auto x = distance * std::cos(angle) + self_position[b][0];
                    auto y = distance * std::sin(angle) + self_position[b][1];
                    
                    batch_positions.push_back(
                        torch::stack({x, y}).unsqueeze(0)
                    );
                }
            }
        }
        
        if (!batch_positions.empty()) {
            positions_list.push_back(torch::cat(batch_positions, 0));
        } else {
            // 空の場合はダミー位置
            positions_list.push_back(torch::zeros({1, 2}, polar_map.options()));
        }
    }
    
    // パディングして統一サイズに
    int max_detections = 0;
    for (const auto& pos : positions_list) {
        max_detections = std::max(max_detections, static_cast<int>(pos.size(0)));
    }
    
    std::vector<torch::Tensor> padded_positions;
    for (const auto& pos : positions_list) {
        auto padding_size = max_detections - pos.size(0);
        if (padding_size > 0) {
            auto padding = torch::zeros({padding_size, 2}, pos.options());
            padded_positions.push_back(torch::cat({pos, padding}, 0));
        } else {
            padded_positions.push_back(pos);
        }
    }
    
    return torch::stack(padded_positions, 0);
}

/// @brief 回転変換行列の生成
/// @param angle 回転角度（ラジアン）
/// @return 2x2回転行列
inline torch::Tensor rotation_matrix_2d(double angle) {
    auto cos_a = std::cos(angle);
    auto sin_a = std::sin(angle);
    
    return torch::tensor({{cos_a, -sin_a},
                         {sin_a,  cos_a}});
}

/// @brief 点群の重心計算
/// @param positions 位置ベクトル [num_points, 2]
/// @return 重心位置 [2]
inline torch::Tensor compute_centroid(const torch::Tensor& positions) {
    return positions.mean(0);
}

/// @brief 点群の分散計算
/// @param positions 位置ベクトル [num_points, 2]
/// @param centroid 重心位置 [2]（省略時は自動計算）
/// @return 分散値（スカラー）
inline torch::Tensor compute_spread(const torch::Tensor& positions,
                                   const torch::Tensor& centroid = {}) {
    auto center = centroid.defined() ? centroid : compute_centroid(positions);
    auto deviations = positions - center.unsqueeze(0);
    return (deviations * deviations).sum(1).mean();
}

/// @brief 最近傍探索（総当たり版）
/// @param query_points クエリ点 [num_queries, 2]
/// @param reference_points 参照点 [num_refs, 2]
/// @return 最近傍インデックス [num_queries]
inline torch::Tensor nearest_neighbors(const torch::Tensor& query_points,
                                      const torch::Tensor& reference_points) {
    auto distances = torch::cdist(query_points, reference_points);
    return std::get<1>(torch::min(distances, 1));
}

/// @brief 範囲内の点をフィルタリング
/// @param positions 位置ベクトル [num_points, 2]
/// @param center 中心位置 [2]
/// @param max_distance 最大距離
/// @return フィルタリング済み位置 [num_filtered, 2]
inline torch::Tensor filter_by_distance(const torch::Tensor& positions,
                                       const torch::Tensor& center,
                                       double max_distance) {
    auto distances = torch::norm(positions - center.unsqueeze(0), 2, 1);
    auto mask = distances <= max_distance;
    return positions.index({mask});
}

/// @brief 局所座標系への変換
/// @param global_positions グローバル位置 [num_points, 2]
/// @param origin 原点位置 [2]
/// @param orientation 方向角（ラジアン）
/// @return 局所座標 [num_points, 2]
inline torch::Tensor global_to_local(const torch::Tensor& global_positions,
                                    const torch::Tensor& origin,
                                    double orientation = 0.0) {
    auto relative_positions = global_positions - origin.unsqueeze(0);
    
    if (std::abs(orientation) > 1e-8) {
        auto rot_matrix = rotation_matrix_2d(-orientation);
        return torch::matmul(relative_positions, rot_matrix.transpose(0, 1));
    }
    
    return relative_positions;
}

/// @brief 局所座標系からグローバル座標系への変換
/// @param local_positions 局所位置 [num_points, 2]
/// @param origin 原点位置 [2]
/// @param orientation 方向角（ラジアン）
/// @return グローバル座標 [num_points, 2]
inline torch::Tensor local_to_global(const torch::Tensor& local_positions,
                                    const torch::Tensor& origin,
                                    double orientation = 0.0) {
    torch::Tensor rotated_positions = local_positions;
    
    if (std::abs(orientation) > 1e-8) {
        auto rot_matrix = rotation_matrix_2d(orientation);
        rotated_positions = torch::matmul(local_positions, rot_matrix.transpose(0, 1));
    }
    
    return rotated_positions + origin.unsqueeze(0);
}

} // namespace utils
} // namespace crlgru

#endif // CRLGRU_UTILS_SPATIAL_TRANSFORMS_HPP
