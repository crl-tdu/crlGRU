#ifndef CRLGRU_UTILS_CONFIG_TYPES_HPP
#define CRLGRU_UTILS_CONFIG_TYPES_HPP

/// @file config_types.hpp
/// @brief 設定構造体のヘッダーオンリー実装

#include <crlgru/common.hpp>
#include <string>
#include <vector>

namespace crlgru {

/// @brief FEPGRUCell設定構造体
struct FEPGRUCellConfig {
    int input_size = 64;                    ///< 入力サイズ
    int hidden_size = 128;                  ///< 隠れ状態サイズ
    bool bias = true;                       ///< バイアス使用フラグ
    double dropout = 0.0;                   ///< ドロップアウト率
    
    // FEP特有のパラメータ
    double free_energy_weight = 1.0;        ///< 自由エネルギー重み
    double prediction_horizon = 10.0;       ///< 予測ホライズン
    double variational_beta = 1.0;          ///< 変分ベータ
    
    // 模倣学習パラメータ
    double imitation_rate = 0.1;            ///< 模倣学習率
    double trust_radius = 5.0;              ///< 信頼半径
    bool enable_hierarchical_imitation = true; ///< 階層的模倣学習有効フラグ
    
    // SOM統合設定
    bool enable_som_extraction = true;      ///< SOM特徴抽出有効フラグ
    int som_grid_size = 16;                 ///< SOMグリッドサイズ
    double som_learning_rate = 0.01;        ///< SOM学習率
};

/// @brief FEPGRUNetwork設定構造体
struct FEPGRUNetworkConfig {
    std::vector<int> layer_sizes = {64, 128, 64}; ///< 各層のサイズ
    FEPGRUCellConfig cell_config;           ///< セル設定
    
    // ネットワーク特有の設定
    bool bidirectional = false;             ///< 双方向フラグ
    double dropout_rate = 0.0;              ///< ドロップアウト率
    bool enable_layer_norm = false;         ///< レイヤー正規化フラグ
    
    // 階層的模倣学習
    bool enable_hierarchical_imitation = true; ///< 階層的模倣学習有効フラグ
    std::vector<double> imitation_weights = {0.3, 0.5, 0.2}; ///< 模倣重み
};

/// @brief PolarSpatialAttention設定構造体
struct PolarSpatialAttentionConfig {
    int input_channels = 64;                ///< 入力チャンネル数
    int num_distance_rings = 8;             ///< 距離リング数
    int num_angle_sectors = 16;             ///< 角度セクター数
    double max_range = 10.0;                ///< 最大範囲
    
    // 注意メカニズムのタイプ
    enum AttentionType {
        SOFTMAX,
        GAUSSIAN,
        LEARNED
    } attention_type = SOFTMAX;
    
    // 学習可能なパラメータ
    bool learnable_range = true;            ///< 学習可能範囲フラグ
    bool learnable_sectors = false;         ///< 学習可能セクターフラグ
};

/// @brief MetaEvaluator設定構造体
struct MetaEvaluatorConfig {
    // 評価指標
    std::vector<std::string> metrics = {
        "prediction_accuracy",
        "free_energy",
        "complexity",
        "coordination_score"
    };
    
    // 重み設定
    std::vector<double> initial_weights;    ///< 初期重み
    bool adaptive_weights = true;           ///< 適応的重みフラグ
    double weight_adaptation_rate = 0.01;   ///< 重み適応率
    
    // 正規化
    bool normalize_metrics = true;          ///< メトリック正規化フラグ
};

/// @brief SPSA最適化器設定構造体
struct SPSAOptimizerConfig {
    double a = 0.16;                        ///< ステップサイズ係数
    double c = 0.16;                        ///< 摂動サイズ係数
    double A = 100.0;                       ///< 安定性パラメータ
    double alpha = 0.602;                   ///< ステップサイズ減衰指数
    double gamma = 0.101;                   ///< 摂動サイズ減衰指数
    
    // 制約
    double param_min = -10.0;               ///< パラメータ下限
    double param_max = 10.0;                ///< パラメータ上限
    bool use_momentum = true;               ///< モメンタム使用フラグ
    double momentum_beta = 0.9;             ///< モメンタム係数
    
    // その他
    uint32_t random_seed = 42;              ///< 乱数シード
    double tolerance = 1e-6;                ///< 収束判定閾値
    int max_iterations = 1000;              ///< 最大イテレーション数
    double learning_rate = 0.01;            ///< 学習率
    double gradient_smoothing = 0.9;        ///< 勾配平滑化係数
};

/// @brief 模倣学習設定構造体
struct ImitationConfig {
    // 階層レベル
    enum Level {
        PARAMETER_LEVEL = 0,
        DYNAMICS_LEVEL = 1,
        INTENTION_LEVEL = 2
    };
    
    // 基本設定
    Level imitation_level = PARAMETER_LEVEL; ///< 模倣レベル
    double learning_rate = 0.1;             ///< 学習率
    double trust_threshold = 0.5;           ///< 信頼閾値
    
    // 階層別の重み
    std::vector<double> level_weights = {0.3, 0.5, 0.2}; ///< レベル別重み
    
    // 距離ベースの減衰
    bool use_distance_decay = true;         ///< 距離減衰使用フラグ
    double decay_factor = 0.1;              ///< 減衰係数
    
    // 時間的減衰
    bool use_temporal_decay = true;         ///< 時間減衰使用フラグ
    double temporal_decay_rate = 0.95;      ///< 時間減衰率
};

/// @brief SOM設定構造体
struct SOMConfig {
    // グリッドサイズ
    int grid_width = 8;                     ///< グリッド幅
    int grid_height = 8;                    ///< グリッド高さ
    
    // 学習パラメータ
    double initial_learning_rate = 0.5;     ///< 初期学習率
    double initial_radius = 3.0;            ///< 初期半径
    double learning_decay = 0.99;           ///< 学習率減衰
    double radius_decay = 0.99;             ///< 半径減衰
    
    // 距離メトリック
    enum DistanceMetric {
        EUCLIDEAN,
        MANHATTAN,
        COSINE
    } distance_metric = EUCLIDEAN;
    
    // 近傍関数
    enum NeighborhoodFunction {
        GAUSSIAN,
        BUBBLE,
        MEXICAN_HAT
    } neighborhood_function = GAUSSIAN;
};

} // namespace crlgru

#endif // CRLGRU_UTILS_CONFIG_TYPES_HPP
