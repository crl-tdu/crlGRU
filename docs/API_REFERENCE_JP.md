# crlGRU API リファレンス

## 目次

1. [基本概念](#基本概念)
2. [主要クラス](#主要クラス)
3. [ユーティリティ関数](#ユーティリティ関数)
4. [設定構造体](#設定構造体)
5. [使用例](#使用例)

## 基本概念

### 自由エネルギー原理（FEP）

crlGRUは、Karl Fristonの自由エネルギー原理に基づいて設計されています。システムは以下を最小化します：

```
F = D_KL[q(x)||p(x)] - log p(y|x)
```

- `F`: 自由エネルギー
- `q(x)`: 認識分布（内部モデル）
- `p(x)`: 生成モデル
- `p(y|x)`: 尤度

### 階層的模倣学習

3つのレベルで実装されています：

1. **パラメータレベル**: 重みの直接共有
2. **ダイナミクスレベル**: 隠れ状態の同期
3. **意図レベル**: 目標と戦略の共有

## 主要クラス

### FEPGRUCell

自由エネルギー原理に基づくGRUセルの実装です。

```cpp
namespace crlgru {

class FEPGRUCell {
public:
    struct Config {
        int input_size = 64;
        int hidden_size = 128;
        
        // FEP specific parameters
        double beta = 1.0;              // 自由エネルギーの温度パラメータ
        double learning_rate = 0.01;
        
        // SOM configuration
        bool enable_som_extraction = true;
        int som_grid_size = 8;
        double som_learning_rate = 0.1;
        double som_radius = 3.0;
        
        // Meta evaluation
        bool enable_meta_evaluation = true;
        std::vector<std::string> evaluation_metrics = {
            "prediction_error", 
            "free_energy", 
            "complexity"
        };
    };
    
    // コンストラクタ
    explicit FEPGRUCell(const Config& config);
    
    // フォワードパス
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(
        const torch::Tensor& input,
        const torch::Tensor& hidden,
        const torch::Tensor& context = {}
    );
    
    // SOM特徴抽出
    torch::Tensor extract_som_features(const torch::Tensor& hidden_states);
    
    // パラメータ更新（ピア学習）
    void update_parameters_from_peer(
        std::shared_ptr<FEPGRUCell> peer,
        double trust_weight = 0.1
    );
    
    // 自由エネルギー計算
    torch::Tensor compute_free_energy(
        const torch::Tensor& prediction,
        const torch::Tensor& observation
    );
    
    // パラメータ取得・設定
    std::unordered_map<std::string, torch::Tensor> get_parameters() const;
    void set_parameters(const std::unordered_map<std::string, torch::Tensor>& params);
};

} // namespace crlgru
```

### FEPGRUNetwork

多層FEP-GRUネットワークの実装です。

```cpp
class FEPGRUNetwork {
public:
    struct NetworkConfig {
        std::vector<int> layer_sizes = {64, 128, 64};
        FEPGRUCell::Config cell_config;
        
        // ネットワーク特有の設定
        bool bidirectional = false;
        double dropout_rate = 0.0;
        
        // 階層的模倣学習
        bool enable_hierarchical_imitation = true;
        std::vector<double> imitation_weights = {0.3, 0.5, 0.2};
    };
    
    // コンストラクタ
    explicit FEPGRUNetwork(const NetworkConfig& config);
    
    // シーケンス処理
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(
        const torch::Tensor& input_sequence,
        const torch::Tensor& initial_hidden = {}
    );
    
    // エージェント間パラメータ共有
    void share_parameters_with_agents(
        const std::vector<std::shared_ptr<FEPGRUNetwork>>& agents,
        const std::vector<double>& trust_scores
    );
    
    // 階層的模倣更新
    void hierarchical_imitation_update(
        std::shared_ptr<FEPGRUNetwork> target_agent,
        int hierarchy_level,
        double learning_rate
    );
    
    // 内部状態の取得
    std::vector<torch::Tensor> get_hidden_states() const;
};
```

### PolarSpatialAttention

極座標ベースの空間注意メカニズムです。

```cpp
class PolarSpatialAttention {
public:
    struct AttentionConfig {
        int input_channels = 64;
        int num_distance_rings = 8;
        int num_angle_sectors = 16;
        double max_range = 10.0;
        
        // 注意メカニズムのタイプ
        enum AttentionType {
            SOFTMAX,
            GAUSSIAN,
            LEARNED
        } attention_type = SOFTMAX;
        
        // 学習可能なパラメータ
        bool learnable_range = true;
        bool learnable_sectors = false;
    };
    
    explicit PolarSpatialAttention(const AttentionConfig& config);
    
    // 注意適用
    torch::Tensor forward(const torch::Tensor& polar_map);
    
    // 注意重みの取得
    torch::Tensor get_attention_weights() const;
    
    // 距離ベースのマスキング
    torch::Tensor apply_distance_mask(
        const torch::Tensor& input,
        double min_distance,
        double max_distance
    );
};
```

### MetaEvaluator

多目的評価と最適化のためのクラスです。

```cpp
class MetaEvaluator {
public:
    struct EvaluationConfig {
        // 評価指標
        std::vector<std::string> metrics = {
            "prediction_accuracy",
            "free_energy",
            "complexity",
            "coordination_score"
        };
        
        // 重み設定
        std::vector<double> initial_weights;
        bool adaptive_weights = true;
        double weight_adaptation_rate = 0.01;
        
        // 正規化
        bool normalize_metrics = true;
    };
    
    explicit MetaEvaluator(const EvaluationConfig& config);
    
    // 状態評価
    std::unordered_map<std::string, double> evaluate(
        const torch::Tensor& state,
        const torch::Tensor& target = {},
        const std::unordered_map<std::string, torch::Tensor>& context = {}
    );
    
    // 重み適応
    void adapt_weights(
        const std::vector<std::unordered_map<std::string, double>>& history,
        const std::string& primary_objective
    );
    
    // 複合スコア計算
    double compute_composite_score(
        const std::unordered_map<std::string, double>& metrics
    );
    
    // パレート最適解の探索
    std::vector<int> find_pareto_optimal(
        const std::vector<std::unordered_map<std::string, double>>& evaluations
    );
};
```

### SPSAOptimizer

SPSA（Simultaneous Perturbation Stochastic Approximation）最適化器です。

```cpp
class SPSAOptimizer {
public:
    struct OptimizerConfig {
        double a = 0.16;        // ステップサイズ係数
        double c = 0.16;        // 摂動サイズ係数
        double A = 100.0;       // 安定性パラメータ
        double alpha = 0.602;   // ステップサイズ減衰指数
        double gamma = 0.101;   // 摂動サイズ減衰指数
        
        // 制約
        double param_min = -10.0;
        double param_max = 10.0;
        bool use_momentum = true;
        double momentum_beta = 0.9;
    };
    
    explicit SPSAOptimizer(
        const std::vector<torch::Tensor>& parameters,
        const OptimizerConfig& config
    );
    
    // 最適化ステップ
    void step(
        std::function<double(void)> loss_function,
        int iteration
    );
    
    // パラメータ摂動
    std::vector<torch::Tensor> perturb_parameters();
    
    // 勾配推定
    std::vector<torch::Tensor> estimate_gradient(
        std::function<double(void)> loss_function,
        const std::vector<torch::Tensor>& perturbations
    );
};
```

## ユーティリティ関数

### 座標変換

```cpp
namespace crlgru::utils {

// デカルト座標から極座標マップへの変換
torch::Tensor cartesian_to_polar_map(
    const torch::Tensor& positions,      // [batch, num_agents, 2]
    const torch::Tensor& self_position,  // [batch, 2]
    int num_rings,
    int num_sectors,
    double max_range
);

// 極座標からデカルト座標への逆変換
torch::Tensor polar_to_cartesian(
    const torch::Tensor& polar_map,      // [batch, channels, rings, sectors]
    const torch::Tensor& self_position,  // [batch, 2]
    double max_range
);

} // namespace crlgru::utils
```

### 情報理論的指標

```cpp
// 相互情報量の計算
double compute_mutual_information(
    const torch::Tensor& x,
    const torch::Tensor& y,
    int num_bins = 256
);

// エントロピー計算
double compute_entropy(
    const torch::Tensor& distribution,
    double epsilon = 1e-8
);

// KLダイバージェンス
double compute_kl_divergence(
    const torch::Tensor& p,
    const torch::Tensor& q,
    double epsilon = 1e-8
);
```

### 信号処理

```cpp
// ガウシアンカーネルの適用
torch::Tensor apply_gaussian_kernel(
    const torch::Tensor& input,
    double sigma,
    int kernel_size = 0  // 0の場合は自動計算
);

// 移動平均フィルタ
torch::Tensor moving_average(
    const torch::Tensor& signal,
    int window_size
);

// フーリエ変換ベースのフィルタリング
torch::Tensor frequency_filter(
    const torch::Tensor& signal,
    double low_cutoff,
    double high_cutoff,
    double sampling_rate = 1.0
);
```

### 信頼度メトリック

```cpp
// エージェント間の信頼度計算
double compute_trust_metric(
    const std::vector<double>& performance_history,
    double spatial_distance,
    double max_distance,
    double temporal_decay = 0.95
);

// 信頼度行列の構築
torch::Tensor build_trust_matrix(
    const std::vector<std::vector<double>>& performance_histories,
    const torch::Tensor& distance_matrix,
    double max_distance
);
```

### パラメータ管理

```cpp
// パラメータの保存
void save_parameters(
    const std::string& filename,
    const std::unordered_map<std::string, torch::Tensor>& parameters,
    bool compressed = true
);

// パラメータの読み込み
std::unordered_map<std::string, torch::Tensor> load_parameters(
    const std::string& filename
);

// チェックポイント管理
class CheckpointManager {
public:
    CheckpointManager(const std::string& checkpoint_dir, int max_checkpoints = 5);
    
    void save_checkpoint(
        const std::string& name,
        const std::unordered_map<std::string, torch::Tensor>& state,
        const std::unordered_map<std::string, double>& metrics = {}
    );
    
    std::unordered_map<std::string, torch::Tensor> load_checkpoint(
        const std::string& name
    );
    
    std::string get_best_checkpoint(const std::string& metric_name);
};
```

## 設定構造体

### ImitationConfig

模倣学習の設定です。

```cpp
struct ImitationConfig {
    // 階層レベル
    enum Level {
        PARAMETER_LEVEL = 0,
        DYNAMICS_LEVEL = 1,
        INTENTION_LEVEL = 2
    };
    
    // 基本設定
    Level imitation_level = PARAMETER_LEVEL;
    double learning_rate = 0.1;
    double trust_threshold = 0.5;
    
    // 階層別の重み
    std::vector<double> level_weights = {0.3, 0.5, 0.2};
    
    // 距離ベースの減衰
    bool use_distance_decay = true;
    double decay_factor = 0.1;
    
    // 時間的減衰
    bool use_temporal_decay = true;
    double temporal_decay_rate = 0.95;
};
```

### SOMConfig

Self-Organizing Mapの設定です。

```cpp
struct SOMConfig {
    // グリッドサイズ
    int grid_width = 8;
    int grid_height = 8;
    
    // 学習パラメータ
    double initial_learning_rate = 0.5;
    double initial_radius = 3.0;
    double learning_decay = 0.99;
    double radius_decay = 0.99;
    
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
```

## 使用例

### 基本的な時系列予測

```cpp
#include <crlgru/crl_gru.hpp>
#include <iostream>

int main() {
    // 設定
    crlgru::FEPGRUCell::Config config;
    config.input_size = 10;
    config.hidden_size = 64;
    config.beta = 1.0;
    config.enable_som_extraction = true;
    
    // FEP-GRUセル作成
    auto gru_cell = std::make_shared<crlgru::FEPGRUCell>(config);
    
    // データ生成
    int sequence_length = 100;
    auto input_sequence = torch::randn({sequence_length, 1, 10});
    auto hidden = torch::zeros({1, 64});
    
    // 予測と学習
    std::vector<double> free_energies;
    
    for (int t = 0; t < sequence_length; ++t) {
        auto input = input_sequence[t];
        
        // フォワードパス
        auto [new_hidden, prediction, free_energy] = gru_cell->forward(input, hidden);
        
        // 自由エネルギーを記録
        free_energies.push_back(free_energy.mean().item<double>());
        
        // 隠れ状態を更新
        hidden = new_hidden;
        
        // 10ステップごとに出力
        if ((t + 1) % 10 == 0) {
            std::cout << "Step " << (t + 1) << "/" << sequence_length
                     << " - Free Energy: " << free_energies.back() << std::endl;
        }
    }
    
    // 平均自由エネルギー
    double avg_fe = std::accumulate(free_energies.begin(), free_energies.end(), 0.0) 
                    / free_energies.size();
    std::cout << "\n平均自由エネルギー: " << avg_fe << std::endl;
    
    return 0;
}
```

### マルチエージェント協調

```cpp
#include <crlgru/crl_gru.hpp>
#include <vector>
#include <random>

class SwarmAgent {
private:
    std::shared_ptr<crlgru::FEPGRUNetwork> brain;
    std::shared_ptr<crlgru::PolarSpatialAttention> spatial_attention;
    torch::Tensor position;
    torch::Tensor velocity;
    torch::Tensor hidden_state;
    
public:
    SwarmAgent(const crlgru::FEPGRUNetwork::NetworkConfig& brain_config,
               const crlgru::PolarSpatialAttention::AttentionConfig& attention_config,
               const torch::Tensor& initial_position)
        : position(initial_position.clone()),
          velocity(torch::zeros_like(initial_position)) {
        
        brain = std::make_shared<crlgru::FEPGRUNetwork>(brain_config);
        spatial_attention = std::make_shared<crlgru::PolarSpatialAttention>(attention_config);
        hidden_state = torch::zeros({1, brain_config.layer_sizes.back()});
    }
    
    void perceive_neighbors(const std::vector<std::shared_ptr<SwarmAgent>>& neighbors) {
        // 近隣エージェントの位置を収集
        std::vector<torch::Tensor> neighbor_positions;
        for (const auto& neighbor : neighbors) {
            neighbor_positions.push_back(neighbor->get_position());
        }
        
        if (neighbor_positions.empty()) return;
        
        // 極座標マップに変換
        auto positions_tensor = torch::stack(neighbor_positions).unsqueeze(0);
        auto polar_map = crlgru::utils::cartesian_to_polar_map(
            positions_tensor, position.unsqueeze(0), 8, 16, 10.0
        );
        
        // 空間注意を適用
        auto attended_map = spatial_attention->forward(
            polar_map.unsqueeze(1).expand({1, 64, 8, 16})
        );
        
        // 特徴を抽出してネットワークに入力
        auto features = attended_map.flatten(1);
        auto [new_hidden, prediction, free_energy] = brain->forward(features, hidden_state);
        hidden_state = new_hidden;
        
        // 速度を更新
        velocity = prediction[0].slice(0, 0, 2).tanh() * 0.1;
    }
    
    void update_position(double dt = 1.0) {
        position += velocity * dt;
    }
    
    torch::Tensor get_position() const { return position; }
    torch::Tensor get_velocity() const { return velocity; }
    
    void share_knowledge(std::shared_ptr<SwarmAgent> partner, double trust_score) {
        brain->share_parameters_with_agents({partner->brain}, {trust_score});
    }
};

int main() {
    // ネットワーク設定
    crlgru::FEPGRUNetwork::NetworkConfig brain_config;
    brain_config.layer_sizes = {64, 128, 64};
    brain_config.cell_config.input_size = 64;
    brain_config.cell_config.hidden_size = 128;
    
    // 空間注意設定
    crlgru::PolarSpatialAttention::AttentionConfig attention_config;
    attention_config.input_channels = 64;
    attention_config.num_distance_rings = 8;
    attention_config.num_angle_sectors = 16;
    
    // スワーム作成
    const int num_agents = 10;
    std::vector<std::shared_ptr<SwarmAgent>> swarm;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-5.0, 5.0);
    
    for (int i = 0; i < num_agents; ++i) {
        auto initial_pos = torch::tensor({dis(gen), dis(gen)});
        swarm.push_back(std::make_shared<SwarmAgent>(
            brain_config, attention_config, initial_pos
        ));
    }
    
    // シミュレーション
    const int num_steps = 100;
    for (int step = 0; step < num_steps; ++step) {
        // 各エージェントが近隣を知覚
        for (int i = 0; i < num_agents; ++i) {
            std::vector<std::shared_ptr<SwarmAgent>> neighbors;
            
            // 近隣エージェントを収集（距離ベース）
            for (int j = 0; j < num_agents; ++j) {
                if (i == j) continue;
                
                auto dist = (swarm[i]->get_position() - swarm[j]->get_position()).norm().item<double>();
                if (dist < 5.0) {
                    neighbors.push_back(swarm[j]);
                }
            }
            
            swarm[i]->perceive_neighbors(neighbors);
        }
        
        // 位置更新
        for (auto& agent : swarm) {
            agent->update_position();
        }
        
        // 知識共有（10ステップごと）
        if ((step + 1) % 10 == 0) {
            for (int i = 0; i < num_agents - 1; ++i) {
                swarm[i]->share_knowledge(swarm[i + 1], 0.1);
            }
            
            std::cout << "Step " << (step + 1) << " - 知識共有完了" << std::endl;
        }
    }
    
    std::cout << "\nシミュレーション完了" << std::endl;
    
    return 0;
}
```

### カスタム評価関数の実装

```cpp
#include <crlgru/crl_gru.hpp>

// カスタム評価関数
class CustomEvaluator : public crlgru::MetaEvaluator {
private:
    double target_distance;
    torch::Tensor target_position;
    
public:
    CustomEvaluator(const EvaluationConfig& config, 
                   double target_dist, 
                   const torch::Tensor& target_pos)
        : MetaEvaluator(config), 
          target_distance(target_dist),
          target_position(target_pos) {}
    
    std::unordered_map<std::string, double> evaluate(
        const torch::Tensor& state,
        const torch::Tensor& target,
        const std::unordered_map<std::string, torch::Tensor>& context) override {
        
        auto metrics = MetaEvaluator::evaluate(state, target, context);
        
        // カスタムメトリック：目標との距離
        if (context.find("position") != context.end()) {
            auto position = context.at("position");
            double distance = (position - target_position).norm().item<double>();
            metrics["target_distance_error"] = std::abs(distance - target_distance);
        }
        
        // カスタムメトリック：エネルギー効率
        if (context.find("velocity") != context.end()) {
            auto velocity = context.at("velocity");
            double energy = velocity.pow(2).sum().item<double>();
            metrics["energy_efficiency"] = 1.0 / (1.0 + energy);
        }
        
        return metrics;
    }
};

int main() {
    // 評価設定
    crlgru::MetaEvaluator::EvaluationConfig eval_config;
    eval_config.metrics = {
        "prediction_accuracy",
        "free_energy",
        "target_distance_error",
        "energy_efficiency"
    };
    eval_config.adaptive_weights = true;
    
    // カスタム評価器作成
    auto target_pos = torch::tensor({10.0, 10.0});
    auto evaluator = std::make_shared<CustomEvaluator>(eval_config, 5.0, target_pos);
    
    // 使用例
    auto state = torch::randn({1, 64});
    auto position = torch::tensor({7.0, 8.0});
    auto velocity = torch::tensor({0.5, 0.3});
    
    std::unordered_map<std::string, torch::Tensor> context = {
        {"position", position},
        {"velocity", velocity}
    };
    
    auto metrics = evaluator->evaluate(state, torch::Tensor(), context);
    
    // 結果表示
    std::cout << "評価結果:" << std::endl;
    for (const auto& [name, value] : metrics) {
        std::cout << "  " << name << ": " << value << std::endl;
    }
    
    // 複合スコア
    double score = evaluator->compute_composite_score(metrics);
    std::cout << "\n複合スコア: " << score << std::endl;
    
    return 0;
}
```

## パフォーマンス最適化

### メモリ管理

```cpp
// テンソルのメモリ事前割り当て
class TensorPool {
private:
    std::unordered_map<std::string, torch::Tensor> pool;
    
public:
    torch::Tensor get_tensor(const std::string& name, 
                           const std::vector<int64_t>& shape,
                           bool zero_init = true) {
        auto key = name + "_" + std::to_string(shape[0]);
        
        if (pool.find(key) == pool.end()) {
            pool[key] = zero_init ? torch::zeros(shape) : torch::empty(shape);
        }
        
        return pool[key];
    }
    
    void clear() {
        pool.clear();
    }
};
```

### バッチ処理

```cpp
// 効率的なバッチ処理
void process_batch(
    std::shared_ptr<crlgru::FEPGRUNetwork> network,
    const torch::Tensor& batch_input,  // [batch_size, seq_len, input_dim]
    int chunk_size = 32) {
    
    int batch_size = batch_input.size(0);
    int num_chunks = (batch_size + chunk_size - 1) / chunk_size;
    
    std::vector<torch::Tensor> outputs;
    
    for (int i = 0; i < num_chunks; ++i) {
        int start = i * chunk_size;
        int end = std::min(start + chunk_size, batch_size);
        
        auto chunk = batch_input.slice(0, start, end);
        auto [output, _, _] = network->forward(chunk);
        outputs.push_back(output);
    }
    
    auto full_output = torch::cat(outputs, 0);
}
```

### 並列処理

```cpp
// OpenMPを使用した並列処理
#include <omp.h>

void parallel_agent_update(
    std::vector<std::shared_ptr<SwarmAgent>>& agents,
    const std::function<void(std::shared_ptr<SwarmAgent>)>& update_function) {
    
    #pragma omp parallel for
    for (size_t i = 0; i < agents.size(); ++i) {
        update_function(agents[i]);
    }
}
```

## デバッグとプロファイリング

### デバッグユーティリティ

```cpp
// テンソルの統計情報表示
void print_tensor_stats(const torch::Tensor& tensor, const std::string& name) {
    std::cout << "=== " << name << " ===" << std::endl;
    std::cout << "Shape: " << tensor.sizes() << std::endl;
    std::cout << "Mean: " << tensor.mean().item<double>() << std::endl;
    std::cout << "Std: " << tensor.std().item<double>() << std::endl;
    std::cout << "Min: " << tensor.min().item<double>() << std::endl;
    std::cout << "Max: " << tensor.max().item<double>() << std::endl;
    std::cout << "Has NaN: " << tensor.isnan().any().item<bool>() << std::endl;
    std::cout << "Has Inf: " << tensor.isinf().any().item<bool>() << std::endl;
}

// 勾配チェック
void check_gradients(const std::unordered_map<std::string, torch::Tensor>& params) {
    for (const auto& [name, param] : params) {
        if (param.requires_grad() && param.grad().defined()) {
            auto grad_norm = param.grad().norm().item<double>();
            if (grad_norm > 10.0) {
                std::cerr << "Warning: Large gradient in " << name 
                         << " (norm=" << grad_norm << ")" << std::endl;
            }
        }
    }
}
```

### プロファイリング

```cpp
// 簡易プロファイラ
class SimpleProfiler {
private:
    std::unordered_map<std::string, std::chrono::duration<double>> timings;
    std::unordered_map<std::string, int> counts;
    
public:
    class Timer {
    private:
        SimpleProfiler* profiler;
        std::string name;
        std::chrono::high_resolution_clock::time_point start;
        
    public:
        Timer(SimpleProfiler* p, const std::string& n) 
            : profiler(p), name(n), start(std::chrono::high_resolution_clock::now()) {}
        
        ~Timer() {
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration<double>(end - start);
            profiler->add_timing(name, duration);
        }
    };
    
    Timer time(const std::string& name) {
        return Timer(this, name);
    }
    
    void add_timing(const std::string& name, std::chrono::duration<double> duration) {
        timings[name] += duration;
        counts[name]++;
    }
    
    void print_summary() {
        std::cout << "\n=== Profiling Summary ===" << std::endl;
        for (const auto& [name, total_time] : timings) {
            double avg_time = total_time.count() / counts[name];
            std::cout << name << ": "
                     << "Total=" << total_time.count() << "s, "
                     << "Avg=" << avg_time * 1000 << "ms, "
                     << "Count=" << counts[name] << std::endl;
        }
    }
};
```

---

**注意**: このAPIリファレンスは、crlGRUライブラリの主要な機能を網羅していますが、実装の詳細は変更される可能性があります。最新の情報は、ヘッダーファイル（`crl_gru.hpp`）を直接参照してください。
