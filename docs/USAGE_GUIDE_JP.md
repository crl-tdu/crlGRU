# crlGRU ライブラリ使用ガイド

## 概要

crlGRUは自由エネルギー原理に基づくGRUニューラルネットワークライブラリです。マルチエージェントシステムとスワーム知能研究に特化した革新的な機械学習フレームワークを提供します。

## 環境要件

### 必須環境
- **C++17以上**: 言語標準
- **CMake 3.18以上**: ビルドシステム
- **LibTorch 2.1.2以上**: PyTorch C++ API

### 推奨環境
- **macOS/Linux**: クロスプラットフォーム対応
- **OpenMP**: 並列計算サポート（オプション）

## インストール

### 1. LibTorchのインストール

詳細は [`LIBTORCH_INSTALLATION_GUIDE.md`](LIBTORCH_INSTALLATION_GUIDE.md) を参照してください。

```bash
# LibTorchを$HOME/local/libtorchにインストール済みの前提
export LIBTORCH_PATH=$HOME/local/libtorch
```

### 2. crlGRUライブラリのビルド

```bash
# プロジェクトディレクトリに移動
cd /Users/igarashi/local/project_workspace/crlGRU

# テスト用ビルド（推奨）
mkdir -p build_test && cd build_test
cmake .. -DCMAKE_BUILD_TYPE=Debug
make -j$(nproc)

# 統合テスト実行
./tests/test_crlgru

# 本格ビルド（リリース版）
cd ..
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# システムへのインストール（オプション）
make install
```

## 基本的な使用方法

### 1. ヘッダーファイルのインクルード

```cpp
#include <crlgru/crl_gru.hpp>
#include <iostream>
#include <memory>
```

### 2. 最小限のサンプル

```cpp
int main() {
    // FEP-GRUセルの設定
    crlgru::FEPGRUCell::Config config;
    config.input_size = 10;
    config.hidden_size = 64;
    config.enable_som_extraction = true;
    
    // FEP-GRUセルの作成
    auto gru_cell = std::make_shared<crlgru::FEPGRUCell>(config);
    
    // 入力データの準備
    auto input = torch::randn({1, 10});
    auto hidden = torch::zeros({1, 64});
    
    // フォワードパス実行
    auto [new_hidden, prediction, free_energy] = gru_cell->forward(input, hidden);
    
    // 結果の表示
    std::cout << "自由エネルギー: " << free_energy.mean().item<double>() << std::endl;
    std::cout << "新しい隠れ状態サイズ: " << new_hidden.sizes() << std::endl;
    
    return 0;
}
```

### 3. CMakeLists.txt設定例

```cmake
cmake_minimum_required(VERSION 3.18)
project(my_crlgru_app)

set(CMAKE_CXX_STANDARD 17)

# LibTorchを検索
find_package(Torch REQUIRED PATHS $ENV{HOME}/local/libtorch)

# crlGRUライブラリを検索（ローカルビルドの場合）
find_library(CRLGRU_LIB 
    NAMES crlGRU
    PATHS ${CMAKE_CURRENT_SOURCE_DIR}/build_test
          ${CMAKE_CURRENT_SOURCE_DIR}/build
          $ENV{HOME}/local/lib
    REQUIRED)

find_path(CRLGRU_INCLUDE 
    NAMES crlgru/crl_gru.hpp
    PATHS ${CMAKE_CURRENT_SOURCE_DIR}/include
          $ENV{HOME}/local/include
    REQUIRED)

# 実行ファイルの作成
add_executable(my_app main.cpp)

# インクルードディレクトリとライブラリをリンク
target_include_directories(my_app PRIVATE ${CRLGRU_INCLUDE})
target_link_libraries(my_app "${TORCH_LIBRARIES}" ${CRLGRU_LIB})

# LibTorch設定
set_property(TARGET my_app PROPERTY CXX_STANDARD 17)
```

## 主要コンポーネントの使用例

### 1. FEP-GRUネットワーク

```cpp
#include <crlgru/crl_gru.hpp>

int main() {
    // ネットワーク設定
    crlgru::FEPGRUNetwork::NetworkConfig config;
    config.layer_sizes = {32, 64, 32};
    config.cell_config.input_size = 32;
    config.cell_config.hidden_size = 64;
    config.enable_hierarchical_imitation = true;
    
    // ネットワーク作成
    auto network = std::make_shared<crlgru::FEPGRUNetwork>(config);
    
    // シーケンスデータ
    auto sequence = torch::randn({10, 1, 32}); // [seq_len, batch, input_size]
    
    // ネットワーク実行
    auto [output, prediction, free_energy] = network->forward(sequence);
    
    std::cout << "出力形状: " << output.sizes() << std::endl;
    std::cout << "集合自由エネルギー: " << free_energy.mean().item<double>() << std::endl;
    
    return 0;
}
```

### 2. 極座標空間注意機構

```cpp
#include <crlgru/crl_gru.hpp>

int main() {
    // 空間注意設定
    crlgru::PolarSpatialAttention::AttentionConfig config;
    config.input_channels = 64;
    config.num_distance_rings = 8;
    config.num_angle_sectors = 16;
    config.max_range = 10.0;
    
    // 空間注意機構作成
    auto spatial_attention = std::make_shared<crlgru::PolarSpatialAttention>(config);
    
    // エージェント位置データ
    std::vector<torch::Tensor> agent_positions = {
        torch::tensor({1.0, 2.0}),
        torch::tensor({-1.5, 0.5}),
        torch::tensor({2.0, -1.0})
    };
    auto self_position = torch::tensor({0.0, 0.0});
    
    // 極座標マップ生成
    auto positions_tensor = torch::stack(agent_positions).unsqueeze(0);
    auto polar_map = crlgru::utils::cartesian_to_polar_map(
        positions_tensor, self_position.unsqueeze(0), 8, 16, 10.0
    );
    
    // 注意適用
    auto attended_map = spatial_attention->forward(
        polar_map.unsqueeze(1).expand({1, 64, 8, 16})
    );
    
    std::cout << "注意適用後マップ形状: " << attended_map.sizes() << std::endl;
    
    return 0;
}
```

### 3. SPSA最適化

```cpp
#include <crlgru/crl_gru.hpp>

int main() {
    // 最適化設定
    crlgru::SPSAOptimizer::OptimizerConfig config;
    config.a = 0.16;
    config.c = 0.16;
    config.alpha = 0.602;
    config.gamma = 0.101;
    
    // 最適化対象パラメータ
    auto parameters = torch::randn({10}, torch::requires_grad(true));
    std::vector<torch::Tensor> param_list = {parameters};
    
    // SPSA最適化器作成
    auto optimizer = std::make_shared<crlgru::SPSAOptimizer>(param_list, config);
    
    // 目的関数（例：二次関数）
    auto objective_function = [&]() -> double {
        auto loss = (parameters - torch::ones({10})).pow(2).sum();
        return loss.item<double>();
    };
    
    // 最適化実行
    for (int i = 0; i < 100; ++i) {
        optimizer->step(objective_function, i);
        
        if ((i + 1) % 20 == 0) {
            std::cout << "Iteration " << (i + 1) << ", Loss: " 
                     << objective_function() << std::endl;
        }
    }
    
    return 0;
}
```

## 統合例：マルチエージェントシステム

```cpp
#include <crlgru/crl_gru.hpp>
#include <vector>
#include <random>

class SimpleAgent {
private:
    std::shared_ptr<crlgru::FEPGRUCell> brain_;
    torch::Tensor position_;
    torch::Tensor hidden_state_;
    
public:
    SimpleAgent(const crlgru::FEPGRUCell::Config& config, 
                const torch::Tensor& initial_pos)
        : position_(initial_pos.clone()) {
        brain_ = std::make_shared<crlgru::FEPGRUCell>(config);
        hidden_state_ = torch::zeros({1, config.hidden_size});
    }
    
    void update(const torch::Tensor& observation) {
        auto [new_hidden, prediction, free_energy] = 
            brain_->forward(observation, hidden_state_);
        hidden_state_ = new_hidden;
        
        // 簡単な移動ルール
        auto velocity = prediction.slice(1, 0, 2).tanh() * 0.1;
        position_ += velocity.squeeze(0);
    }
    
    torch::Tensor get_position() const { return position_; }
    std::shared_ptr<crlgru::FEPGRUCell> get_brain() const { return brain_; }
};

int main() {
    // エージェント設定
    crlgru::FEPGRUCell::Config config;
    config.input_size = 4; // 観測データ
    config.hidden_size = 32;
    config.enable_som_extraction = true;
    
    // 複数エージェント作成
    std::vector<std::unique_ptr<SimpleAgent>> agents;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-2.0, 2.0);
    
    for (int i = 0; i < 5; ++i) {
        auto initial_pos = torch::tensor({dis(gen), dis(gen)});
        agents.push_back(std::make_unique<SimpleAgent>(config, initial_pos));
    }
    
    // シミュレーション実行
    for (int step = 0; step < 50; ++step) {
        for (auto& agent : agents) {
            // 簡単な観測データ生成
            auto observation = torch::randn({1, 4});
            agent->update(observation);
        }
        
        if ((step + 1) % 10 == 0) {
            std::cout << "Step " << (step + 1) << " - Positions:" << std::endl;
            for (size_t i = 0; i < agents.size(); ++i) {
                auto pos = agents[i]->get_position();
                std::cout << "  Agent " << i << ": [" 
                         << pos[0].item<double>() << ", " 
                         << pos[1].item<double>() << "]" << std::endl;
            }
        }
    }
    
    return 0;
}
```

## デバッグとトラブルシューティング

### 1. ビルドエラー

**LibTorchが見つからない場合**:
```bash
export LIBTORCH_PATH=/Users/igarashi/local/libtorch
export CMAKE_PREFIX_PATH=$LIBTORCH_PATH:$CMAKE_PREFIX_PATH
```

**動的ライブラリエラー**:
```bash
export DYLD_LIBRARY_PATH=$LIBTORCH_PATH/lib:$DYLD_LIBRARY_PATH
```

### 2. 実行時エラー

**テンソルサイズ不一致**:
- 入力データの次元を確認してください
- `input_size`と実際の入力次元が一致することを確認

**メモリエラー**:
- 大きなバッチサイズを避ける
- 適切なスコープ管理でテンソルを解放

### 3. パフォーマンス最適化

**CPU最適化**:
```cpp
// OpenMPを有効にしてビルド
// CMakeLists.txtに追加:
// find_package(OpenMP REQUIRED)
// target_link_libraries(my_app OpenMP::OpenMP_CXX)
```

**メモリ使用量削減**:
```cpp
// 不要なテンソルの明示的な解放
torch::cuda::empty_cache(); // GPU使用時
```

## 詳細情報とリファレンス

- **API詳細**: [`API_REFERENCE_JP.md`](API_REFERENCE_JP.md)
- **理論的基盤**: [`THEORETICAL_FOUNDATIONS.md`](THEORETICAL_FOUNDATIONS.md)
- **開発者向け**: [`DEVELOPMENT_BASELINE_PROMPT.md`](DEVELOPMENT_BASELINE_PROMPT.md)
- **LibTorchインストール**: [`LIBTORCH_INSTALLATION_GUIDE.md`](LIBTORCH_INSTALLATION_GUIDE.md)

## サンプルプロジェクト

統合テストファイル `tests/test_crlgru.cpp` には18個の包括的なテストケースが含まれており、各機能の使用例として参考にできます。

```bash
# 統合テスト実行
cd /Users/igarashi/local/project_workspace/crlGRU/build_test
./tests/test_crlgru
```

## サポート

問題や質問がある場合は、プロジェクトリポジトリのIssueを利用するか、開発者にお問い合わせください。理論的な背景や実装の詳細については、理論基盤ドキュメントを参照してください。
