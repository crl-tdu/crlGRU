# crlGRU ライブラリ - 身体性FEP統合版

[![Version](https://img.shields.io/badge/version-1.2-blue.svg)](https://github.com/crl-tdu/crlGRU/releases/tag/v1.2)
[![LibTorch](https://img.shields.io/badge/LibTorch-2.7.0-orange.svg)](https://pytorch.org/cppdocs/)
[![crlNexus](https://img.shields.io/badge/crlNexus-Compatible-green.svg)](https://github.com/crl-tdu/crlNexus)

## 📌 概要

crlGRUは、**身体性AI**と**Free Energy Principle (自由エネルギー原理)**を完全統合したC++ライブラリです。従来の仮想環境でのMARLを超越し、**物理制約・センサーノイズ・部分観測性**を考慮した現実的な群制御システムを実現します。

## 🚀 **Release 1.2 - crlNexus SEPAM統合対応**

### ✨ **新機能・改善**
- **crlNexus統合最適化**: SEPAM (Saliency-Enhanced Predictive Attention Model) との完全統合
- **ビルド安定性向上**: CMake設定の最適化とエラー修正
- **互換性強化**: LibTorch 2.7.0 対応、macOS ARM64/x86_64 サポート
- **パフォーマンス改善**: メモリ使用量最適化とコンパイル時間短縮

## 🧠 **NEW!** 身体性FEP統合機能

### 🧠 **EmbodiedFEPGRUCell**
- **物理制約考慮**: 質量・慣性・摩擦を統合したFEP-GRU
- **顕在性極座標統合**: crlNexus `SoftAssignmentPolarMap`との完全互換
- **108次元特徴対応**: 既存システムとの無縫統合
- **微分可能制御**: 端から端まで微分可能な制御システム

### 🗺️ **顕在性極座標マップ統合**
```cpp
// crlNexusとの完全統合例
#include <crlgru/integration/nexus_compatibility.hpp>

crlgru::integration::NexusCompatibilityLayer compatibility;
crlnexus::swarm::base::SoftAssignmentPolarMap saliency_map;

// 108次元特徴ベクトルの相互変換
auto conversion_data = compatibility.convert_from_saliency_map(
    saliency_map, agent_position
);

// 制御勾配の計算（∂features/∂u）
auto [grad_ux, grad_uy] = compatibility.compute_control_gradients(
    embodied_gru_cell, control_input, saliency_map, neighbors
);
```

### 🔧 **物理制約レイヤー**
```cpp
// 身体性物理制約の例
crlgru::core::PhysicalConstraintConfig constraint_config;
constraint_config.mass = 1.0;              // 質量 [kg]
constraint_config.inertia = 0.1;           // 慣性モーメント [kg⋅m²]
constraint_config.max_force = 10.0;        // 最大制御力 [N]
constraint_config.friction_coefficient = 0.1; // 摩擦係数

auto constraint_layer = crlgru::core::create_physical_constraint_layer(constraint_config);

// 物理的に妥当な制御入力の生成
auto constrained_control = constraint_layer->apply_constraints(
    raw_control_input, current_physical_state
);
```

### 📡 **身体性センサーモデル**
```cpp
// リアルなセンサーノイズ・遅延のシミュレーション
crlgru::integration::EmbodiedSensorConfig sensor_config;
sensor_config.noise_variance = 0.01;       // ノイズ分散
sensor_config.measurement_delay = 0.033;   // 30FPS遅延
sensor_config.enable_kalman_filter = true; // カルマンフィルタ

auto sensor_model = crlgru::integration::create_embodied_sensor_model(sensor_config);

// ノイズ・遅延を含む観測の生成
auto noisy_observation = sensor_model->simulate_observation(true_state, current_time);
```

## 🚀 主な機能

### 🧠 **FEP-GRU Cell**
- 自由エネルギー原理に基づく予測的符号化
- リアルタイムパラメータ修正による相互模倣学習
- SOM (Self-Organizing Map) 特徴抽出
- メタ評価関数統合

### 🤖 **Multi-Agent Coordination**
- 極座標空間注意メカニズム
- 階層的模倣学習 (3レベル)
- リアルタイムパラメータ共有
- 信頼度ベースの協調

### 🔧 **Optimization Tools**
- SPSA (Simultaneous Perturbation Stochastic Approximation) オプティマイザー
- メタ評価による多目的最適化
- 適応的重み調整

## 📋 前提条件

- **LibTorch** (PyTorch C++ API) - [インストールガイド](./LIBTORCH_INSTALL_JP.md)を参照
- **CMake** 3.18以降
- **C++17**対応コンパイラ
- **OpenMP** (オプション)

## 💾 インストール済み内容

ライブラリは `$HOME/local/` にインストールされています：

```
$HOME/local/
├── lib/
│   ├── libcrlGRU.dylib          # メインライブラリ
│   ├── libc10.dylib             # PyTorch C++ライブラリ
│   ├── libtorch.dylib           # PyTorchメインライブラリ
│   ├── libtorch_cpu.dylib       # PyTorch CPUライブラリ
│   └── ...                      # その他依存ライブラリ
├── include/
│   └── crlgru/
│       └── crl_gru.hpp          # APIヘッダーファイル
└── bin/
    ├── simple_prediction_example    # 時系列予測サンプル
    └── swarm_coordination_example   # スワーム協調サンプル
```

## 使用方法

### 1. サンプルプログラムの実行

#### 時系列予測サンプル
```bash
$HOME/local/bin/simple_prediction_example
```

#### スワーム協調シミュレーション
```bash
$HOME/local/bin/swarm_coordination_example
```

### 2. 新しいプロジェクトでの使用

#### CMakeを使用する場合

**CMakeLists.txt:**
```cmake
cmake_minimum_required(VERSION 3.18)
project(YourProject)

# C++17を使用
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# crlGRUライブラリを検索
find_package(PkgConfig REQUIRED)
find_library(CRLGRU_LIBRARY crlGRU PATHS $ENV{HOME}/local/lib)
find_path(CRLGRU_INCLUDE_DIR crlgru/crl_gru.hpp PATHS $ENV{HOME}/local/include)

# 実行ファイルを作成
add_executable(your_program main.cpp)

# インクルードディレクトリとライブラリをリンク
target_include_directories(your_program PRIVATE ${CRLGRU_INCLUDE_DIR})
target_link_libraries(your_program ${CRLGRU_LIBRARY})

# RPATHを設定（実行時にライブラリを見つけるため）
set_target_properties(your_program PROPERTIES
    INSTALL_RPATH "$ENV{HOME}/local/lib")
```

**ビルド方法:**
```bash
mkdir build && cd build
cmake ..
make
```

#### 手動コンパイルの場合

```bash
# 基本的なコンパイル
g++ -std=c++17 -I$HOME/local/include -L$HOME/local/lib -lcrlGRU your_program.cpp -o your_program

# 詳細オプション付き
g++ -std=c++17 \
    -I$HOME/local/include \
    -L$HOME/local/lib \
    -lcrlGRU \
    -Wl,-rpath,$HOME/local/lib \
    your_program.cpp -o your_program
```

### 3. 基本的なC++コード例

#### FEP-GRU Cellの基本使用例

```cpp
#include <crlgru/crl_gru.hpp>
#include <iostream>

int main() {
    // FEP-GRU設定
    crlgru::FEPGRUCell::Config config;
    config.input_size = 10;
    config.hidden_size = 64;
    config.enable_som_extraction = true;
    
    // FEP-GRUセル作成
    auto gru_cell = std::make_shared<crlgru::FEPGRUCell>(config);
    
    // 入力データ
    auto input = torch::randn({1, 10});
    auto hidden = torch::zeros({1, 64});
    
    // フォワードパス
    auto [new_hidden, prediction, free_energy] = gru_cell->forward(input, hidden);
    
    std::cout << "自由エネルギー: " << free_energy.mean().item<double>() << std::endl;
    
    return 0;
}
```

#### スワームエージェントの例

```cpp
#include <crlgru/crl_gru.hpp>
#include <vector>

int main() {
    // ネットワーク設定
    crlgru::FEPGRUNetwork::NetworkConfig network_config;
    network_config.layer_sizes = {64, 128, 64};
    network_config.cell_config.input_size = 64;
    network_config.cell_config.hidden_size = 128;
    
    // FEP-GRUネットワーク作成
    auto brain = std::make_shared<crlgru::FEPGRUNetwork>(network_config);
    
    // 空間注意モジュール設定
    crlgru::PolarSpatialAttention::AttentionConfig attention_config;
    attention_config.input_channels = 64;
    attention_config.num_distance_rings = 8;
    attention_config.num_angle_sectors = 16;
    
    auto attention = std::make_shared<crlgru::PolarSpatialAttention>(attention_config);
    
    // 極座標マップの生成と処理
    auto positions = torch::randn({1, 5, 2}); // 5エージェントの位置
    auto self_pos = torch::zeros({1, 2});
    
    auto polar_map = crlgru::utils::cartesian_to_polar_map(
        positions, self_pos, 8, 16, 10.0);
    
    // 注意メカニズム適用
    auto expanded_map = polar_map.unsqueeze(1).expand({1, 64, 8, 16});
    auto attended_map = attention->forward(expanded_map);
    
    return 0;
}
```

## API リファレンス

### 主要クラス

#### `FEPGRUCell`
- **目的**: 自由エネルギー原理に基づくGRUセル
- **主要メソッド**:
  - `forward()`: フォワードパス実行
  - `extract_som_features()`: SOM特徴抽出
  - `update_parameters_from_peer()`: ピア学習

#### `FEPGRUNetwork`
- **目的**: 多層FEP-GRUネットワーク
- **主要メソッド**:
  - `forward()`: シーケンス処理
  - `share_parameters_with_agents()`: エージェント間パラメータ共有

#### `PolarSpatialAttention`
- **目的**: 極座標空間注意メカニズム
- **主要メソッド**:
  - `forward()`: 空間注意適用

#### `MetaEvaluator`
- **目的**: 多目的評価・最適化
- **主要メソッド**:
  - `evaluate()`: 状態評価
  - `adapt_weights()`: 重み適応

### ユーティリティ関数

```cpp
namespace crlgru::utils {
    // 極座標変換
    torch::Tensor cartesian_to_polar_map(positions, self_position, rings, sectors, range);
    
    // 相互情報量計算
    double compute_mutual_information(state1, state2);
    
    // ガウシアンカーネル適用
    torch::Tensor apply_gaussian_kernel(input, sigma, kernel_size);
    
    // 信頼度メトリック計算
    double compute_trust_metric(performance_history, distance, max_distance);
    
    // パラメータ保存・読み込み
    void save_parameters(filename, parameters);
    std::unordered_map<std::string, torch::Tensor> load_parameters(filename);
}
```

## トラブルシューティング

### よくある問題

#### 1. ライブラリが見つからないエラー
```
dyld: Library not loaded: @rpath/libcrlGRU.dylib
```

**解決方法:**
- RPATHが正しく設定されているか確認
- `$HOME/local/lib/` にライブラリファイルが存在するか確認

#### 2. ヘッダーファイルが見つからないエラー
```
fatal error: 'crlgru/crl_gru.hpp' file not found
```

**解決方法:**
- インクルードパスに `-I$HOME/local/include` を追加
- `$HOME/local/include/crlgru/crl_gru.hpp` が存在するか確認

#### 3. リンクエラー
```
Undefined symbols for architecture arm64
```

**解決方法:**
- `-lcrlGRU` フラグが含まれているか確認
- ライブラリパス `-L$HOME/local/lib` が設定されているか確認

### デバッグ方法

```bash
# ライブラリの依存関係確認
otool -L $HOME/local/lib/libcrlGRU.dylib

# バイナリのRPATH確認
otool -l your_program | grep -A 2 LC_RPATH

# ライブラリパス確認
echo $HOME/local/lib
ls -la $HOME/local/lib/libcrlGRU*
```

## 研究・開発での活用

### 適用分野
- **マルチエージェントシステム**: 協調行動・群集知能
- **ロボティクス**: スワームロボティクス・分散制御
- **時系列予測**: 予測的符号化による時系列モデル
- **強化学習**: 自由エネルギー最小化による学習

### カスタマイズのポイント
- **目的関数**: `MetaEvaluator`で独自の評価基準を追加
- **注意メカニズム**: `PolarSpatialAttention`の設定調整
- **模倣学習**: `hierarchical_imitation_update`の実装拡張
- **最適化**: `SPSAOptimizer`のパラメータ調整

## 📚 開発者向けドキュメント

プロジェクト開発に参加する場合は、以下のドキュメントを参照してください：

- **[開発ベースラインプロンプト](./docs/DEVELOPMENT_BASELINE_PROMPT.md)**: プロジェクト開発の包括的ガイドライン
- **[API リファレンス](./docs/API_REFERENCE_JP.md)**: 詳細なAPI仕様書
- **[使用ガイド](./docs/USAGE_GUIDE_JP.md)**: 実践的な使用方法
- **[LibTorch インストールガイド](./docs/LIBTORCH_INSTALLATION_GUIDE.md)**: 開発環境構築手順

## ライセンス・引用

このライブラリを研究で使用する場合は、適切な引用をお願いします。詳細はプロジェクトのライセンスファイルを参照してください。

## サポート

- 技術的な質問: プロジェクトのIssuesページ
- バグレポート: GitHubリポジトリ
- 機能リクエスト: Discussionsページ

---

**Happy Coding with crlGRU! 🤖✨**