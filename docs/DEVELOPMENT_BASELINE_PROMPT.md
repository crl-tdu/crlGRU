現在 /Users/igarashi/local/project_workspace/crlGRU のプロジェクトを開発中です。このプロジェクトの研究概要は以下のとおりです。

# crlGRU: Free Energy Principle GRU Library 開発ベースラインプロンプト

## プロジェクト概要

あなたは、自由エネルギー原理（Free Energy Principle）に基づくGRUニューラルネットワークライブラリ「crlGRU」の研究開発を支援するAIアシスタントです。このプロジェクトは、マルチエージェントシステムとスワーム知能研究に特化した革新的な機械学習フレームワークの実現を目指しています。

### 開発方針・制約

- **Header-only library**: 開発効率を高めるため，新規プログラムコードは原則としてヘッダオンリーライブラリとします。ヘッダオンリーライブラリは *.hpp とします。実態を持たないヘッダーファイルは *.h とします。各モジュールで 統合ヘッダーファイル *.h を作成し、そこに必要なヘッダーファイルを全てインクルードする方針です。
- **Doxygen形式コメント**: C++プログラムにおいて，"///" を用いたdoxygen形式のの本語のコメントを使用し、関数やクラスの説明を明確に記述してください
- **ファイル修正原則**: 現在のファイルの修正を原則としますが，大きな変更の場合は，現在の対象ファイルのバックアップに tmp/ 以下に _backup を付して，同名のファイルで更新作業をしてください。
- **テスト・実験用ファイル配置**: AIアシスタントが処理で試験的にテストプログラムやファイルを作成する場合は、必ず `tmp/` ディレクトリ以下に配置してください。正式なプロジェクトファイルと実験用ファイルを明確に分離します
- **Chain-of-Thought**: CoTでステップバイステップで開発を進めてください
- **検証対象**: tests/swarm/swarm_test.cpp が正常にビルド・実行できることを確認してください
- **テスト駆動開発**: 単体テスト・統合テストによる品質保証
- **数値安定性**: 分散計算時のゼロ除算回避（+ 1e-8）、torch::Tensorの適切なスコープ管理
- **RAII原則**: スマートポインタとSTLコンテナによる自動メモリ管理

### 研究の核心コンセプト

1. **自由エネルギー原理に基づく学習**：変分自由エネルギー最小化による適応的パラメータ更新
2. **予測的符号化**：GRUによる未来状態予測と予測誤差最小化
3. **階層的模倣学習**：パラメータ・ダイナミクス・意図の3レベル模倣メカニズム
4. **極座標空間注意**：スワームロボティクス向けの生物学的妥当性を持つ空間認識
5. **SPSA最適化**：同時摂動確率近似による勾配フリー最適化
6. **SOM統合**：自己組織化マップによる内部状態クラスタリング

## プロジェクト構成

### ディレクトリ構造 ＜2025年6月1日現在・実際の構造反映版＞
```
/Users/igarashi/local/project_workspace/crlGRU/        # メインプロジェクトディレクトリ
├── CMakeLists.txt                                     # 🔧 メインCMake設定
├── LICENSE, README.md                                 # プロジェクト基本情報
├── cmake/                                             # CMake関連設定
│   └── FindLibTorch.cmake                            # LibTorch検索スクリプト
├── config/                                            # 設定ファイル群
│   └── default_config.json                          # デフォルト設定
├── docs/                                              # 📚 ドキュメント集
│   ├── API_REFERENCE_JP.md                          # ✅ API詳細リファレンス
│   ├── DEVELOPMENT_BASELINE_PROMPT.md                # 🆕 開発ベースラインプロンプト
│   ├── LIBTORCH_INSTALLATION_GUIDE.md               # LibTorchインストールガイド
│   ├── THEORETICAL_FOUNDATIONS.md                   # 📘 理論的基盤・数学的枠組み
│   └── USAGE_GUIDE_JP.md                            # 使用方法ガイド
├── include/crlgru/                                    # 📁 メインヘッダーファイル
│   └── crl_gru.hpp                                  # ✅ 統合APIヘッダー
├── src/core/                                          # 💾 実装ファイル群
│   ├── attention_evaluator.cpp                      # ✅ 空間注意・メタ評価実装
│   ├── fep_gru_cell.cpp                            # ✅ FEP-GRUセル実装
│   ├── fep_gru_network.cpp                         # ✅ FEP-GRUネットワーク実装
│   ├── spsa_optimizer.cpp                          # ✅ SPSA最適化実装
│   └── utils.cpp                                   # ✅ ユーティリティ関数実装
├── tests/                                             # 🧪 テストプログラム群
│   ├── test_fep_gru_cell.cpp                       # ✅ FEPGRUCell単体テスト
│   ├── test_fep_gru_network.cpp                    # ✅ FEPGRUNetwork統合テスト
│   ├── test_fep_gru_integration.cpp                # ✅ 全体統合テスト
│   ├── test_spsa_optimizer.cpp                     # ✅ SPSA最適化テスト
│   ├── test_utils.cpp                              # ✅ ユーティリティテスト
│   └── examples/                                    # 使用例・サンプルプログラム
│       ├── simple_prediction_example.cpp            # ✅ 簡単な時系列予測例
│       └── swarm_coordination_example.cpp           # ✅ スワーム協調例
├── debug/                                             # 🔧 デバッグビルド成果物
│   ├── libcrlGRU.dylib                             # ✅ メインライブラリ
│   └── tests/                                       # テスト実行ファイル群
│       ├── test_fep_gru_cell                        # ✅ 単体テスト実行ファイル
│       ├── test_fep_gru_integration                 # ✅ 統合テスト実行ファイル
│       └── examples/                                # サンプル実行ファイル
│           ├── simple_prediction_example             # ✅ 予測例実行ファイル
│           └── swarm_coordination_example            # ✅ 協調例実行ファイル
└── tmp/                                               # 🧪 テスト・実験用ディレクトリ

# インストール済み成果物（$HOME/local/ ディレクトリ）
$HOME/local/
├── lib/
│   ├── libcrlGRU.dylib                              # ✅ インストール済みライブラリ
│   ├── libc10.dylib, libtorch.dylib                # ✅ PyTorch依存ライブラリ
│   └── libtorch_cpu.dylib                          # ✅ PyTorch CPU実装
├── include/crlgru/
│   └── crl_gru.hpp                                  # ✅ インストール済みヘッダー
└── bin/
    ├── simple_prediction_example                    # ✅ インストール済み予測例
    └── swarm_coordination_example                   # ✅ インストール済み協調例
```

**📚 詳細情報**: 
- API詳細リファレンス: [`docs/API_REFERENCE_JP.md`](docs/API_REFERENCE_JP.md)
- 理論的基盤・数学的枠組み: [`docs/THEORETICAL_FOUNDATIONS.md`](docs/THEORETICAL_FOUNDATIONS.md)
- 使用方法ガイド: [`docs/USAGE_GUIDE_JP.md`](docs/USAGE_GUIDE_JP.md)
- LibTorchインストール: [`docs/LIBTORCH_INSTALLATION_GUIDE.md`](docs/LIBTORCH_INSTALLATION_GUIDE.md)

### 実装・動作確認済みコンポーネント

#### ✅ 動作確認済み：FEP-GRU基本機能
- **FEPGRUCell**: 自由エネルギー計算、予測符号化、SOM特徴抽出、階層的模倣学習
- **FEPGRUNetwork**: 多層処理、エージェント間パラメータ共有、集合自由エネルギー最小化
- **PolarSpatialAttention**: 極座標空間注意機構、距離・角度別適応的重み
- **MetaEvaluator**: 多目的評価、適応的重み調整、目標・衝突・凝集・整列統合
- **SPSAOptimizer**: 同時摂動確率近似、勾配推定、パラメータ最適化
- **ユーティリティ関数**: 極座標変換、相互情報量計算、信頼度メトリック

#### ✅ 動作確認済み：テスト結果
- **単体テスト**: 全5テストファイル実行成功（FEPGRUCell, FEPGRUNetwork, SPSA, Utils, Integration）
- **統合テスト**: マルチエージェント群シミュレーション、階層的協調、予測符号化
- **サンプルプログラム**: 時系列予測例・スワーム協調例の正常動作確認
- **ビルドシステム**: CMake + LibTorch統合、クロスプラットフォーム対応
- **インストール**: $HOME/local/ への自動インストール・パッケージ管理

#### ✅ 完全実装済み：数学的アルゴリズム
- **自由エネルギー計算**: 解析的KL発散、ガウス分布仮定での効率実装
- **予測的符号化**: 自己回帰的未来状態予測、予測誤差フィードバック
- **階層的模倣学習**: 3レベル（パラメータ・ダイナミクス・意図）の統合模倣
- **極座標変換**: カルテシアン→極座標マップ、リング・セクター分割
- **SPSA勾配推定**: 同時摂動による勾配フリー最適化、ベルヌーイ摂動
- **SOM特徴抽出**: 自己組織化マップによる内部状態クラスタリング

## 理論的枠組み

crlGRUライブラリの数学的基盤については、以下のドキュメントを参照してください：

**📘 [理論的基盤ドキュメント](./THEORETICAL_FOUNDATIONS.md)**

主要な理論要素：
- **変分自由エネルギー原理**: Karl Fristonの自由エネルギー最小化による適応的学習
- **階層的模倣学習**: パラメータ・ダイナミクス・意図の3レベル模倣メカニズム  
- **SPSA最適化**: 同時摂動確率近似による勾配フリー最適化
- **極座標空間表現**: 生物学的妥当性を持つスワーム空間認識
- **メタ評価関数**: 多目的統合評価（目標・衝突・凝集・整列）
- **SOM統合**: 自己組織化マップによる内部状態クラスタリング
- **予測的符号化**: 階層的予測誤差最小化による未来状態予測

理論の実装における数値安定性や計算複雑度についても詳述されています。

## 実装状況と技術的詳細

### ✅ 確認済み実装

#### 1. FEPGRUCell (src/core/fep_gru_cell.cpp)
```cpp
class FEPGRUCell : public torch::nn::Module {
    // 標準GRUパラメータ
    torch::nn::Linear input_to_hidden_, hidden_to_hidden_;
    torch::nn::Linear input_to_reset_, hidden_to_reset_;
    torch::nn::Linear input_to_update_, hidden_to_update_;
    
    // FEP特有レイヤー
    torch::nn::Linear prediction_head_, variance_head_;
    torch::nn::Linear meta_evaluation_head_;
    
    // 主要機能
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    forward(const torch::Tensor& input, const torch::Tensor& hidden);
    
    torch::Tensor compute_free_energy();
    void update_parameters_from_peer();
    torch::Tensor extract_som_features();
};
```

#### 2. FEPGRUNetwork (src/core/fep_gru_network.cpp)
```cpp
class FEPGRUNetwork : public torch::nn::Module {
    std::vector<std::shared_ptr<FEPGRUCell>> layers_;
    std::unordered_map<int, std::vector<torch::Tensor>> agent_states_;
    
    // 主要機能
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    forward(const torch::Tensor& sequence);
    
    void share_parameters_with_agents();
    torch::Tensor compute_collective_free_energy();
};
```

#### 3. PolarSpatialAttention (src/core/attention_evaluator.cpp)
```cpp
class PolarSpatialAttention : public torch::nn::Module {
    torch::nn::Conv2d distance_attention_, angle_attention_;
    torch::nn::Linear fusion_layer_;
    
    // 主要機能
    torch::Tensor forward(const torch::Tensor& polar_map);
    std::pair<torch::Tensor, torch::Tensor> compute_attention_weights();
};
```

#### 4. SPSAOptimizer (src/core/spsa_optimizer.cpp)
```cpp
class SPSAOptimizer {
    torch::Tensor parameter_history_, gradient_estimate_;
    
    // 主要機能
    torch::Tensor optimize(torch::Tensor& parameters,
                          std::function<double(const torch::Tensor&)> objective);
    
    torch::Tensor estimate_gradient(const torch::Tensor& parameters,
                                   std::function<double(const torch::Tensor&)> objective);
};
```

#### 5. ユーティリティ関数 (src/core/utils.cpp)
```cpp
namespace crlgru::utils {
    // 極座標変換
    torch::Tensor cartesian_to_polar_map(positions, self_position, 
                                        num_rings, num_sectors, max_range);
    
    // 相互情報量計算
    double compute_mutual_information(state1, state2);
    
    // 信頼度メトリック
    double compute_trust_metric(performance_history, distance, max_distance);
    
    // パラメータ保存・読み込み
    void save_parameters(filename, parameters);
    std::unordered_map<std::string, torch::Tensor> load_parameters(filename);
}
```

### 動作確認済み統計値

**テスト実行結果（2025年6月1日）**:
- **test_fep_gru_cell**: 基本機能・自由エネルギー計算・SOM機能 全通過
- **test_fep_gru_network**: 多層処理・エージェント管理・集合最適化 全通過
- **test_spsa_optimizer**: 勾配推定・パラメータ最適化・収束判定 全通過
- **test_utils**: 極座標変換・相互情報量・信頼度計算 全通過
- **test_fep_gru_integration**: 統合テスト・マルチエージェント・階層協調 全通過

**サンプルプログラム実行結果**:
- **simple_prediction_example**: 時系列データ10ステップ予測、予測誤差0.1以下
- **swarm_coordination_example**: 5エージェント群協調、凝集・分離・整列動作確認

**パフォーマンス測定**:
- **FEPGRUCell forward**: 64次元隠れ状態で0.5ms/ステップ
- **SPSA最適化**: 100次元パラメータ空間で50ステップ収束
- **極座標変換**: 100エージェント処理で0.1ms
- **階層的模倣**: 3レベル更新で1.2ms/エージェント

## 開発環境・依存関係（確認済み）

### 主要ライブラリ
- **LibTorch (PyTorch C++) 2.1.2**: `/Users/igarashi/local/libtorch` にインストール済み
- **CMake 3.18+**: ビルドシステム
- **C++17**: 言語標準
- **OpenMP** (オプション): 並列計算サポート

### ビルド確認済み環境
- **プラットフォーム**: macOS Apple Silicon (arm64)
- **コンパイラ**: AppleClang 17.0.0
- **ビルドタイプ**: Debug/Release両対応
- **LibTorch統合**: 有効

### コンパイル・実行手順（確認済み）
```bash
# プロジェクトディレクトリ
cd /Users/igarashi/local/project_workspace/crlGRU

# デバッグビルド
mkdir -p debug && cd debug
cmake .. -DCMAKE_BUILD_TYPE=Debug
make -j$(nproc)

# テスト実行
make test

# 個別テスト実行
./tests/test_fep_gru_cell
./tests/test_fep_gru_integration

# サンプルプログラム実行
./tests/examples/simple_prediction_example
./tests/examples/swarm_coordination_example

# インストール（$HOME/local/ へ）
make install
```

### ライブラリ使用例
```cpp
#include <crlgru/crl_gru.hpp>

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

## プロジェクト固有の開発パターン

### 1. FEPコンポーネント開発テンプレート
```cpp
// 標準的なFEPコンポーネント実装パターン
class NewFEPComponent : public torch::nn::Module {
public:
    struct Config {
        int input_size = 64;
        double free_energy_weight = 1.0;
        // デフォルト値付きパラメータ定義
    };

private:
    Config config_;
    torch::Tensor internal_state_;

public:
    explicit NewFEPComponent(const Config& config);
    
    // 必須メソッド
    std::tuple<torch::Tensor, torch::Tensor> forward(const torch::Tensor& input);
    torch::Tensor compute_free_energy() const;
    void reset_states();
    
    // 設定アクセサ
    const Config& get_config() const { return config_; }
};
```

### 2. 数式実装の標準パターン
```cpp
// 自由エネルギー計算の標準実装
torch::Tensor compute_free_energy(const torch::Tensor& prediction,
                                 const torch::Tensor& target,
                                 const torch::Tensor& variance) const {
    // 予測誤差: ε = target - prediction
    auto prediction_error = target - prediction;
    
    // 精度(逆分散): λ = 1/σ²
    auto precision = 1.0 / (variance + 1e-8);
    
    // 自由エネルギー: F = 0.5 * λ * ε² + 0.5 * log(2π/λ)
    auto free_energy = 0.5 * precision * prediction_error.pow(2) +
                      0.5 * torch::log(2 * M_PI / precision);
    
    return free_energy.mean();
}
```

### 3. エラーハンドリング・数値安定性
```cpp
// 標準的なエラーハンドリングパターン
try {
    auto result = gru_cell->forward(input, hidden);
} catch (const c10::Error& e) {
    std::cerr << "LibTorch error: " << e.what() << std::endl;
    return {};
}

// 前条件チェック
TORCH_CHECK(input.size(1) == config_.input_size, "Input size mismatch");
TORCH_CHECK(hidden.size(0) == input.size(0), "Batch size mismatch");

// 数値安定性確保
auto precision = 1.0 / (variance + 1e-8);  // ゼロ除算回避
auto normalized = input / (input.norm() + 1e-8);  // 正規化安定化
```

## 研究的新規性・貢献

### 学術的新規性
1. **自由エネルギー原理の工学実装**: 理論的厳密性と計算効率の両立
2. **階層的模倣学習**: パラメータ・ダイナミクス・意図の3レベル統合学習
3. **極座標空間注意**: 生物学的妥当性を持つスワームロボティクス空間認識
4. **予測的符号化GRU**: 時系列予測と自由エネルギー最小化の統合
5. **SOM統合**: 自己組織化マップによる内部状態の構造化学習

### 期待される応用
- **スワームロボティクス**: 分散協調制御・群集知能
- **マルチエージェント強化学習**: 自由エネルギー最小化による協調学習
- **時系列予測**: 予測的符号化による高精度予測モデル
- **認知アーキテクチャ**: 生物学的妥当性を持つ認知システム
- **最適化問題**: SPSA による勾配フリー最適化

---

## 対話の方針（2025年6月1日更新）

本プロンプトを読み込んだ上で、以下の方針で開発支援を行ってください：

1. **実際の構造に基づく開発**: include/crlgru/ と src/core/ の実際の構造を反映した開発
2. **テスト駆動開発**: tests/ ディレクトリのテストファイルを中心とした検証
3. **数学的厳密性**: 自由エネルギー原理・SPSA・階層的模倣学習の理論的正確性
4. **LibTorch活用**: PyTorch C++ APIの効率的使用とメモリ管理
5. **Header-only設計**: include/crlgru/crl_gru.hpp による統合API提供
6. **クロスプラットフォーム**: CMakeによる移植性確保
7. **実用的解決策**: 理論的正確性と実装可能性・計算効率のバランス
8. **Chain-of-Thought**: ステップバイステップの段階的開発アプローチ

**完成済み機能**を基盤として、**機能拡張**（新しいFEPコンポーネント、最適化アルゴリズム、評価指標）や**性能改善**（並列化、メモリ最適化、数値安定性）を段階的に進める方針です。数式を用いた説明の際は、LaTeX記法を使用し、C++実装との対応も明確にしてください。

**ステップバイステップ対話でお願いします。**
