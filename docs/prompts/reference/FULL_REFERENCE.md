現在 /Users/igarashi/local/project_workspace/crlGRU のプロジェクトを開発中です。このプロジェクトの研究概要は以下のとおりです。

# crlGRU: Free Energy Principle GRU Library 開発ベースラインプロンプト

## プロジェクト概要

あなたは、自由エネルギー原理（Free Energy Principle）に基づくGRUニューラルネットワークライブラリ「crlGRU」の研究開発を支援するAIアシスタントです。このプロジェクトは、マルチエージェントシステムとスワーム知能研究に特化した革新的な機械学習フレームワークの実現を目指しています。

### 開発方針・制約

- **Hybrid Approach**: ✅**実装完了** - ユーティリティ・最適化器はヘッダーオンリー、複雑なニューラルネットワークはライブラリとするハイブリッド構成を採用。コンパイル効率と依存関係柔軟性を両立。
- **Header-only Components**: ヘッダーオンリーコンポーネントは *.hpp として完全実装を含む。ライブラリコンポーネントは宣言ヘッダー *.hpp と実装ファイル *.cpp に分離。
- **Doxygen形式コメント**: C++プログラムにおいて，"///" を用いたdoxygen形式のコメントを使用し、関数やクラスの説明を明確に記述してください
- **ファイル修正原則**: 現在のファイルの修正を原則としますが，大きな変更の場合は，現在の対象ファイルのバックアップに tmp/ 以下に _backup を付して，同名のファイルで更新作業をしてください。
- **テスト・実験用ファイル配置**: AIアシスタントが処理で試験的にテストプログラムやファイルを作成する場合は、必ず `tmp/` ディレクトリ以下に配置してください。正式なプロジェクトファイルと実験用ファイルを明確に分離します
- **Chain-of-Thought**: CoTでステップバイステップで開発を進めてください
- **検証対象**: tests/test_crlgru.cpp が正常にビルド・実行できることを確認してください
- **テスト駆動開発**: 統合テストによる品質保証
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

### ディレクトリ構造 ＜2025年6月1日現在・ビルドエラー修正版＞
```
/Users/igarashi/local/project_workspace/crlGRU/        # メインプロジェクトディレクトリ
├── CMakeLists.txt                                     # 🔧 ハイブリッドアプローチ対応CMake設定
├── LICENSE, README.md                                 # プロジェクト基本情報
├── cmake/                                             # CMake関連設定
│   └── FindLibTorch.cmake                            # LibTorch検索スクリプト
├── config/                                            # 設定ファイル群
│   └── default_config.json                          # デフォルト設定
├── docs/                                              # 📚 ドキュメント集
│   ├── guides/                                       # ガイド集
│   │   ├── API_REFERENCE.md                         # ✅ API詳細リファレンス
│   │   ├── installation/
│   │   │   └── LIBTORCH_SETUP.md                   # LibTorchセットアップガイド
│   │   └── usage/
│   │       └── BASIC_USAGE.md                       # 基本使用方法
│   ├── prompts/                                      # プロンプト集
│   │   ├── compact/
│   │   │   └── BASELINE_COMPACT.md                 # 🆕 簡潔版ベースライン
│   │   ├── core/
│   │   │   └── CURRENT_TASK.md                     # 現在のタスク
│   │   ├── reference/
│   │   │   ├── DEVELOPMENT_BASELINE_PROMPT.md       # 🆕 開発ベースラインプロンプト
│   │   │   └── FULL_REFERENCE.md                   # 本ファイル
│   │   └── specialized/
│   │       ├── build_fix.md                        # ビルド修正用
│   │       ├── header_only_dev.md                  # ヘッダーオンリー開発用
│   │       └── testing.md                          # テスト用
│   └── theory/
│       └── THEORETICAL_FOUNDATIONS.md               # 📘 理論的基盤・数学的枠組み
├── include/crlgru/                                    # 📁 ハイブリッドヘッダー構成
│   ├── crl_gru.hpp                                  # ✅ ハイブリッド統合APIヘッダー
│   ├── common.hpp                                   # ✅ 共通定義・フォワード宣言
│   ├── core/                                        # 🔧 ライブラリコンポーネント宣言
│   │   ├── fep_gru_cell.hpp                        # ✅ FEPGRUCell宣言ヘッダー
│   │   ├── fep_gru_network.hpp                     # ✅ FEPGRUNetwork宣言ヘッダー（新規追加）
│   │   └── polar_spatial_attention.hpp             # ✅ PolarSpatialAttention・MetaEvaluator宣言（新規追加）
│   ├── optimizers/                                  # ✅ ヘッダーオンリー最適化器
│   │   └── spsa_optimizer.hpp                      # ✅ SPSAOptimizer（完全実装）
│   └── utils/                                       # ✅ ヘッダーオンリーユーティリティ
│       ├── config_types.hpp                        # ✅ 設定構造体（完全実装・拡張済み）
│       ├── math_utils.hpp                          # ✅ 数学関数（完全実装・警告修正）
│       └── spatial_transforms.hpp                  # ✅ 空間変換（完全実装）
├── src/core/                                          # 💾 ライブラリ実装ファイル（修正中）
│   ├── attention_evaluator.cpp                      # 🔄 空間注意・メタ評価実装（エラー修正中）
│   ├── fep_gru_cell.cpp                            # ✅ FEP-GRUセル実装
│   └── fep_gru_network.cpp                         # 🔄 FEP-GRUネットワーク実装（部分修正済み）
├── tests/                                             # 🧪 テストプログラム群
│   ├── CMakeLists.txt                              # ✅ テストビルド設定
│   └── test_crlgru.cpp                             # ✅ 統合テストファイル（修正完了）
├── build/                                             # 🔧 メインビルドディレクトリ（Ninja）
├── build_test/                                        # 🔧 テスト用ビルドディレクトリ（Make）
├── build_test_new/                                    # 🔧 新規ビルドディレクトリ（エラー修正用）
├── debug/                                             # 🔧 デバッグビルドディレクトリ（CLion）
└── tmp/                                               # 🧪 テスト・実験用ディレクトリ
    ├── backup/                                      # 🔒 完全バックアップ
    ├── test_header_only.cpp                       # ✅ ヘッダーオンリーテスト（11/11成功）
    ├── test_crlgru_fixed.cpp                      # ✅ 修正版統合テスト
    ├── test_crlgru_backup.cpp                     # 🔒 元統合テストバックアップ
    ├── build_fix_progress.sh                      # 📊 修正進捗レポート
    ├── hybrid_implementation_report.md             # 📄 ハイブリッド実装レポート
    └── cleanup_completion_report.md                # 📄 不要ファイル削除完了レポート
```

**📚 詳細情報**: 
- API詳細リファレンス: [`docs/guides/API_REFERENCE.md`](docs/guides/API_REFERENCE.md)
- 理論的基盤・数学的枠組み: [`docs/theory/THEORETICAL_FOUNDATIONS.md`](docs/theory/THEORETICAL_FOUNDATIONS.md)
- 基本使用方法: [`docs/guides/usage/BASIC_USAGE.md`](docs/guides/usage/BASIC_USAGE.md)
- LibTorchセットアップ: [`docs/guides/installation/LIBTORCH_SETUP.md`](docs/guides/installation/LIBTORCH_SETUP.md)

### ✅ ハイブリッドアプローチ実装状況（2025年6月1日更新）

#### **ヘッダーオンリーコンポーネント（完全動作確認済み）**
- **SPSAOptimizer** (`include/crlgru/optimizers/spsa_optimizer.hpp`)
  - テンプレート化（float/double両対応）
  - モメンタム、制約付き最適化、勾配平滑化機能
  - **テスト結果**: 3/3テスト成功
  
- **Math Utils** (`include/crlgru/utils/math_utils.hpp`)
  - 数値安定化関数（safe_normalize, stable_softmax）
  - テンプレート化テンソル統計関数
  - **修正完了**: 未使用パラメータ警告解決
  - **テスト結果**: 2/2テスト成功
  
- **Spatial Transforms** (`include/crlgru/utils/spatial_transforms.hpp`)
  - 効率的極座標変換（ベクトル化演算）
  - 回転変換、最近傍探索、局所座標変換
  - **テスト結果**: 2/2テスト成功
  
- **Config Types** (`include/crlgru/utils/config_types.hpp`)
  - 型安全な設定構造体群
  - デフォルト値とバリデーション機能
  - **拡張完了**: FEPGRUNetworkConfig, PolarSpatialAttentionConfig, MetaEvaluatorConfig追加
  - **テスト結果**: 4/4テスト成功

**ヘッダーオンリー統合テスト**: ✅ **11/11テスト成功（100%成功率）**

#### **ライブラリコンポーネント（ビルド修正中）**

**✅ 完全修正済み:**
- **FEPGRUCell** (`src/core/fep_gru_cell.cpp`)
  - ヘッダー宣言完成（`include/crlgru/core/fep_gru_cell.hpp`）
  - torch::nn::Linear初期化問題解決（{nullptr}）
  - peer_parameters_メンバー変数追加
  - get_free_energy()関数追加

**🔄 部分修正済み:**
- **FEPGRUNetwork** (`src/core/fep_gru_network.cpp`)
  - ヘッダー宣言完成（`include/crlgru/core/fep_gru_network.hpp`）
  - torch::nn::Dropout初期化問題解決
  - 設定パラメータ名修正（layer_dropout → dropout_rate）
  - エージェント管理変数名エイリアス追加

**❌ 修正待ち:**
- **PolarSpatialAttention** (`src/core/attention_evaluator.cpp`)
  - ヘッダー宣言完成（`include/crlgru/core/polar_spatial_attention.hpp`）
  - 実装ファイルの未定義メンバー変数エラー残存
  - 設定パラメータ不足（attention_dim, attention_dropout等）

#### **テストファイル（修正完了）**
- **統合テストファイル** (`tests/test_crlgru.cpp`)
  - SPSAOptimizer<double>テンプレート引数修正
  - 設定構造体名修正
  - ヘッダーオンリー機能テスト追加
  - 段階的テスト構造の実装

#### **削除済み不要ファイル**
- ~~`src/core/spsa_optimizer.cpp`~~ → ヘッダーオンリー化により削除
- ~~`src/core/utils.cpp`~~ → ヘッダーオンリー化により削除

### 🔧 ビルドエラー解決状況（2025年6月1日）

#### **✅ 解決済みエラー**
1. **不完全型エラー**: FEPGRUNetwork, PolarSpatialAttention, MetaEvaluatorのヘッダー作成
2. **テンプレート引数エラー**: SPSAOptimizer<double>明示的テンプレート化
3. **未定義設定構造体**: NetworkConfig, AttentionConfig, EvaluationConfig追加
4. **Linear初期化エラー**: torch::nn::Linear{nullptr}による明示的初期化
5. **設定パラメータ不一致**: layer_dropout → dropout_rate修正
6. **テストファイル型エラー**: 全テンプレート引数とメンバー変数修正

#### **❌ 残存エラー**
1. **attention_evaluator.cpp**: 
   - 未定義メンバー変数（distance_attention_, angle_attention_, fusion_layer_）
   - 不足設定パラメータ（attention_dim, attention_dropout）
   - ヘッダーと実装の不整合

2. **fep_gru_network.cpp**: 
   - エージェント管理変数名の不整合（部分解決済み）

#### **🎯 次回修正項目**
1. PolarSpatialAttentionヘッダーのメンバー変数追加
2. PolarSpatialAttentionConfig設定パラメータ拡張
3. attention_evaluator.cpp実装修正
4. ビルド成功確認とテスト実行

### 実装状況と技術的詳細

#### ✅ ハイブリッドアプローチの実装パターン（確立済み）

**1. ヘッダーオンリーコンポーネント実装パターン**
```cpp
// ✅ 実装例: SPSAOptimizer (include/crlgru/optimizers/spsa_optimizer.hpp)
template<typename FloatType = double>
class SPSAOptimizer {
public:
    struct Config {
        FloatType a = 0.16;        // ステップサイズ係数
        FloatType c = 0.16;        // 摂動サイズ係数
        // ... 設定パラメータ
    };

    explicit SPSAOptimizer(const std::vector<torch::Tensor>& parameters,
                          const Config& config = Config{});
    
    void step(std::function<FloatType()> objective_function, int iteration = -1);
    FloatType optimize(std::function<FloatType()> objective_function);
    
    // 完全実装をヘッダー内に記述
};
```

**2. ライブラリコンポーネント実装パターン**
```cpp
// ✅ 実装例: FEPGRUCell 
// 宣言: include/crlgru/core/fep_gru_cell.hpp
class FEPGRUCell : public torch::nn::Module {
public:
    using Config = FEPGRUCellConfig;
    
    explicit FEPGRUCell(const Config& config);
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(
        const torch::Tensor& input, const torch::Tensor& hidden);
        
private:
    // ✅ 修正済み: torch::nn::Linear初期化
    torch::nn::Linear input_to_hidden_{nullptr};
    torch::nn::Linear hidden_to_hidden_{nullptr};
    // ... 他のメンバー変数
};

// 実装: src/core/fep_gru_cell.cpp
// 複雑なニューラルネットワーク実装
```

**3. 統合ヘッダーパターン**
```cpp
// ✅ ハイブリッド統合: include/crlgru/crl_gru.hpp
#include <crlgru/common.hpp>

// ヘッダーオンリーコンポーネント（完全実装）
#include <crlgru/utils/config_types.hpp>
#include <crlgru/utils/math_utils.hpp>
#include <crlgru/utils/spatial_transforms.hpp>
#include <crlgru/optimizers/spsa_optimizer.hpp>

// ライブラリコンポーネント（宣言のみ）
#include <crlgru/core/fep_gru_cell.hpp>
#include <crlgru/core/fep_gru_network.hpp>
#include <crlgru/core/polar_spatial_attention.hpp>
```

### ✅ 動作確認済み使用例

#### SPSAOptimizer（ヘッダーオンリー）
```cpp
#include <crlgru/optimizers/spsa_optimizer.hpp>

// float版とdouble版の作成
auto params = torch::randn({10}, torch::requires_grad(true));
std::vector<torch::Tensor> param_list = {params};

// ✅ 修正済み: テンプレート引数明示
auto optimizer_d = std::make_shared<crlgru::SPSAOptimizer<double>>(param_list);
auto optimizer_f = std::make_shared<crlgru::SPSAOptimizer<float>>(param_list);

// 目的関数の最適化
auto objective = []() -> double {
    return 1.0; // 簡単な目的関数
};

optimizer_d->step(objective, 1);
```

#### 設定構造体（拡張済み）
```cpp
#include <crlgru/utils/config_types.hpp>

// ✅ 拡張済み: FEPGRUNetwork設定
crlgru::FEPGRUNetworkConfig network_config;
network_config.layer_sizes = {64, 128, 64};
network_config.dropout_rate = 0.1;

// ✅ 新規追加: PolarSpatialAttention設定
crlgru::PolarSpatialAttentionConfig attention_config;
attention_config.input_channels = 64;
attention_config.num_distance_rings = 8;
attention_config.num_angle_sectors = 16;

// ✅ 新規追加: MetaEvaluator設定
crlgru::MetaEvaluatorConfig eval_config;
eval_config.metrics = {"prediction_accuracy", "free_energy"};
eval_config.adaptive_weights = true;
```

## 開発環境・依存関係（確認済み）

### 主要ライブラリ
- **LibTorch (PyTorch C++) 2.1.2**: `/Users/igarashi/local/libtorch` にインストール済み
- **CMake 3.18+**: ハイブリッドアプローチ対応ビルドシステム
- **C++17**: 言語標準
- **OpenMP** (オプション): 並列計算サポート

### ビルド確認済み環境
- **プラットフォーム**: macOS Apple Silicon (arm64)
- **コンパイラ**: AppleClang 17.0.0
- **ビルドタイプ**: Debug/Release両対応
- **LibTorch統合**: 有効

### コンパイル・実行手順（更新版）
```bash
# プロジェクトディレクトリ
cd /Users/igarashi/local/project_workspace/crlGRU

# ヘッダーオンリーテスト（LibTorchリンク）
export LIBTORCH_PATH="/Users/igarashi/local/libtorch"
g++ -std=c++17 -I./include -I$LIBTORCH_PATH/include \
    -o tmp/test_header_only tmp/test_header_only.cpp \
    -L$LIBTORCH_PATH/lib -ltorch -ltorch_cpu -lc10 \
    -Wl,-rpath,$LIBTORCH_PATH/lib

# テスト用ビルド（ライブラリ含む）
mkdir -p build_test_new && cd build_test_new
cmake .. -DCMAKE_BUILD_TYPE=Debug
make -j$(nproc)

# 統合テスト実行（ビルド成功後）
./tests/test_crlgru
```

### ハイブリッド使用例（軽量版）
```cpp
// ヘッダーオンリーのみ使用（.dylibリンク不要）
#include <crlgru/utils/spatial_transforms.hpp>
#include <crlgru/optimizers/spsa_optimizer.hpp>

int main() {
    // 空間変換とSPSA最適化のみ使用
    // ライブラリリンク不要
    return 0;
}
```

### ハイブリッド使用例（完全版）
```cpp
// 完全機能使用（.dylibリンク必要・ビルド修正後）
#include <crlgru/crl_gru.hpp>

int main() {
    // FEP-GRU設定
    crlgru::FEPGRUCellConfig config;
    config.input_size = 10;
    config.hidden_size = 64;
    
    // FEP-GRUセル作成（ライブラリ機能）
    auto gru_cell = std::make_shared<crlgru::FEPGRUCell>(config);
    
    // ヘッダーオンリー機能も同時使用可能
    auto params = torch::randn({10}, torch::requires_grad(true));
    std::vector<torch::Tensor> param_list = {params};
    auto optimizer = std::make_shared<crlgru::SPSAOptimizer<double>>(param_list);
    
    return 0;
}
```

## プロジェクト固有の開発パターン

### 1. Chain-of-Thoughtビルドエラー修正パターン
```markdown
## Step 1: エラー分析
- コンパイルエラーの分類（不完全型、未定義識別子、テンプレート等）
- 影響範囲の特定（ヘッダー vs 実装）

## Step 2: ヘッダー構造修正
- 不足ヘッダーファイルの作成
- フォワード宣言の追加
- 設定構造体の拡張

## Step 3: 実装ファイル修正
- ヘッダーとの整合性確保
- メンバー変数・関数の追加
- 初期化問題の解決

## Step 4: テストファイル修正
- 型エイリアスの修正
- テンプレート引数の明示
- テスト構造の更新

## Step 5: ビルド確認
- 段階的ビルド実行
- エラー修正の確認
- 統合テスト実行
```

### 2. ハイブリッドコンポーネント開発方針
```cpp
// 軽量・汎用的機能 → ヘッダーオンリー
namespace crlgru::utils {
    template<typename T>
    inline torch::Tensor efficient_function(const T& input) {
        // 完全実装をヘッダー内に記述
        return result;
    }
}

// 複雑・特化的機能 → ライブラリコンポーネント
namespace crlgru {
    class ComplexNeuralNetwork : public torch::nn::Module {
        // 宣言のみヘッダーに、実装は.cppファイル
        // ✅ メンバー変数初期化: torch::nn::Linear layer_{nullptr};
    };
}
```

### 3. 数式実装の標準パターン
```cpp
// ✅ 実装例: 自由エネルギー計算（ヘッダーオンリー向け）
template<typename FloatType>
inline torch::Tensor compute_free_energy_optimized(
    const torch::Tensor& prediction,
    const torch::Tensor& target,
    const torch::Tensor& variance = {}) {
    
    // 予測誤差: ε = target - prediction
    auto prediction_error = target - prediction;
    
    // 精度(逆分散): λ = 1/σ²
    auto precision = 1.0 / (variance.defined() ? variance + 1e-8 : 
                           torch::ones_like(prediction_error));
    
    // 自由エネルギー: F = 0.5 * λ * ε² + 0.5 * log(2π/λ)
    auto free_energy = 0.5 * precision * prediction_error.pow(2) +
                      0.5 * torch::log(2 * M_PI / precision);
    
    return free_energy.mean();
}
```

### 4. エラーハンドリング・数値安定性
```cpp
// ハイブリッド環境でのエラーハンドリング
try {
    // ヘッダーオンリー機能
    auto result1 = crlgru::utils::safe_normalize(tensor);
    
    // ライブラリ機能（LibTorchリンク必要）
    auto result2 = gru_cell->forward(input, hidden);
    
} catch (const c10::Error& e) {
    std::cerr << "LibTorch error: " << e.what() << std::endl;
} catch (const std::exception& e) {
    std::cerr << "Standard error: " << e.what() << std::endl;
}

// 前条件チェック（ハイブリッド対応）
#ifdef CRLGRU_HAS_TORCH
    TORCH_CHECK(input.size(1) == expected_size, "Size mismatch");
#else
    if (input.size(1) != expected_size) {
        throw std::invalid_argument("Size mismatch");
    }
#endif
```

## 研究的新規性・貢献

### 学術的新規性
1. **ハイブリッドアーキテクチャ**: ヘッダーオンリーとライブラリの最適な組み合わせ
2. **自由エネルギー原理の工学実装**: 理論的厳密性と計算効率の両立
3. **階層的模倣学習**: パラメータ・ダイナミクス・意図の3レベル統合学習
4. **極座標空間注意**: 生物学的妥当性を持つスワームロボティクス空間認識
5. **予測的符号化GRU**: 時系列予測と自由エネルギー最小化の統合

### 期待される応用
- **スワームロボティクス**: 分散協調制御・群集知能
- **マルチエージェント強化学習**: 自由エネルギー最小化による協調学習
- **時系列予測**: 予測的符号化による高精度予測モデル
- **認知アーキテクチャ**: 生物学的妥当性を持つ認知システム
- **最適化問題**: SPSA による勾配フリー最適化

## 対話の方針（2025年6月1日更新）

本プロンプトを読み込んだ上で、以下の方針で開発支援を行ってください：

1. **Chain-of-Thoughtビルドエラー修正**: エラー分析→ヘッダー修正→実装修正→テスト修正→確認のステップ化
2. **実際の構造に基づく開発**: include/crlgru/ の階層構造を反映した開発
3. **動作確認済み基盤活用**: ヘッダーオンリー部分（11/11テスト成功）を積極活用
4. **段階的ビルド修正**: 複数ファイルの同時修正ではなく、一つずつ確実に修正
5. **LibTorch活用**: PyTorch C++ APIの効率的使用とメモリ管理
6. **ハイブリッドアプローチ**: ヘッダーオンリー→ライブラリ→統合の段階的アプローチ
7. **クロスプラットフォーム**: CMakeによる移植性確保
8. **実用的解決策**: 理論的正確性と実装可能性・計算効率のバランス
9. **数学的厳密性**: 自由エネルギー原理・SPSA・階層的模倣学習の理論的正確性

**現在の最優先課題**: attention_evaluator.cppのビルドエラー修正により、完全なライブラリビルド成功を実現する。**完成済みハイブリッド基盤**を活用して、残存するビルドエラーを段階的に解決し、**統合テスト実行**を目指します。数式を用いた説明の際は、LaTeX記法を使用し、C++実装との対応も明確にしてください。

**ステップバイステップ対話でお願いします。**
