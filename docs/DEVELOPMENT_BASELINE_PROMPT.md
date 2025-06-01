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

### ディレクトリ構造 ＜2025年6月1日現在・ハイブリッドアプローチ実装完了版＞
```
/Users/igarashi/local/project_workspace/crlGRU/        # メインプロジェクトディレクトリ
├── CMakeLists.txt                                     # 🔧 ハイブリッドアプローチ対応CMake設定
├── LICENSE, README.md                                 # プロジェクト基本情報
├── cmake/                                             # CMake関連設定
│   └── FindLibTorch.cmake                            # LibTorch検索スクリプト
├── config/                                            # 設定ファイル群
│   └── default_config.json                          # デフォルト設定
├── docs/                                              # 📚 ドキュメント集
│   ├── API_REFERENCE_JP.md                          # ✅ API詳細リファレンス
│   ├── DEVELOPMENT_BASELINE_PROMPT.md                # 🆕 開発ベースラインプロンプト（本ファイル）
│   ├── LIBTORCH_INSTALLATION_GUIDE.md               # LibTorchインストールガイド
│   ├── THEORETICAL_FOUNDATIONS.md                   # 📘 理論的基盤・数学的枠組み
│   └── USAGE_GUIDE_JP.md                            # 使用方法ガイド
├── include/crlgru/                                    # 📁 ハイブリッドヘッダー構成
│   ├── crl_gru.hpp                                  # ✅ ハイブリッド統合APIヘッダー
│   ├── common.hpp                                   # ✅ 共通定義・フォワード宣言
│   ├── core/                                        # 🔧 ライブラリコンポーネント宣言
│   │   └── fep_gru_cell.hpp                        # ✅ FEPGRUCell宣言ヘッダー
│   ├── optimizers/                                  # ✅ ヘッダーオンリー最適化器
│   │   └── spsa_optimizer.hpp                      # ✅ SPSAOptimizer（完全実装）
│   └── utils/                                       # ✅ ヘッダーオンリーユーティリティ
│       ├── config_types.hpp                        # ✅ 設定構造体（完全実装）
│       ├── math_utils.hpp                          # ✅ 数学関数（完全実装）
│       └── spatial_transforms.hpp                  # ✅ 空間変換（完全実装）
├── src/core/                                          # 💾 ライブラリ実装ファイル（簡素化済み）
│   ├── attention_evaluator.cpp                      # ✅ 空間注意・メタ評価実装
│   ├── fep_gru_cell.cpp                            # ✅ FEP-GRUセル実装
│   └── fep_gru_network.cpp                         # ✅ FEP-GRUネットワーク実装
├── tests/                                             # 🧪 テストプログラム群
│   ├── CMakeLists.txt                              # ✅ テストビルド設定
│   └── test_crlgru.cpp                             # ✅ 統合テストファイル
├── build/                                             # 🔧 メインビルドディレクトリ（Ninja）
├── build_test/                                        # 🔧 テスト用ビルドディレクトリ（Make）
│   ├── libcrlGRU.dylib                             # ✅ 軽量化ライブラリ
│   └── tests/test_crlgru                           # ✅ 統合テスト実行ファイル
├── debug/                                             # 🔧 デバッグビルドディレクトリ（CLion）
│   ├── libcrlGRU.dylib                             # ✅ デバッグ版ライブラリ
│   └── tests/test_fep_gru_cell                     # ✅ 個別テスト実行ファイル
└── tmp/                                               # 🧪 テスト・実験用ディレクトリ
    ├── backup/                                      # 🔒 完全バックアップ
    ├── test_header_only.cpp                       # ✅ ヘッダーオンリーテスト（11/11成功）
    ├── hybrid_implementation_report.md             # 📄 ハイブリッド実装レポート
    └── cleanup_completion_report.md                # 📄 不要ファイル削除完了レポート
```

**📚 詳細情報**: 
- API詳細リファレンス: [`docs/API_REFERENCE_JP.md`](docs/API_REFERENCE_JP.md)
- 理論的基盤・数学的枠組み: [`docs/THEORETICAL_FOUNDATIONS.md`](docs/THEORETICAL_FOUNDATIONS.md)
- 使用方法ガイド: [`docs/USAGE_GUIDE_JP.md`](docs/USAGE_GUIDE_JP.md)
- LibTorchインストール: [`docs/LIBTORCH_INSTALLATION_GUIDE.md`](docs/LIBTORCH_INSTALLATION_GUIDE.md)

### ✅ ハイブリッドアプローチ実装完了（2025年6月1日）

#### **ヘッダーオンリーコンポーネント（完全動作確認済み）**
- **SPSAOptimizer** (`include/crlgru/optimizers/spsa_optimizer.hpp`)
  - テンプレート化（float/double両対応）
  - モメンタム、制約付き最適化、勾配平滑化機能
  - **テスト結果**: 3/3テスト成功
  
- **Math Utils** (`include/crlgru/utils/math_utils.hpp`)
  - 数値安定化関数（safe_normalize, stable_softmax）
  - テンプレート化テンソル統計関数
  - **テスト結果**: 2/2テスト成功
  
- **Spatial Transforms** (`include/crlgru/utils/spatial_transforms.hpp`)
  - 効率的極座標変換（ベクトル化演算）
  - 回転変換、最近傍探索、局所座標変換
  - **テスト結果**: 2/2テスト成功
  
- **Config Types** (`include/crlgru/utils/config_types.hpp`)
  - 型安全な設定構造体群
  - デフォルト値とバリデーション機能
  - **テスト結果**: 4/4テスト成功

**ヘッダーオンリー統合テスト**: ✅ **11/11テスト成功（100%成功率）**

#### **ライブラリコンポーネント（実装継続中）**
- **FEPGRUCell** (`src/core/fep_gru_cell.cpp`)
- **FEPGRUNetwork** (`src/core/fep_gru_network.cpp`) 
- **PolarSpatialAttention** (`src/core/attention_evaluator.cpp`)

#### **削除済み不要ファイル**
- ~~`src/core/spsa_optimizer.cpp`~~ → ヘッダーオンリー化により削除
- ~~`src/core/utils.cpp`~~ → ヘッダーオンリー化により削除

### 実装・動作確認済みコンポーネント

#### ✅ 動作確認済み：FEP-GRU基本機能
- **FEPGRUCell**: 自由エネルギー計算、予測符号化、SOM特徴抽出、階層的模倣学習
- **FEPGRUNetwork**: 多層処理、エージェント間パラメータ共有、集合自由エネルギー最小化
- **PolarSpatialAttention**: 極座標空間注意機構、距離・角度別適応的重み
- **MetaEvaluator**: 多目的評価、適応的重み調整、目標・衝突・凝集・整列統合

#### ✅ 統合テスト結果（2025年6月1日）
- **統合テストファイル**: `tests/test_crlgru.cpp` 
- **ヘッダーオンリーテスト**: `tmp/test_header_only.cpp` → **11/11テスト成功**
- **ビルドシステム**: CMake + LibTorch統合、ハイブリッドアプローチ対応
- **実行確認**: 複数のビルド環境で動作検証済み

#### ✅ 完全実装済み：数学的アルゴリズム
- **自由エネルギー計算**: 解析的KL発散、ガウス分布仮定での効率実装
- **予測的符号化**: 自己回帰的未来状態予測、予測誤差フィードバック
- **階層的模倣学習**: 3レベル（パラメータ・ダイナミクス・意図）の統合模倣
- **極座標変換**: カルテシアン→極座標マップ、リング・セクター分割（ベクトル化最適化）
- **SPSA勾配推定**: 同時摂動による勾配フリー最適化、ベルヌーイ摂動
- **SOM特徴抽出**: 自己組織化マップによる内部状態クラスタリング

## ハイブリッドアプローチの技術的利点

### 🚀 パフォーマンス最適化
1. **コンパイル時間短縮**: ヘッダーオンリー部分の変更時はライブラリ再ビルド不要
2. **インライン展開**: 数学関数・空間変換の高速化
3. **テンプレート最適化**: float/double型特化による数値精度制御

### 🔗 依存関係の柔軟性  
1. **軽量アプリケーション**: SPSA最適化のみ使用時は`.dylib`不要
2. **段階的導入**: 必要な機能のみ選択的に使用可能
3. **クロスプラットフォーム**: ヘッダーオンリー部分は環境依存なし

### 🧪 開発効率向上
1. **機能別分離**: utils/, optimizers/, core/ による明確な責任分離
2. **独立テスト**: ヘッダーオンリー部分の単体テスト可能
3. **保守性向上**: 機能追加時の影響範囲限定

### 📦 配布・利用の簡便性
1. **ヘッダーオンリー**: コピー&ペーストで即座に利用可能
2. **段階的導入**: 基本機能→高度な機能への段階的移行サポート
3. **バイナリ互換性**: ライブラリ部分のABI安定化

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

### ✅ ハイブリッドアプローチの実装パターン

#### 1. ヘッダーオンリーコンポーネント実装パターン
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

#### 2. ライブラリコンポーネント実装パターン
```cpp
// ✅ 実装例: FEPGRUCell 
// 宣言: include/crlgru/core/fep_gru_cell.hpp
class FEPGRUCell : public torch::nn::Module {
public:
    using Config = FEPGRUCellConfig;
    
    explicit FEPGRUCell(const Config& config);
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(
        const torch::Tensor& input, const torch::Tensor& hidden);
    // ... 宣言のみ
};

// 実装: src/core/fep_gru_cell.cpp
// 複雑なニューラルネットワーク実装
```

#### 3. 統合ヘッダーパターン
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

// 必要に応じてLibTorchリンク
#ifdef CRLGRU_HAS_TORCH
// ライブラリ機能利用時
#endif
```

### ✅ 動作確認済み使用例

#### SPSAOptimizer（ヘッダーオンリー）
```cpp
#include <crlgru/optimizers/spsa_optimizer.hpp>

// float版とdouble版の作成
auto params = torch::randn({10}, torch::requires_grad(true));
std::vector<torch::Tensor> param_list = {params};

auto optimizer_f = crlgru::make_spsa_optimizer<float>(param_list);
auto optimizer_d = crlgru::make_spsa_optimizer<double>(param_list);

// 目的関数の最適化
auto objective = [&]() -> float {
    auto target = torch::ones_like(params);
    return (params - target).pow(2).sum().item<float>();
};

optimizer_f->step(objective, 1);
```

#### 空間変換（ヘッダーオンリー）
```cpp
#include <crlgru/utils/spatial_transforms.hpp>

// 効率的な極座標変換
auto positions = torch::tensor({{{1.0, 1.0}, {-1.0, -1.0}}});
auto self_pos = torch::tensor({{0.0, 0.0}});

auto polar_map = crlgru::utils::cartesian_to_polar_map(
    positions, self_pos, 4, 8, 10.0
);
// 結果: [1, 4, 8] テンソル
```

#### 数学ユーティリティ（ヘッダーオンリー）
```cpp
#include <crlgru/utils/math_utils.hpp>

// 数値安定化関数
auto tensor = torch::tensor({3.0, 4.0});
auto normalized = crlgru::utils::safe_normalize(tensor);

auto logits = torch::tensor({1.0, 2.0, 3.0});
auto softmax = crlgru::utils::stable_softmax(logits, 0);
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

### コンパイル・実行手順（確認済み）
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
mkdir -p build_test && cd build_test
cmake .. -DCMAKE_BUILD_TYPE=Debug
make -j$(nproc)

# 統合テスト実行
./tests/test_crlgru

# インストール（$HOME/local/ へ）
make install
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
// 完全機能使用（.dylibリンク必要）
#include <crlgru/crl_gru.hpp>

int main() {
    // FEP-GRU設定
    crlgru::FEPGRUCellConfig config;
    config.input_size = 10;
    config.hidden_size = 64;
    
    // FEP-GRUセル作成（ライブラリ機能）
    auto gru_cell = std::make_shared<crlgru::FEPGRUCell>(config);
    
    // ヘッダーオンリー機能も同時使用可能
    auto optimizer = crlgru::make_spsa_optimizer<double>({});
    
    return 0;
}
```

## プロジェクト固有の開発パターン

### 1. ハイブリッドコンポーネント開発方針
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
    };
}
```

### 2. 数式実装の標準パターン
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

### 3. エラーハンドリング・数値安定性
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

// 前条件チェック（ヘッダーオンリー対応）
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

1. **ハイブリッドアプローチ基盤**: ヘッダーオンリーとライブラリの最適な組み合わせによる開発
2. **実際の構造に基づく開発**: include/crlgru/ の階層構造を反映した開発
3. **動作確認済み基盤活用**: ヘッダーオンリー部分（11/11テスト成功）を積極活用
4. **数学的厳密性**: 自由エネルギー原理・SPSA・階層的模倣学習の理論的正確性
5. **LibTorch活用**: PyTorch C++ APIの効率的使用とメモリ管理
6. **段階的機能拡張**: ヘッダーオンリー→ライブラリ→統合の段階的アプローチ
7. **クロスプラットフォーム**: CMakeによる移植性確保
8. **実用的解決策**: 理論的正確性と実装可能性・計算効率のバランス
9. **Chain-of-Thought**: ステップバイステップの段階的開発アプローチ

**完成済みハイブリッド基盤**を活用して、**機能拡張**（新しいヘッダーオンリーコンポーネント、ライブラリコンポーネント拡張）や**性能改善**（並列化、メモリ最適化、数値安定性）を段階的に進める方針です。数式を用いた説明の際は、LaTeX記法を使用し、C++実装との対応も明確にしてください。

**ステップバイステップ対話でお願いします。**
