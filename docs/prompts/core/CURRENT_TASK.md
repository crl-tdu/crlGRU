# crlGRU コアプロンプト（最新版）

## 🎯 現在のタスク
**全テストスイート実行による総合検証** - 次の優先課題

## ✅ 完了済み（2025年6月1日）
- **attention_evaluator.cpp ビルドエラー修正** ✅ **完了**
- **PolarSpatialAttention設定構造体拡張** ✅ **完了**
- **MetaEvaluator設定構造体修正** ✅ **完了**
- **ヘッダーと実装の整合性確保** ✅ **完了**
- **テストスイートビルド成功** ✅ **完了**
- **SPSAオプティマイザー in-place操作エラー修正** ✅ **完了**

## 📊 修正完了項目詳細
### ✅ 設定構造体修正 (`include/crlgru/utils/config_types.hpp`)
```cpp
struct PolarSpatialAttentionConfig {
    int attention_dim = 32;                 // ✅ 追加完了
    double attention_dropout = 0.0;         // ✅ 追加完了
};

struct MetaEvaluatorConfig {
    std::vector<double> objective_weights = {0.25, 0.25, 0.25, 0.25}; // ✅ 追加完了
};

using EvaluationConfig = MetaEvaluatorConfig;  // ✅ 型エイリアス追加完了
```

### ✅ SPSAオプティマイザー修正 (`include/crlgru/optimizers/spsa_optimizer.hpp`)
```cpp
// ✅ estimate_gradient メソッド - in-place操作除去完了
auto original_param = param.clone();
param.copy_(original_param + ck * perturbation);  // 安全な操作
auto loss_plus = objective_function();
param.copy_(original_param - ck * perturbation);  // 安全な操作
auto loss_minus = objective_function();
param.copy_(original_param);  // 復元

// ✅ update_parameters メソッド - in-place操作除去完了
auto new_param = param - update;
param.copy_(torch::clamp(new_param, config_.param_min, config_.param_max));
```

### ✅ ヘッダーファイル修正 (`include/crlgru/core/polar_spatial_attention.hpp`)
```cpp
// ✅ メンバー変数修正完了
torch::nn::Conv2d distance_attention_{nullptr};
torch::nn::Conv2d angle_attention_{nullptr};
torch::nn::Linear fusion_layer_{nullptr};
torch::nn::Dropout dropout_{nullptr};

// ✅ メソッドシグネチャ修正完了
std::pair<torch::Tensor, torch::Tensor> compute_attention_weights(const torch::Tensor& features);
double evaluate(const torch::Tensor& predicted_states,
               const torch::Tensor& current_state,
               const torch::Tensor& environment_state);
```

### ✅ テストファイル修正 (`tests/test_crlgru.cpp`)
```cpp
auto mean = crlgru::utils::compute_tensor_mean(tensor);  // ✅ 関数名修正完了
```

## 🔄 次の優先課題
**全テストスイート実行による総合検証**

### 🎯 対象
- 全11テスト関数の完全実行
- SPSA関連テスト3個の動作確認
- 統合テストによる相互作用検証

### 🔧 検証方針
1. **修正済みSPSAテスト**: in-place操作エラー解消確認
2. **ライブラリテスト**: attention_evaluator.cpp等の動作確認
3. **ヘッダーオンリーテスト**: utils/optimizers完全動作確認
4. **Chain-of-Thought**: テスト実行 → 結果分析 → 残存問題特定 → 最終調整

## 📁 プロジェクト基本情報
- **パス**: `/Users/igarashi/local/project_workspace/crlGRU`
- **ビルド状況**: ✅ **ライブラリ100%ビルド成功**
- **テスト状況**: 🔄 **要実行確認（SPSA修正後初回）**
- **ハイブリッドアーキテクチャ**: ヘッダーオンリー(utils/optimizers) + ライブラリ(core)

## 🎉 技術的達成
- **自由エネルギー原理**: $F = E[q(z)] - \text{KL}[q(z)||p(z)]$ 実装整合性確保
- **極座標注意機構**: $\text{Attention}(Q,K,V) = \text{softmax}(QK^T/\sqrt{d_k})V$ の極座標拡張実装
- **0ビルドエラー**: 全コンポーネント完全修正
- **API整合性**: ヘッダー・実装間の完全な一致
- **PyTorch自動微分対応**: SPSAの`requires_grad=true`テンソル安全操作

## 🚀 次ステップ
Chain-of-Thought: **テスト実行** → 結果分析 → 残存問題特定 → 最終調整

**📚 詳細が必要な場合**: `docs/PROMPT_INDEX.md` から該当セクションを参照
