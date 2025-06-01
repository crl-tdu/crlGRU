# crlGRU 状況別プロンプト（タスク特化版）

## 🔧 ビルドエラー修正専用プロンプト

### 現在のタスク
**attention_evaluator.cpp のビルドエラー修正**

### エラー詳細
```cpp
// 未定義メンバー変数エラー
distance_attention_    // Conv2dが必要
angle_attention_       // Conv2dが必要  
fusion_layer_         // Linearが必要
dropout_              // Dropoutが必要

// 不足設定パラメータ
config_.attention_dim     // 新規追加が必要
config_.attention_dropout // 新規追加が必要
```

### 修正方針
1. **PolarSpatialAttentionヘッダー修正**: メンバー変数追加
2. **PolarSpatialAttentionConfig拡張**: 設定パラメータ追加
3. **attention_evaluator.cpp修正**: 実装との整合性確保

### 参照ファイル
- ヘッダー: `include/crlgru/core/polar_spatial_attention.hpp`
- 設定: `include/crlgru/utils/config_types.hpp`
- 実装: `src/core/attention_evaluator.cpp`

### Chain-of-Thoughtステップ
1. エラー分析 → ヘッダー修正 → 実装修正 → ビルド確認

---

## 🧪 テスト実行専用プロンプト

### テスト対象
- ファイル: `tests/test_crlgru.cpp`
- ビルドディレクトリ: `build_test_new/`
- 期待結果: 全テスト成功

### 実行手順
```bash
cd /Users/igarashi/local/project_workspace/crlGRU/build_test_new
make -j4
./tests/test_crlgru
```

---

## 🔍 ヘッダーオンリー開発専用プロンプト

### 対象ディレクトリ
- `include/crlgru/utils/`
- `include/crlgru/optimizers/`

### 開発パターン
```cpp
// テンプレート化 + 完全実装
template<typename T>
inline T function(const T& input) {
    // ヘッダー内完全実装
    return result;
}
```

### テスト方法
- ファイル: `tmp/test_header_only.cpp`
- 成功基準: 11/11テスト成功
