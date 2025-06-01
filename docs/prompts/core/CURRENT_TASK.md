# crlGRU コアプロンプト（最小版）

## 🎯 現在のタスク
**attention_evaluator.cpp ビルドエラー修正** - 最優先

## 📁 作業対象
- `src/core/attention_evaluator.cpp` ❌ 修正待ち
- `include/crlgru/core/polar_spatial_attention.hpp` 🔄 拡張必要
- `include/crlgru/utils/config_types.hpp` 🔄 拡張必要

## ❌ 具体的エラー
```cpp
// 未定義メンバー変数
distance_attention_    // torch::nn::Conv2d 必要
angle_attention_       // torch::nn::Conv2d 必要  
fusion_layer_         // torch::nn::Linear 必要
dropout_              // torch::nn::Dropout 必要

// 不足設定パラメータ
config_.attention_dim     // PolarSpatialAttentionConfig に追加
config_.attention_dropout // PolarSpatialAttentionConfig に追加
```

## 🔧 修正手順
1. **PolarSpatialAttentionヘッダー修正**: 未定義メンバー変数4つを追加
2. **設定構造体拡張**: attention_dim, attention_dropout パラメータ追加
3. **実装修正**: ヘッダーとの整合性確保
4. **ビルド確認**: `build_test_new/` でmake実行

## 🎯 次ステップ
Chain-of-Thought: エラー分析完了 → **ヘッダー修正** → 実装修正 → ビルド確認

## 📂 プロジェクト基本情報
- **パス**: `/Users/igarashi/local/project_workspace/crlGRU`
- **ハイブリッドアーキテクチャ**: ヘッダーオンリー(utils/optimizers) + ライブラリ(core)
- **検証対象**: `tests/test_crlgru.cpp` のビルド・実行成功
- **実験ファイル**: `tmp/` ディレクトリに配置

**📚 詳細が必要な場合**: `docs/PROMPT_INDEX.md` から該当セクションを参照
