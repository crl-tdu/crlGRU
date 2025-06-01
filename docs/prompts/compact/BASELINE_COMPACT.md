# crlGRU 開発ベースラインプロンプト（簡潔版）

## プロジェクト概要
**crlGRU**: 自由エネルギー原理に基づくGRUライブラリ（ハイブリッドアーキテクチャ）
- **パス**: `/Users/igarashi/local/project_workspace/crlGRU`
- **目標**: マルチエージェントシステム・スワーム知能向け機械学習フレームワーク

## 🔧 開発制約・方針
- **ハイブリッドアプローチ**: ユーティリティはヘッダーオンリー、複雑なNNはライブラリ
- **Chain-of-Thought**: ステップバイステップ開発
- **テスト対象**: `tests/test_crlgru.cpp` のビルド・実行成功
- **実験ファイル**: 必ず `tmp/` ディレクトリに配置

## 📁 重要ディレクトリ構造
```
├── include/crlgru/
│   ├── core/                    # ライブラリコンポーネント（宣言）
│   │   ├── fep_gru_cell.hpp    # ✅ 修正完了
│   │   ├── fep_gru_network.hpp # ✅ 新規作成済み  
│   │   └── polar_spatial_attention.hpp # ✅ 新規作成済み
│   ├── optimizers/              # ヘッダーオンリー
│   └── utils/                   # ヘッダーオンリー
├── src/core/
│   ├── fep_gru_cell.cpp        # ✅ 修正完了
│   ├── fep_gru_network.cpp     # 🔄 部分修正済み
│   └── attention_evaluator.cpp # ❌ 修正待ち（最優先）
└── tests/test_crlgru.cpp       # ✅ 修正完了
```

## 🎯 現在の状況（2025年6月1日）

### ✅ 完了済み
- **ヘッダーオンリー**: 100%動作確認（11/11テスト成功）
- **FEPGRUCell**: ビルドエラー修正完了
- **テストファイル**: 修正完了

### 🔄 進行中  
- **FEPGRUNetwork**: 部分修正済み（エージェント管理変数名調整中）

### ❌ 最優先課題
**attention_evaluator.cpp ビルドエラー修正**
- 未定義メンバー変数: `distance_attention_`, `angle_attention_`, `fusion_layer_`
- 不足設定パラメータ: `attention_dim`, `attention_dropout`

## 🔧 開発環境
- **LibTorch**: `/Users/igarashi/local/libtorch` (2.1.2)
- **ビルド**: `build_test_new/` でエラー修正中
- **コンパイラ**: AppleClang 17.0.0 (macOS ARM64)

## 🎯 次のステップ
1. `attention_evaluator.cpp` の未定義メンバー変数修正
2. `PolarSpatialAttentionConfig` 拡張
3. ビルド成功確認
4. 統合テスト実行

---
**詳細情報**: 必要時に以下を参照
- 📘 理論的基盤: `docs/theory/THEORETICAL_FOUNDATIONS.md`
- 🔧 API詳細: `docs/guides/API_REFERENCE.md`  
- 📊 実装例: `tmp/build_fix_progress.sh`
