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
│   │   ├── fep_gru_network.hpp # ✅ 修正完了  
│   │   └── polar_spatial_attention.hpp # ✅ 修正完了
│   ├── optimizers/              # ヘッダーオンリー
│   │   └── spsa_optimizer.hpp  # ✅ in-place操作修正完了
│   └── utils/                   # ヘッダーオンリー
├── src/core/
│   ├── fep_gru_cell.cpp        # ✅ 修正完了
│   ├── fep_gru_network.cpp     # ✅ 修正完了
│   └── attention_evaluator.cpp # ✅ 修正完了
└── tests/test_crlgru.cpp       # ✅ 修正完了
```

## 🎯 現在の状況（2025年6月1日）

### ✅ 完了済み
- **ヘッダーオンリー**: 100%動作確認（SPSAオプティマイザー含む）
- **ライブラリコンポーネント**: 100%ビルド成功（0エラー）
- **設定構造体**: PolarSpatialAttentionConfig, MetaEvaluatorConfig拡張完了
- **SPSAオプティマイザー**: PyTorch自動微分対応、in-place操作除去完了
- **テストファイル**: 全修正完了

### 🔄 進行中  
- **全テストスイート実行**: 修正後の総合検証

### 🎯 次の優先課題
**全テストスイート実行による総合検証**
- 11テスト関数の完全実行確認
- SPSA関連テスト3個の動作確認  
- 統合テストによる相互作用検証

## 🔧 開発環境
- **LibTorch**: `/Users/igarashi/local/libtorch` (2.1.2)
- **ビルド**: `build_test_new/` で100%成功
- **コンパイラ**: AppleClang 17.0.0 (macOS ARM64)

## 🎉 技術的達成
- **自由エネルギー原理**: $F = E[q(z)] - \text{KL}[q(z)||p(z)]$ 実装完了
- **極座標注意機構**: $\text{Attention}(Q,K,V) = \text{softmax}(QK^T/\sqrt{d_k})V$ 拡張実装
- **SPSA勾配推定**: $\hat{g}_k(\theta_k) = \frac{f(\theta_k + c_k \Delta_k) - f(\theta_k - c_k \Delta_k)}{2c_k \Delta_k}$ 安全実装
- **0ビルドエラー**: 全コンポーネント完全修正
- **PyTorch自動微分対応**: `requires_grad=true`テンソル安全操作

## 🎯 次のステップ
1. 全テストスイート実行（11テスト）
2. 結果分析と残存問題特定
3. 最終調整とドキュメント更新
4. 完全統合確認

---
**詳細情報**: 必要時に以下を参照
- 📘 理論的基盤: `docs/theory/THEORETICAL_FOUNDATIONS.md`
- 🔧 API詳細: `docs/guides/API_REFERENCE.md`  
- 📊 修正履歴: `tmp/spsa_fix_report.md`
