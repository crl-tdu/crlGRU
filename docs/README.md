# 📚 crlGRU Documentation

自由エネルギー原理に基づくGRUライブラリのドキュメント集

## 📁 ディレクトリ構造

### 📋 `prompts/` - AIプロンプト（階層化）
効率的なAI支援開発のための階層化プロンプトシステム

```
prompts/
├── core/                      # 🎯 レベル1: 現在タスク専用（最小・1,667文字）
│   └── CURRENT_TASK.md        #     attention_evaluator.cpp修正用
├── compact/                   # 🚀 レベル2: 日常使用（簡潔・2,682文字）
│   └── BASELINE_COMPACT.md    #     プロジェクト概要・新機能開発用
├── specialized/               # 🔧 レベル3: 専門分野別（特化・1,000-2,000文字）
│   ├── build_fix.md          #     ビルドエラー修正専用
│   ├── testing.md            #     テスト実行専用
│   └── header_only_dev.md    #     ヘッダーオンリー開発専用
├── reference/                 # 📚 レベル4: 完全リファレンス（詳細・23,787文字）
│   ├── FULL_REFERENCE.md     #     完全版（検索・参照用）
│   └── DEVELOPMENT_BASELINE_PROMPT.md  # 元ベースラインプロンプト
├── INDEX.md                   # 📋 プロンプト選択ガイド・動的参照システム
└── USAGE_GUIDE.md            # 🤖 AI読み込み方法・使い分けガイド
```

#### 🎯 使用方法
| タスク | 推奨プロンプト | 削減率 |
|--------|---------------|--------|
| **現在のタスク** | `prompts/core/CURRENT_TASK.md` | **93.0%削減** |
| **新機能開発** | `prompts/compact/BASELINE_COMPACT.md` | **88.8%削減** |
| **専門作業** | `prompts/specialized/[該当ファイル].md` | **90%+削減** |

### 📚 `guides/` - ユーザーガイド
実際の使用方法とAPI仕様

```
guides/
├── installation/              # インストール・セットアップ
│   └── LIBTORCH_SETUP.md     # LibTorchインストールガイド
├── usage/                     # 使用方法・チュートリアル
│   └── BASIC_USAGE.md        # 基本的な使用方法
└── API_REFERENCE.md          # API詳細リファレンス
```

### 📘 `theory/` - 理論・学術
数学的基盤と研究背景

```
theory/
└── THEORETICAL_FOUNDATIONS.md # 自由エネルギー原理・数学的枠組み
```

## 🚀 クイックスタート

### 現在のタスク（attention_evaluator.cpp修正）
→ **`prompts/core/CURRENT_TASK.md`** をAIに読み込ませてください

### 新機能開発・プロジェクト概要
→ **`prompts/compact/BASELINE_COMPACT.md`** をAIに読み込ませてください

### 詳細な読み込み方法
→ **`prompts/USAGE_GUIDE.md`** を参照してください

## 📊 効率化実績
- **最大93.0%のコンテキスト削減**を実現
- **必要な情報は100%保持**
- **段階的詳細化**により柔軟な情報提供

## 🔗 関連リンク
- [プロジェクトルート](../)
- [ソースコード](../src/)
- [テストファイル](../tests/)
- [実験ファイル](../tmp/)
