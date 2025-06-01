# crlGRU プロンプトインデックス（動的参照システム）

## 🎯 階層化プロンプト構造

### 📍 レベル1: コアプロンプト（日常使用・最小）
**ファイル**: `prompts/core/CURRENT_TASK.md` | **サイズ**: ~1,667文字 | **削減率**: 93.0%
- **用途**: 現在のタスク専用（attention_evaluator.cpp修正）
- **内容**: 具体的エラー + 修正手順 + 次ステップ

### 📍 レベル2: 簡潔ベースライン（プロジェクト概要）
**ファイル**: `prompts/compact/BASELINE_COMPACT.md` | **サイズ**: ~2,682文字 | **削減率**: 88.8%
- **用途**: プロジェクト全体把握、新機能開発
- **内容**: ハイブリッドアーキテクチャ + 実装状況 + 開発環境

### 📍 レベル3: 専門分野別プロンプト（必要時参照）
**ディレクトリ**: `prompts/specialized/`

| ファイル | 用途 | サイズ | 特徴 |
|---------|------|--------|------|
| `build_fix.md` | ビルドエラー修正 | ~1,950文字 | Chain-of-Thought手順 |
| `testing.md` | テスト実行 | ~1,021文字 | 実行コマンド+成功指標 |
| `header_only_dev.md` | ヘッダーオンリー開発 | ~1,592文字 | 実装パターン集 |

### 📍 レベル4: 完全リファレンス（検索・参照用）
**ファイル**: `prompts/reference/FULL_REFERENCE.md` | **サイズ**: 23,787文字
- **用途**: 詳細実装例、完全な技術仕様、新人教育
- **参照**: 必要時のみ該当セクションを検索

## 🔄 動的選択ガイド

### 現在のタスクベース選択
```
📋 タスク: ビルドエラー修正
→ 使用: prompts/core/CURRENT_TASK.md + prompts/specialized/build_fix.md

📋 タスク: 新機能開発  
→ 使用: prompts/compact/BASELINE_COMPACT.md + prompts/specialized/header_only_dev.md

📋 タスク: テスト実行
→ 使用: prompts/specialized/testing.md

📋 タスク: 詳細API確認
→ 使用: prompts/reference/FULL_REFERENCE.md の該当セクション
```

### コンテキスト効率最適化
```
🚀 超高効率 (95.8%削減): CORE_PROMPT.md のみ
🚀 高効率 (88.8%削減): BASELINE_COMPACT.md
🔧 専門特化 (~90%削減): specialized_prompts/*.md
📚 完全参照 (0%削減): 元ファイル検索
```

## 📚 参照方法

### 即座に必要な情報
→ **CORE_PROMPT.md**（現在: attention_evaluator.cpp修正）

### プロジェクト背景理解
→ **BASELINE_COMPACT.md**

### 専門作業の詳細手順  
→ **specialized_prompts/[該当ファイル].md**

### 包括的技術仕様
→ **元ファイルから検索**

## 🎯 効率化実績
- **最小版**: 23,787文字 → 1,000文字（95.8%削減）
- **日常版**: 23,787文字 → 2,600文字（88.8%削減）  
- **専門版**: 23,787文字 → 800-1,800文字（90%+削減）

**結果**: 90%以上のコンテキスト削減を実現しながら、必要な情報は確実に提供
