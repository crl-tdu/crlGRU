# .gitignore 設定説明

このファイルは、crlGRUプロジェクトで追跡すべきでないファイルとディレクトリを定義します。

## 🎯 主要な除外カテゴリ

### 🔧 **ビルド関連**
```
build/              # メインビルドディレクトリ
build_*/            # 名前付きビルドディレクトリ
cmake-build-*/      # IDEが作成するビルドディレクトリ
*.a, *.so, *.dylib  # コンパイル済みライブラリ
CMakeCache.txt      # CMakeキャッシュ
```

### 💻 **IDE・エディタ固有**
```
.vscode/            # Visual Studio Code設定
.idea/              # CLion/IntelliJ設定
.vs/                # Visual Studio設定
*.xcodeproj/        # Xcode設定
```

### 🗂️ **システムファイル**
```
.DS_Store           # macOS Finder情報
Thumbs.db           # Windows サムネイル
*~                  # Linux/Unix バックアップファイル
```

### 🧪 **テスト・デバッグ**
```
test_results/       # テスト結果
coverage/           # カバレッジレポート
*.log               # ログファイル
core, core.*        # コアダンプ
```

### 📚 **ドキュメント生成**
```
doc/html/           # Doxygen出力
docs/_build/        # Sphinx出力
```

### 🤖 **機械学習関連**
```
*.pt, *.pth         # PyTorchモデル
runs/               # TensorBoard ログ
data/               # データセット
```

### 🔒 **セキュリティ**
```
*.key, *.pem        # 秘密鍵
.env                # 環境変数ファイル
config.json         # 設定ファイル（秘密情報含む可能性）
```

## 🔄 submodule利用時の特別考慮

### **変更検知ファイル**
```
.last_build_hash    # ビルドスクリプトが使用
*/.last_build_hash  # サブディレクトリ内の検知ファイル
```

### **キャッシュディレクトリ**
```
build_cache/        # 効率的ビルド用キャッシュ
*_cache/            # 各種キャッシュディレクトリ
```

## 📝 カスタマイズガイド

### **プロジェクト固有の追加**
```gitignore
# Custom additions セクションに追加
experimental/       # 実験的コード
private_configs/    # プライベート設定
local_scripts/      # ローカルスクリプト
```

### **一時的に追跡したい場合**
```bash
# 強制的に追加
git add -f ignored_file.txt

# 一時的に無視を停止
git update-index --no-skip-worktree file.txt
```

## 🎯 ベストプラクティス

### **✅ 推奨される使用方法**
1. **ビルド前**: 必ず.gitignoreを確認
2. **新規ファイル**: `git status`で意図しないファイルをチェック
3. **定期的**: 不要ファイルの蓄積を防ぐため定期クリーンアップ

### **🔍 状況確認コマンド**
```bash
# 無視されているファイルを表示
git status --ignored

# 追跡されていないファイルを表示
git ls-files --others --exclude-standard

# 無視ファイルのクリーンアップ（注意：削除されます）
git clean -fdX
```

### **⚠️ 注意事項**

1. **重要ファイルの誤無視防止**
   - `!CMakeLists.txt` で明示的に追跡
   - `!cmake/*.cmake` でCMakeモジュール保護

2. **submodule使用時**
   - 親プロジェクトの.gitignoreと競合しないよう配慮
   - キャッシュファイルの適切な除外

3. **セキュリティ**
   - 機密情報を含むファイルは確実に除外
   - APIキーや証明書の漏洩防止

## 🔧 トラブルシューティング

### **問題**: ファイルが無視されない
```bash
# キャッシュをクリア
git rm -r --cached .
git add .
git commit -m "Update .gitignore"
```

### **問題**: 必要なファイルが無視される
```bash
# 強制追加
git add -f important_file.txt

# .gitignoreを調整
echo "!important_file.txt" >> .gitignore
```

### **問題**: 大量の無視ファイルが表示される
```bash
# 無視ファイルをクリーンアップ
git clean -fdX  # 注意：削除されます
```

この.gitignoreにより、crlGRUプロジェクトのリポジトリが清潔に保たれ、重要なソースコードとドキュメントのみが追跡されます。
