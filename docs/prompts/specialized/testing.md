# テスト実行専用プロンプト

## 🧪 テスト実行手順

### 統合テスト
- **ファイル**: `tests/test_crlgru.cpp`
- **ビルドディレクトリ**: `build_test_new/`
- **期待結果**: 全テスト成功

### 実行コマンド
```bash
cd /Users/igarashi/local/project_workspace/crlGRU/build_test_new
make -j4                 # ビルド
./tests/test_crlgru      # テスト実行
```

### ヘッダーオンリーテスト
- **ファイル**: `tmp/test_header_only.cpp`
- **成功基準**: 11/11テスト成功
- **実行**: 既に100%成功確認済み

## 🔍 テスト構造
```cpp
// 主要テストカテゴリ
1. FEP-GRU Cell Tests      ✅ 修正済み
2. FEP-GRU Network Tests   🔄 部分修正済み
3. SPSA Optimizer Tests    ✅ 完了
4. Utility Functions       ✅ 完了
5. Integration Tests       ⏳ ビルド成功後
```

## 📊 成功指標
- ビルドエラー: 0件
- テスト成功率: 100%
- メモリリーク: なし
- セグメンテーションフォルト: なし
