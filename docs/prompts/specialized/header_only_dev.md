# ヘッダーオンリー開発専用プロンプト

## 🔍 ヘッダーオンリーコンポーネント開発

### 対象ディレクトリ
- `include/crlgru/utils/` ✅ 完全実装済み
- `include/crlgru/optimizers/` ✅ 完全実装済み

### 実装パターン
```cpp
// テンプレート化 + 完全実装パターン
template<typename FloatType = double>
class UtilityFunction {
public:
    // 完全実装をヘッダー内に記述
    inline FloatType process(const FloatType& input) {
        // 数値安定化を考慮した実装
        return safe_computation(input);
    }
    
private:
    inline FloatType safe_computation(const FloatType& x) {
        return x + static_cast<FloatType>(1e-8);
    }
};
```

### 数値安定性パターン
```cpp
// ゼロ除算回避
auto safe_result = value / (denominator + 1e-8);

// torch::Tensor 安全正規化
auto normalized = tensor / (tensor.norm() + 1e-8);

// ソフトマックス数値安定化
auto max_vals = std::get<0>(torch::max(logits, dim, true));
auto shifted = logits - max_vals;
```

## ✅ 動作確認済みコンポーネント
1. **SPSAOptimizer**: テンプレート化、モメンタム、制約付き最適化
2. **Math Utils**: safe_normalize, stable_softmax, tensor統計
3. **Spatial Transforms**: 極座標変換、回転変換、近傍探索  
4. **Config Types**: 型安全設定構造体、デフォルト値

## 🧪 テスト方法
- **ファイル**: `tmp/test_header_only.cpp`
- **成功基準**: 11/11テスト成功（既達成）
- **特徴**: LibTorchリンク不要での動作確認
