# ビルドエラー修正専用プロンプト

## 🔧 現在のビルドエラー詳細

### attention_evaluator.cpp エラー
```cpp
// 未定義メンバー変数 (torch::nn::Module系)
distance_attention_    // torch::nn::Conv2d が必要
angle_attention_       // torch::nn::Conv2d が必要  
fusion_layer_         // torch::nn::Linear が必要
dropout_              // torch::nn::Dropout が必要

// 不足設定パラメータ
config_.attention_dim     // int, デフォルト64
config_.attention_dropout // double, デフォルト0.0
```

## 🎯 Chain-of-Thought修正手順

### Step 1: ヘッダー修正
- **ファイル**: `include/crlgru/core/polar_spatial_attention.hpp`
- **追加**: torch::nn::Conv2d, torch::nn::Linear, torch::nn::Dropout メンバー
- **初期化**: `{nullptr}` パターン使用

### Step 2: 設定構造体拡張  
- **ファイル**: `include/crlgru/utils/config_types.hpp`
- **追加**: PolarSpatialAttentionConfig に attention_dim, attention_dropout

### Step 3: 実装修正
- **ファイル**: `src/core/attention_evaluator.cpp`
- **修正**: 新メンバー変数への参照を適切に修正

### Step 4: ビルド確認
```bash
cd build_test_new && make -j4
```

## 🔍 修正パターン例

### torch::nn::Module メンバー追加パターン
```cpp
// ヘッダーファイル (.hpp)
private:
    torch::nn::Conv2d distance_attention_{nullptr};
    torch::nn::Conv2d angle_attention_{nullptr};
    torch::nn::Linear fusion_layer_{nullptr};
    torch::nn::Dropout dropout_{nullptr};

// 実装ファイル (.cpp)  
Constructor::Constructor() {
    distance_attention_ = register_module("distance_attention", 
        torch::nn::Conv2d(...));
}
```

## 📋 エラー修正チェックリスト
- [ ] ヘッダーにメンバー変数追加
- [ ] 設定構造体に新パラメータ追加  
- [ ] コンストラクタで初期化
- [ ] 実装での参照修正
- [ ] ビルド成功確認
