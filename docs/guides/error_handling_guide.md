# crlGRU エラーハンドリングガイド

## 概要

このガイドでは、crlGRUプロジェクトに実装されたエラーハンドリング機能について説明します。

## エラーハンドリングの強化ポイント

### 1. 入力検証の強化

#### テンソルの形状・次元チェック

```cpp
// 例: FEPGRUCell::forward()内での入力検証
TensorValidator::check_not_empty(input, "input");
TensorValidator::check_dimensions(input, "input", 2);

if (input.size(1) != config_.input_size) {
    throw TensorShapeError("input", {-1, config_.input_size}, input.sizes());
}
```

#### 数値範囲の検証

```cpp
// ゲート値の範囲チェック
TensorValidator::check_range(reset_gate, "reset_gate", 0.0, 1.0);
TensorValidator::check_range(update_gate, "update_gate", 0.0, 1.0);
```

#### nullptr/未初期化チェック

```cpp
if (!layer) {
    throw ComputationError("Layer " + std::to_string(layer_idx) + " is null");
}
```

### 2. 例外安全性の向上

#### RAII原則の実装

```cpp
// TensorGuardクラスによる自動ロールバック
class TensorGuard {
    torch::Tensor& tensor_;
    torch::Tensor backup_;
    bool committed_ = false;

public:
    explicit TensorGuard(torch::Tensor& tensor) : tensor_(tensor) {
        if (tensor_.defined()) {
            backup_ = tensor_.clone();
        }
    }

    ~TensorGuard() {
        if (!committed_ && backup_.defined()) {
            tensor_ = backup_;  // エラー時は自動的に元の値に戻す
        }
    }

    void commit() { committed_ = true; }
};
```

使用例：
```cpp
TensorGuard hidden_guard(hidden_state_);
TensorGuard pred_error_guard(prediction_error_);
TensorGuard free_energy_guard(free_energy_);

try {
    // 計算処理...
    
    // すべての操作が成功したらコミット
    hidden_guard.commit();
    pred_error_guard.commit();
    free_energy_guard.commit();
} catch (...) {
    // エラー時は自動的にロールバック
    throw;
}
```

#### 適切な例外型の使用

```cpp
// 階層化された例外クラス
class CRLGRUException : public std::runtime_error { ... };
class InputValidationError : public CRLGRUException { ... };
class TensorShapeError : public InputValidationError { ... };
class NumericRangeError : public InputValidationError { ... };
class ConfigurationError : public CRLGRUException { ... };
class ComputationError : public CRLGRUException { ... };
```

### 3. デバッグ情報の充実

#### 詳細なエラーメッセージ

```cpp
std::ostringstream oss;
oss << "Forward pass failed in FEPGRUCell. "
    << "Batch size: " << input.size(0) << ", "
    << "Input size: " << input.size(1) << ", "
    << "Hidden size: " << config_.hidden_size << ". "
    << "Error: " << e.what();
throw ComputationError(oss.str());
```

#### デバッグモード対応

```cpp
#ifdef NDEBUG
    #define CRLGRU_ASSERT(condition, message) ((void)0)
#else
    #define CRLGRU_ASSERT(condition, message) \
        do { \
            if (!(condition)) { \
                std::ostringstream oss; \
                oss << "Assertion failed: " << #condition << " - " << message \
                    << " (File: " << __FILE__ << ", Line: " << __LINE__ << ")"; \
                throw crlgru::utils::CRLGRUException(oss.str()); \
            } \
        } while (false)
#endif
```

### 4. Graceful Degradation

#### 部分的な機能停止時の継続動作

```cpp
// SOM更新エラーは警告として扱い、処理を継続
if (config_.enable_som_extraction) {
    ErrorRecovery::with_fallback<void>(
        [&]() { update_som(new_hidden); return; },
        nullptr,
        "update_som"
    );
}
```

#### フォールバック機能

```cpp
// 自由エネルギー計算でエラーが発生した場合、ゼロテンソルを返す
free_energy_ = ErrorRecovery::with_fallback<torch::Tensor>(
    [&]() { return compute_free_energy(prediction, input, variance); },
    torch::zeros_like(prediction_error_),
    "compute_free_energy"
);
```

#### エラー状態のリカバリ

```cpp
// タイムステップレベルのエラーリカバリ
catch (const std::exception& e) {
    if (!layer_hidden_states.empty()) {
        // 前のタイムステップの値を使用して継続
        std::cerr << "[WARNING] " << oss.str() 
                  << ". Using previous timestep values." << std::endl;
        
        layer_hidden_states.push_back(layer_hidden_states.back());
        layer_predictions.push_back(layer_predictions.back());
        layer_free_energies.push_back(layer_free_energies.back());
        hidden_state = layer_hidden_states.back();
    } else {
        // 最初のタイムステップでのエラーは回復不可能
        throw ComputationError(oss.str());
    }
}
```

## 使用例

### 基本的な使用方法

```cpp
#include "crlgru/utils/error_handling.hpp"

void example_function() {
    try {
        // 入力検証
        TensorValidator::check_not_empty(input, "input");
        TensorValidator::check_dimensions(input, "input", 3);
        TensorValidator::check_finite(input, "input");
        
        // パラメータ検証
        ParameterValidator::check_positive(learning_rate, "learning_rate");
        ParameterValidator::check_probability(dropout_rate, "dropout_rate");
        
        // 処理実行...
        
    } catch (const InputValidationError& e) {
        // 入力エラーの処理
        std::cerr << "Input validation failed: " << e.what() << std::endl;
        throw;
    } catch (const ComputationError& e) {
        // 計算エラーの処理
        std::cerr << "Computation failed: " << e.what() << std::endl;
        // フォールバック処理...
    }
}
```

### エラーリカバリの実装

```cpp
// リトライ付き実行
auto result = ErrorRecovery::with_retry<torch::Tensor>(
    [&]() {
        return compute_complex_operation(input);
    },
    3,  // 最大3回リトライ
    "complex_operation"
);

// フォールバック値を使用
auto safe_result = ErrorRecovery::with_fallback<double>(
    [&]() {
        return risky_computation();
    },
    0.5,  // フォールバック値
    "risky_computation"
);
```

## パフォーマンスへの影響

エラーハンドリングの強化により、わずかなオーバーヘッドが発生しますが、以下の最適化により最小限に抑えています：

1. **リリースビルドでの最適化**
   - デバッグアサーションはリリースビルドで無効化
   - インライン化による関数呼び出しオーバーヘッドの削減

2. **選択的な検証**
   - クリティカルパスでは最小限の検証のみ実行
   - 非クリティカルな処理では警告として扱い、処理を継続

3. **効率的なエラーリカバリ**
   - 必要な場合のみバックアップを作成
   - move semanticsの活用によるコピーの削減

## 今後の拡張

1. **カスタムエラーハンドラの登録**
```cpp
ErrorHandler::register_handler(
    typeid(ComputationError),
    [](const std::exception& e) {
        // カスタムエラー処理
    }
);
```

2. **エラー統計の収集**
```cpp
ErrorStatistics::log_error(error_type, context);
auto stats = ErrorStatistics::get_summary();
```

3. **自動エラーレポート生成**
```cpp
ErrorReporter::generate_report("error_report.json");
```