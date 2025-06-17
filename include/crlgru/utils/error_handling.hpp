#ifndef CRLGRU_UTILS_ERROR_HANDLING_HPP
#define CRLGRU_UTILS_ERROR_HANDLING_HPP

/// @file error_handling.hpp
/// @brief エラーハンドリングユーティリティ

#include <stdexcept>
#include <string>
#include <sstream>
#include <torch/torch.h>

namespace crlgru {
namespace utils {

/// @brief crlGRU固有の例外基底クラス
class CRLGRUException : public std::runtime_error {
public:
    explicit CRLGRUException(const std::string& message) 
        : std::runtime_error("crlGRU Error: " + message) {}
};

/// @brief 入力検証エラー
class InputValidationError : public CRLGRUException {
public:
    explicit InputValidationError(const std::string& message)
        : CRLGRUException("Input Validation Failed: " + message) {}
};

/// @brief テンソル形状エラー
class TensorShapeError : public InputValidationError {
public:
    TensorShapeError(const std::string& tensor_name, 
                     const torch::IntArrayRef& expected_shape,
                     const torch::IntArrayRef& actual_shape)
        : InputValidationError(format_message(tensor_name, expected_shape, actual_shape)) {}

private:
    static std::string format_message(const std::string& tensor_name,
                                     const torch::IntArrayRef& expected_shape,
                                     const torch::IntArrayRef& actual_shape) {
        std::ostringstream oss;
        oss << "Tensor '" << tensor_name << "' shape mismatch. Expected: [";
        for (size_t i = 0; i < expected_shape.size(); ++i) {
            if (i > 0) oss << ", ";
            oss << expected_shape[i];
        }
        oss << "], Got: [";
        for (size_t i = 0; i < actual_shape.size(); ++i) {
            if (i > 0) oss << ", ";
            oss << actual_shape[i];
        }
        oss << "]";
        return oss.str();
    }
};

/// @brief 数値範囲エラー
class NumericRangeError : public InputValidationError {
public:
    NumericRangeError(const std::string& parameter_name,
                      double value, double min_value, double max_value)
        : InputValidationError(format_message(parameter_name, value, min_value, max_value)) {}

private:
    static std::string format_message(const std::string& parameter_name,
                                     double value, double min_value, double max_value) {
        std::ostringstream oss;
        oss << "Parameter '" << parameter_name << "' value " << value 
            << " is out of range [" << min_value << ", " << max_value << "]";
        return oss.str();
    }
};

/// @brief 設定エラー
class ConfigurationError : public CRLGRUException {
public:
    explicit ConfigurationError(const std::string& message)
        : CRLGRUException("Configuration Error: " + message) {}
};

/// @brief 計算エラー
class ComputationError : public CRLGRUException {
public:
    explicit ComputationError(const std::string& message)
        : CRLGRUException("Computation Error: " + message) {}
};

/// @brief デバッグ情報を含むアサーション
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

/// @brief テンソル検証ユーティリティ
class TensorValidator {
public:
    /// @brief テンソルが定義されていることを確認
    static void check_defined(const torch::Tensor& tensor, const std::string& tensor_name) {
        if (!tensor.defined()) {
            throw InputValidationError("Tensor '" + tensor_name + "' is not defined");
        }
    }

    /// @brief テンソルが空でないことを確認
    static void check_not_empty(const torch::Tensor& tensor, const std::string& tensor_name) {
        check_defined(tensor, tensor_name);
        if (tensor.numel() == 0) {
            throw InputValidationError("Tensor '" + tensor_name + "' is empty");
        }
    }

    /// @brief テンソルの次元数を確認
    static void check_dimensions(const torch::Tensor& tensor, 
                                const std::string& tensor_name,
                                int expected_dims) {
        check_defined(tensor, tensor_name);
        if (tensor.dim() != expected_dims) {
            std::ostringstream oss;
            oss << "Tensor '" << tensor_name << "' expected " << expected_dims 
                << " dimensions, got " << tensor.dim();
            throw InputValidationError(oss.str());
        }
    }

    /// @brief テンソルの形状を確認
    static void check_shape(const torch::Tensor& tensor,
                           const std::string& tensor_name,
                           const torch::IntArrayRef& expected_shape) {
        check_defined(tensor, tensor_name);
        if (tensor.sizes() != expected_shape) {
            throw TensorShapeError(tensor_name, expected_shape, tensor.sizes());
        }
    }

    /// @brief バッチサイズの一致を確認
    static void check_batch_size_match(const torch::Tensor& tensor1,
                                      const torch::Tensor& tensor2,
                                      const std::string& tensor1_name,
                                      const std::string& tensor2_name) {
        check_defined(tensor1, tensor1_name);
        check_defined(tensor2, tensor2_name);
        
        if (tensor1.size(0) != tensor2.size(0)) {
            std::ostringstream oss;
            oss << "Batch size mismatch between '" << tensor1_name << "' (" 
                << tensor1.size(0) << ") and '" << tensor2_name << "' (" 
                << tensor2.size(0) << ")";
            throw InputValidationError(oss.str());
        }
    }

    /// @brief 数値が有限であることを確認
    static void check_finite(const torch::Tensor& tensor, const std::string& tensor_name) {
        check_defined(tensor, tensor_name);
        if (torch::any(torch::isnan(tensor)).item<bool>()) {
            throw ComputationError("Tensor '" + tensor_name + "' contains NaN values");
        }
        if (torch::any(torch::isinf(tensor)).item<bool>()) {
            throw ComputationError("Tensor '" + tensor_name + "' contains infinite values");
        }
    }

    /// @brief 数値範囲の確認
    static void check_range(const torch::Tensor& tensor,
                           const std::string& tensor_name,
                           double min_value,
                           double max_value) {
        check_defined(tensor, tensor_name);
        auto min_tensor = torch::min(tensor).item<double>();
        auto max_tensor = torch::max(tensor).item<double>();
        
        if (min_tensor < min_value || max_tensor > max_value) {
            std::ostringstream oss;
            oss << "Tensor '" << tensor_name << "' values [" << min_tensor 
                << ", " << max_tensor << "] out of range [" 
                << min_value << ", " << max_value << "]";
            throw NumericRangeError(tensor_name, min_tensor, min_value, max_value);
        }
    }
};

/// @brief パラメータ検証ユーティリティ
class ParameterValidator {
public:
    /// @brief 正の値であることを確認
    static void check_positive(double value, const std::string& param_name) {
        if (value <= 0) {
            throw NumericRangeError(param_name, value, 0.0, std::numeric_limits<double>::max());
        }
    }

    /// @brief 範囲内であることを確認
    static void check_range(double value, const std::string& param_name,
                           double min_value, double max_value) {
        if (value < min_value || value > max_value) {
            throw NumericRangeError(param_name, value, min_value, max_value);
        }
    }

    /// @brief 確率値であることを確認
    static void check_probability(double value, const std::string& param_name) {
        check_range(value, param_name, 0.0, 1.0);
    }
};

/// @brief エラーリカバリ機能
class ErrorRecovery {
public:
    /// @brief フォールバック値を使用したリカバリ
    template<typename T>
    static T with_fallback(std::function<T()> operation, 
                          const T& fallback_value,
                          const std::string& operation_name) {
        try {
            return operation();
        } catch (const std::exception& e) {
            std::cerr << "[WARNING] Operation '" << operation_name 
                      << "' failed: " << e.what() 
                      << ". Using fallback value." << std::endl;
            return fallback_value;
        }
    }

    /// @brief リトライ付き実行
    template<typename T>
    static T with_retry(std::function<T()> operation,
                       int max_retries,
                       const std::string& operation_name) {
        int retry_count = 0;
        while (retry_count < max_retries) {
            try {
                return operation();
            } catch (const std::exception& e) {
                retry_count++;
                if (retry_count >= max_retries) {
                    throw ComputationError("Operation '" + operation_name + 
                                         "' failed after " + std::to_string(max_retries) + 
                                         " retries: " + e.what());
                }
                std::cerr << "[WARNING] Operation '" << operation_name 
                          << "' failed (attempt " << retry_count << "/" << max_retries 
                          << "): " << e.what() << std::endl;
            }
        }
        throw ComputationError("Unexpected error in retry loop");
    }
};

} // namespace utils
} // namespace crlgru

#endif // CRLGRU_UTILS_ERROR_HANDLING_HPP