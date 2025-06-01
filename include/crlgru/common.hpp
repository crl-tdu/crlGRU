#ifndef CRLGRU_COMMON_HPP
#define CRLGRU_COMMON_HPP

/// @file common.hpp
/// @brief crlGRU ライブラリ共通定義とフォワード宣言

#include <torch/torch.h>
#include <memory>
#include <vector>
#include <functional>
#include <unordered_map>

/// @brief crlGRU ライブラリ名前空間
namespace crlgru {

// バージョン情報
constexpr const char* VERSION = "1.0.0-hybrid";
constexpr int VERSION_MAJOR = 1;
constexpr int VERSION_MINOR = 0;
constexpr int VERSION_PATCH = 0;

// フォワード宣言
class FEPGRUCell;
class FEPGRUNetwork;
class PolarSpatialAttention;
class MetaEvaluator;

namespace utils {
    // ユーティリティ関数の前方宣言（ヘッダーオンリーなので実際は不要だが、文書化目的）
}

} // namespace crlgru

#endif // CRLGRU_COMMON_HPP
