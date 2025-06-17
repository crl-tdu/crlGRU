/**
 * @file crlgru_simple.h
 * @brief crlGRU簡易統合ヘッダー
 * @author 五十嵐研究室
 * @date 2025年6月
 *
 * 条件分岐なしの簡易インクルード
 */

#ifndef CRLGRU_SIMPLE_H
#define CRLGRU_SIMPLE_H

// バージョン情報
#define CRLGRU_VERSION_MAJOR 1
#define CRLGRU_VERSION_MINOR 0
#define CRLGRU_VERSION_PATCH 0
#define CRLGRU_VERSION "1.0.0"

// 実装されているコアコンポーネント
#include "crlgru/core/fep_gru_cell.hpp"
#include "crlgru/core/fep_gru_network.hpp"
#include "crlgru/core/polar_spatial_attention.hpp"
#include "crlgru/core/embodied_fep_gru_cell.hpp"

// ヘッダーオンリーコンポーネント
#include "crlgru/utils/config_types.hpp"
#include "crlgru/utils/math_utils.hpp"
#include "crlgru/utils/spatial_transforms.hpp"
#include "crlgru/optimizers/spsa_optimizer.hpp"

// 共通定義
#include "crlgru/common.h"

namespace crlgru {
    inline const char* version() {
        return CRLGRU_VERSION;
    }
}

#endif // CRLGRU_SIMPLE_H