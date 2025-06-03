/**
 * @file crlgru.h
 * @brief crlGRU統合ヘッダー（宣言のみ）
 * @author 五十嵐研究室
 * @date 2025年6月
 *
 * 自由エネルギー原理に基づくGRUライブラリ
 * git submodule として使用される際のメインエントリーポイント
 */

#ifndef CRLGRU_H
#define CRLGRU_H

// ============================================================================
// バージョン情報
// ============================================================================
#define CRLGRU_VERSION_MAJOR 1
#define CRLGRU_VERSION_MINOR 0
#define CRLGRU_VERSION_PATCH 0
#define CRLGRU_VERSION "1.0.0"

// ============================================================================
// Forward Declarations
// ============================================================================
namespace crlgru {
    
    // Core components (implemented in src/core/)
    namespace core {
        class FEPGRUCell;
        class FEPGRUNetwork;
        class EmbodiedFEPGRUCell;        // NEW: 身体性FEP-GRUセル
        class PhysicalConstraintLayer;   // NEW: 物理制約レイヤー
    }
    
    // Attention mechanisms (implemented in src/core/)
    namespace attention {
        class PolarSpatialAttention;
        class MetaEvaluator;
    }
    
    // Optimizers (header-only in optimizers/)
    namespace optimizers {
        template<typename T>
        class SPSAOptimizer;
    }
    
    // Configuration structures (header-only in utils/)
    namespace config {
        struct FEPGRUCellConfig;
        struct FEPGRUNetworkConfig;
        struct PolarSpatialAttentionConfig;
        struct MetaEvaluatorConfig;
        struct SPSAOptimizerConfig;
    }
    
    // Integration layer (NEW: crlNexus互換性)
    namespace integration {
        class NexusCompatibilityLayer;   // NEW: crlNexus統合レイヤー
        class EmbodiedSensorModel;       // NEW: 身体性センサーモデル
        class MultiModalSensorFusion;    // NEW: マルチモーダルセンサー融合
        class EmbodiedSwarmIntegrator;   // NEW: 統合インターフェース
    }
    
    // Version information
    const char* version();
    void print_version();
}

// ============================================================================
// Conditional Includes
// ============================================================================

// Always include configuration types (header-only)
#include "crlgru/utils/config_types.hpp"

// Core components (requires linking with library)
#ifdef CRLGRU_INCLUDE_CORE
#include "crlgru/core/fep_gru_cell.hpp"
#include "crlgru/core/fep_gru_network.hpp"
#include "crlgru/core/polar_spatial_attention.hpp"
#endif

// Optimizers (header-only)
#ifdef CRLGRU_INCLUDE_OPTIMIZERS
#include "crlgru/optimizers/spsa_optimizer.hpp"
#endif

// Embodied FEP integration (NEW: 身体性FEP統合)
#ifdef CRLGRU_INCLUDE_EMBODIED
#include "crlgru/core/embodied_fep_gru_cell.hpp"
#include "crlgru/core/physical_constraint_layer.hpp"
#include "crlgru/integration/nexus_compatibility.hpp"
#include "crlgru/integration/embodied_sensors.hpp"
#endif

// Convenience: include everything when no specific flags are set
#if !defined(CRLGRU_INCLUDE_CORE) && !defined(CRLGRU_INCLUDE_OPTIMIZERS) && !defined(CRLGRU_INCLUDE_UTILS)
#define CRLGRU_INCLUDE_ALL
#endif

#ifdef CRLGRU_INCLUDE_ALL
#include "crlgru/core/fep_gru_cell.hpp"
#include "crlgru/core/fep_gru_network.hpp"
#include "crlgru/core/polar_spatial_attention.hpp"
#include "crlgru/optimizers/spsa_optimizer.hpp"
#include "crlgru/utils/math_utils.hpp"
#include "crlgru/utils/spatial_transforms.hpp"
#endif

// ============================================================================
// Version Implementation (inline)
// ============================================================================
namespace crlgru {
    inline const char* version() {
        return CRLGRU_VERSION;
    }
    
    inline void print_version() {
        // Implementation depends on whether iostream is available
        // Keep this simple for header-only usage
    }
}

#endif // CRLGRU_H
