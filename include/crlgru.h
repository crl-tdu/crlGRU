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
        // Future implementations (not yet available):
        // class EmbodiedFEPGRUCell;        // TODO: 身体性FEP-GRUセル
        // class PhysicalConstraintLayer;   // TODO: 物理制約レイヤー
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
    
    // Integration layer (Future implementations)
    namespace integration {
        // Future implementations (not yet available):
        // class NexusCompatibilityLayer;   // TODO: crlNexus統合レイヤー
        // class EmbodiedSensorModel;       // TODO: 身体性センサーモデル
        // class MultiModalSensorFusion;    // TODO: マルチモーダルセンサー融合
        // class EmbodiedSwarmIntegrator;   // TODO: 統合インターフェース
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

// Embodied FEP integration (Future implementations)
// #ifdef CRLGRU_INCLUDE_EMBODIED
// #include "crlgru/core/embodied_fep_gru_cell.hpp"     // TODO: Not implemented
// #include "crlgru/core/physical_constraint_layer.hpp"  // TODO: Not implemented
// #include "crlgru/integration/nexus_compatibility.hpp" // TODO: Not implemented
// #include "crlgru/integration/embodied_sensors.hpp"    // TODO: Not implemented
// #endif

// Convenience: include everything when no specific flags are set
#if !defined(CRLGRU_INCLUDE_CORE) && !defined(CRLGRU_INCLUDE_OPTIMIZERS) && !defined(CRLGRU_INCLUDE_UTILS) && !defined(CRLGRU_HEADER_ONLY)
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

// Header-only mode (utilities and optimizers only)
#ifdef CRLGRU_HEADER_ONLY
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
