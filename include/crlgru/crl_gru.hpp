/**
 * @file crl_gru.hpp
 * @brief crlGRU-crlNexus統合ヘッダー
 * @author 五十嵐研究室
 * @date 2025年6月
 * 
 * crlNexus プロジェクトからの互換性アクセスポイント
 * メイン統合ヘッダーへのリダイレクト
 */

#ifndef CRL_GRU_HPP
#define CRL_GRU_HPP

// メイン統合ヘッダーを読み込み
#include "crlgru.hpp"

// crlNexus向け下位互換性名前空間
namespace crl {
    namespace gru {
        // crlgru名前空間へのエイリアス
        using namespace crlgru;
        
        /**
         * @brief crlNexus向け初期化関数
         */
        inline bool initialize() {
            std::cout << "crlGRU initialized for crlNexus integration" << std::endl;
            crlgru::print_version();
            return true;
        }
        
        /**
         * @brief crlNexus統合テスト実行
         */
        inline void run_integration_test() {
            crlgru::test_nexus_integration();
        }
    }
}

// crlNexus向けマクロ定義
#define CRL_GRU_VERSION CRLGRU_VERSION
#define CRL_GRU_NEXUS_INTEGRATION

// crlNexus向けインクルードガード完了
#ifdef CRLNEXUS_SWARM_INTEGRATION
    // swarmモジュール向け追加定義
    namespace crl {
        namespace gru {
            namespace swarm {
                /**
                 * @brief swarmエージェント向けGRU初期化
                 */
                inline bool initializeForSwarm(int agent_count = 10) {
                    std::cout << "crlGRU initialized for " << agent_count << " swarm agents" << std::endl;
                    return true;
                }
            }
        }
    }
#endif

#ifdef CRLNEXUS_STA_INTEGRATION  
    // STAモジュール向け追加定義
    namespace crl {
        namespace gru {
            namespace sta {
                /**
                 * @brief STA向けGRU初期化
                 */
                inline bool initializeForSTA() {
                    std::cout << "crlGRU initialized for STA integration" << std::endl;
                    return true;
                }
            }
        }
    }
#endif

#endif // CRL_GRU_HPP