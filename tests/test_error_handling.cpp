#include <gtest/gtest.h>
#include <torch/torch.h>
#include "crlgru/core/fep_gru_cell.hpp"
#include "crlgru/core/fep_gru_network.hpp"
#include "crlgru/core/polar_spatial_attention.hpp"
#include "crlgru/utils/error_handling.hpp"

using namespace crlgru;
using namespace crlgru::utils;

class ErrorHandlingTest : public ::testing::Test {
protected:
    void SetUp() override {
        // テスト用の設定
        cell_config_.input_size = 10;
        cell_config_.hidden_size = 20;
        cell_config_.enable_som_extraction = true;
        cell_config_.som_grid_size = 5;
        
        network_config_.cell_config = cell_config_;
        network_config_.layer_sizes = {20, 15, 10};
        network_config_.dropout_rate = 0.1;
        
        attention_config_.input_channels = 16;
        attention_config_.attention_dim = 32;
        attention_config_.num_rings = 8;
        attention_config_.num_sectors = 12;
    }
    
    config::FEPGRUCellConfig cell_config_;
    config::FEPGRUNetworkConfig network_config_;
    config::PolarSpatialAttentionConfig attention_config_;
};

// テンソル検証のテスト
TEST_F(ErrorHandlingTest, TensorValidation) {
    // 未定義テンソルのチェック
    torch::Tensor undefined_tensor;
    EXPECT_THROW(
        TensorValidator::check_defined(undefined_tensor, "test_tensor"),
        InputValidationError
    );
    
    // 空テンソルのチェック
    torch::Tensor empty_tensor = torch::empty({0});
    EXPECT_THROW(
        TensorValidator::check_not_empty(empty_tensor, "test_tensor"),
        InputValidationError
    );
    
    // 次元数チェック
    torch::Tensor tensor_2d = torch::randn({10, 20});
    EXPECT_THROW(
        TensorValidator::check_dimensions(tensor_2d, "test_tensor", 3),
        InputValidationError
    );
    
    // 形状チェック
    EXPECT_THROW(
        TensorValidator::check_shape(tensor_2d, "test_tensor", {10, 30}),
        TensorShapeError
    );
    
    // 数値チェック（NaN）
    torch::Tensor nan_tensor = torch::full({5, 5}, std::numeric_limits<float>::quiet_NaN());
    EXPECT_THROW(
        TensorValidator::check_finite(nan_tensor, "test_tensor"),
        ComputationError
    );
    
    // 数値チェック（無限大）
    torch::Tensor inf_tensor = torch::full({5, 5}, std::numeric_limits<float>::infinity());
    EXPECT_THROW(
        TensorValidator::check_finite(inf_tensor, "test_tensor"),
        ComputationError
    );
    
    // 範囲チェック
    torch::Tensor out_of_range = torch::full({5, 5}, 2.0);
    EXPECT_THROW(
        TensorValidator::check_range(out_of_range, "test_tensor", 0.0, 1.0),
        NumericRangeError
    );
}

// パラメータ検証のテスト
TEST_F(ErrorHandlingTest, ParameterValidation) {
    // 正の値チェック
    EXPECT_THROW(
        ParameterValidator::check_positive(-1.0, "test_param"),
        NumericRangeError
    );
    
    // 範囲チェック
    EXPECT_THROW(
        ParameterValidator::check_range(1.5, "test_param", 0.0, 1.0),
        NumericRangeError
    );
    
    // 確率値チェック
    EXPECT_THROW(
        ParameterValidator::check_probability(1.1, "test_param"),
        NumericRangeError
    );
}

// FEPGRUCellのエラーハンドリングテスト
TEST_F(ErrorHandlingTest, FEPGRUCellErrorHandling) {
    // 無効な設定でのセル作成
    auto invalid_config = cell_config_;
    invalid_config.input_size = 0;
    EXPECT_THROW(
        core::FEPGRUCell(invalid_config),
        std::exception
    );
    
    // 正常なセル作成
    core::FEPGRUCell cell(cell_config_);
    
    // 無効な入力形状
    torch::Tensor invalid_input = torch::randn({5, 15}); // 期待: {batch, 10}
    torch::Tensor hidden = torch::randn({5, 20});
    EXPECT_THROW(
        cell.forward(invalid_input, hidden),
        TensorShapeError
    );
    
    // バッチサイズ不一致
    torch::Tensor input = torch::randn({5, 10});
    torch::Tensor mismatched_hidden = torch::randn({3, 20});
    EXPECT_THROW(
        cell.forward(input, mismatched_hidden),
        InputValidationError
    );
    
    // NaN入力
    torch::Tensor nan_input = torch::full({5, 10}, std::numeric_limits<float>::quiet_NaN());
    EXPECT_THROW(
        cell.forward(nan_input, hidden),
        ComputationError
    );
    
    // 正常な実行
    torch::Tensor valid_input = torch::randn({5, 10});
    torch::Tensor valid_hidden = torch::randn({5, 20});
    EXPECT_NO_THROW({
        auto [new_hidden, prediction, free_energy] = cell.forward(valid_input, valid_hidden);
        EXPECT_EQ(new_hidden.size(0), 5);
        EXPECT_EQ(new_hidden.size(1), 20);
        EXPECT_EQ(prediction.size(0), 5);
        EXPECT_EQ(prediction.size(1), 10);
    });
}

// FEPGRUNetworkのエラーハンドリングテスト
TEST_F(ErrorHandlingTest, FEPGRUNetworkErrorHandling) {
    // 無効な設定でのネットワーク作成
    auto invalid_config = network_config_;
    invalid_config.layer_sizes.clear();
    EXPECT_THROW(
        core::FEPGRUNetwork(invalid_config),
        ConfigurationError
    );
    
    // 正常なネットワーク作成
    core::FEPGRUNetwork network(network_config_);
    
    // 無効な入力形状（2次元）
    torch::Tensor invalid_input = torch::randn({5, 10});
    EXPECT_THROW(
        network.forward(invalid_input),
        InputValidationError
    );
    
    // 無効な入力形状（特徴次元不一致）
    torch::Tensor wrong_features = torch::randn({5, 10, 15}); // 期待: {batch, seq, 10}
    EXPECT_THROW(
        network.forward(wrong_features),
        TensorShapeError
    );
    
    // ゼロ長シーケンス
    torch::Tensor zero_seq = torch::randn({5, 0, 10});
    EXPECT_THROW(
        network.forward(zero_seq),
        InputValidationError
    );
    
    // 正常な実行
    torch::Tensor valid_input = torch::randn({5, 10, 10});
    EXPECT_NO_THROW({
        auto [output, prediction, free_energy] = network.forward(valid_input);
        EXPECT_EQ(output.size(0), 5);
        EXPECT_EQ(output.size(1), 10);
        EXPECT_EQ(output.size(2), network_config_.layer_sizes.back());
    });
}

// PolarSpatialAttentionのエラーハンドリングテスト
TEST_F(ErrorHandlingTest, PolarSpatialAttentionErrorHandling) {
    // 無効な設定でのアテンション作成
    auto invalid_config = attention_config_;
    invalid_config.input_channels = 0;
    EXPECT_THROW(
        attention::PolarSpatialAttention(invalid_config),
        std::exception
    );
    
    // 正常なアテンション作成
    attention::PolarSpatialAttention attention(attention_config_);
    
    // 無効な入力次元
    torch::Tensor invalid_input = torch::randn({5, 10, 20}); // 期待: 4次元
    EXPECT_THROW(
        attention.forward(invalid_input),
        InputValidationError
    );
    
    // NaN入力
    torch::Tensor nan_input = torch::full({5, 8, 12, 16}, 
                                         std::numeric_limits<float>::quiet_NaN());
    EXPECT_THROW(
        attention.forward(nan_input),
        ComputationError
    );
    
    // 正常な実行（フォーマット1）
    torch::Tensor valid_input1 = torch::randn({5, 8, 12, 16}); // [batch, rings, sectors, channels]
    EXPECT_NO_THROW({
        auto output = attention.forward(valid_input1);
        EXPECT_EQ(output.size(0), 5);
        EXPECT_EQ(output.size(1), 16);
        EXPECT_EQ(output.size(2), 8);
        EXPECT_EQ(output.size(3), 12);
    });
    
    // 正常な実行（フォーマット2）
    torch::Tensor valid_input2 = torch::randn({5, 16, 8, 12}); // [batch, channels, rings, sectors]
    EXPECT_NO_THROW({
        auto output = attention.forward(valid_input2);
        EXPECT_EQ(output.size(0), 5);
        EXPECT_EQ(output.size(1), 16);
        EXPECT_EQ(output.size(2), 8);
        EXPECT_EQ(output.size(3), 12);
    });
}

// エラーリカバリのテスト
TEST_F(ErrorHandlingTest, ErrorRecovery) {
    // フォールバックテスト
    auto result = ErrorRecovery::with_fallback<int>(
        []() -> int { throw std::runtime_error("Test error"); },
        42,
        "test_operation"
    );
    EXPECT_EQ(result, 42);
    
    // リトライテスト（成功）
    int attempt = 0;
    auto retry_result = ErrorRecovery::with_retry<int>(
        [&attempt]() -> int {
            if (++attempt < 2) {
                throw std::runtime_error("Retry test");
            }
            return 100;
        },
        3,
        "retry_operation"
    );
    EXPECT_EQ(retry_result, 100);
    EXPECT_EQ(attempt, 2);
    
    // リトライテスト（失敗）
    EXPECT_THROW(
        ErrorRecovery::with_retry<int>(
            []() -> int { throw std::runtime_error("Always fails"); },
            2,
            "failing_operation"
        ),
        ComputationError
    );
}

// Graceful Degradationのテスト
TEST_F(ErrorHandlingTest, GracefulDegradation) {
    core::FEPGRUNetwork network(network_config_);
    
    // 一部のタイムステップでエラーが発生するシミュレーション
    torch::Tensor input = torch::randn({2, 5, 10});
    
    // 2番目のタイムステップをNaNにする
    input.select(1, 1).fill_(std::numeric_limits<float>::quiet_NaN());
    
    // ネットワークは処理を継続すべき（警告は出るが例外は投げない）
    // 実際の実装では、前のタイムステップの値を使用して継続
    try {
        auto [output, prediction, free_energy] = network.forward(input);
        // エラーがあっても出力は生成される
        EXPECT_TRUE(output.defined());
        EXPECT_TRUE(prediction.defined());
        EXPECT_TRUE(free_energy.defined());
    } catch (const std::exception& e) {
        // 最初のタイムステップでない限り、エラーは発生しないはず
        ADD_FAILURE() << "Unexpected exception: " << e.what();
    }
}

// カスタムアサーションのテスト
TEST_F(ErrorHandlingTest, DebugAssertions) {
#ifndef NDEBUG
    // デバッグモードでのみ実行
    EXPECT_THROW(
        CRLGRU_ASSERT(false, "Test assertion"),
        CRLGRUException
    );
    
    // 条件が真の場合は例外を投げない
    EXPECT_NO_THROW(
        CRLGRU_ASSERT(true, "This should not throw")
    );
#endif
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}