# crlGRU 使用ガイド

## 目次

1. [概要](#概要)
2. [インストール方法](#インストール方法)
3. [基本的な使い方](#基本的な使い方)
4. [高度な使用例](#高度な使用例)
5. [トラブルシューティング](#トラブルシューティング)
6. [よくある質問](#よくある質問)

## 概要

crlGRUは、自由エネルギー原理（Free Energy Principle）に基づくGRUニューラルネットワークライブラリです。主な特徴：

- 🧠 **自由エネルギー最小化**: Karl Fristonの理論に基づく予測的符号化
- 🤖 **マルチエージェント対応**: エージェント間の相互模倣学習
- 🎯 **空間認知**: 極座標ベースの空間注意メカニズム
- 📊 **多目的最適化**: メタ評価による柔軟な目的関数設計

## インストール方法

### 前提条件

1. **C++17対応コンパイラ**
   - GCC 7以降
   - Clang 5以降
   - Apple Clang (macOS)

2. **CMake 3.18以降**
   ```bash
   # macOS
   brew install cmake
   
   # Ubuntu/Debian
   sudo apt-get install cmake
   ```

3. **LibTorch（PyTorch C++ API）**
   - [LibTorchインストールガイド](./LIBTORCH_INSTALL_JP.md)を参照

### crlGRUのビルドとインストール

```bash
# ソースコードの取得（仮定）
cd ~/local/project_workspace/crlGRU

# ビルドディレクトリの作成
mkdir build && cd build

# CMake設定
cmake .. -DCMAKE_BUILD_TYPE=Release

# ビルド
make -j8

# インストール（~/local/にインストール）
make install
```

### 環境変数の設定

```bash
# ~/.bashrcまたは~/.zshrcに追加
export CRLGRU_HOME=$HOME/local
export LD_LIBRARY_PATH=$CRLGRU_HOME/lib:$LD_LIBRARY_PATH
export DYLD_LIBRARY_PATH=$CRLGRU_HOME/lib:$DYLD_LIBRARY_PATH  # macOS
```

## 基本的な使い方

### 1. 最小限のサンプル

```cpp
#include <crlgru/crl_gru.hpp>
#include <iostream>

int main() {
    // FEP-GRUセルの設定
    crlgru::FEPGRUCell::Config config;
    config.input_size = 10;
    config.hidden_size = 64;
    
    // セルの作成
    auto gru_cell = std::make_shared<crlgru::FEPGRUCell>(config);
    
    // 入力データ
    auto input = torch::randn({1, 10});
    auto hidden = torch::zeros({1, 64});
    
    // フォワードパス
    auto [new_hidden, prediction, free_energy] = gru_cell->forward(input, hidden);
    
    std::cout << "自由エネルギー: " << free_energy.mean().item<double>() << std::endl;
    
    return 0;
}
```

### 2. 時系列予測

```cpp
#include <crlgru/crl_gru.hpp>
#include <vector>

// 時系列データの予測
void predict_time_series() {
    // ネットワーク設定
    crlgru::FEPGRUNetwork::NetworkConfig config;
    config.layer_sizes = {1, 32, 64, 32, 1};
    
    auto network = std::make_shared<crlgru::FEPGRUNetwork>(config);
    
    // サンプルデータ生成（サイン波）
    std::vector<float> data;
    for (int i = 0; i < 100; ++i) {
        data.push_back(std::sin(i * 0.1));
    }
    
    // 予測
    const int window_size = 10;
    for (int i = 0; i < data.size() - window_size; ++i) {
        auto input = torch::from_blob(&data[i], {1, window_size, 1});
        auto [output, _, free_energy] = network->forward(input);
        
        std::cout << "時刻 " << i + window_size 
                  << ": 予測=" << output[0][0][0].item<float>()
                  << ", 実際=" << data[i + window_size] << std::endl;
    }
}
```

### 3. マルチエージェント協調

```cpp
#include <crlgru/crl_gru.hpp>

class Agent {
    std::shared_ptr<crlgru::FEPGRUCell> brain;
    torch::Tensor position;
    torch::Tensor hidden_state;
    
public:
    Agent() {
        crlgru::FEPGRUCell::Config config;
        config.input_size = 64;
        config.hidden_size = 128;
        brain = std::make_shared<crlgru::FEPGRUCell>(config);
        
        position = torch::randn({2});
        hidden_state = torch::zeros({1, 128});
    }
    
    void interact_with(const std::vector<Agent>& others) {
        // 他エージェントの位置を極座標で表現
        std::vector<torch::Tensor> positions;
        for (const auto& other : others) {
            positions.push_back(other.position);
        }
        
        auto positions_tensor = torch::stack(positions).unsqueeze(0);
        auto polar_map = crlgru::utils::cartesian_to_polar_map(
            positions_tensor, position.unsqueeze(0), 8, 16, 10.0
        );
        
        // 入力として使用
        auto input = polar_map.flatten(1);
        auto [new_hidden, action, _] = brain->forward(input, hidden_state);
        hidden_state = new_hidden;
        
        // 行動を速度に変換
        auto velocity = action[0].slice(0, 0, 2).tanh();
        position += velocity * 0.1;
    }
};
```

## 高度な使用例

### カスタム評価関数

```cpp
class MyEvaluator : public crlgru::MetaEvaluator {
public:
    std::unordered_map<std::string, double> evaluate(
        const torch::Tensor& state,
        const torch::Tensor& target,
        const std::unordered_map<std::string, torch::Tensor>& context) override {
        
        std::unordered_map<std::string, double> metrics;
        
        // カスタムメトリクスの計算
        if (context.find("position") != context.end()) {
            auto pos = context.at("position");
            metrics["distance_from_origin"] = pos.norm().item<double>();
        }
        
        if (context.find("velocity") != context.end()) {
            auto vel = context.at("velocity");
            metrics["speed"] = vel.norm().item<double>();
        }
        
        return metrics;
    }
};
```

### SPSA最適化

```cpp
void optimize_with_spsa() {
    // ネットワークの作成
    auto network = std::make_shared<crlgru::FEPGRUNetwork>(config);
    
    // パラメータの取得
    auto params = network->get_parameters();
    std::vector<torch::Tensor> param_list;
    for (const auto& [name, param] : params) {
        param_list.push_back(param);
    }
    
    // SPSA最適化器
    crlgru::SPSAOptimizer::OptimizerConfig opt_config;
    auto optimizer = std::make_shared<crlgru::SPSAOptimizer>(param_list, opt_config);
    
    // 最適化ループ
    for (int iter = 0; iter < 1000; ++iter) {
        auto loss_fn = [&]() {
            // 損失関数の計算
            auto output = network->forward(input);
            return torch::mse_loss(output, target).item<double>();
        };
        
        optimizer->step(loss_fn, iter);
    }
}
```

### 階層的模倣学習

```cpp
void hierarchical_imitation() {
    auto teacher = std::make_shared<crlgru::FEPGRUNetwork>(config);
    auto student = std::make_shared<crlgru::FEPGRUNetwork>(config);
    
    // パラメータレベルの模倣
    student->hierarchical_imitation_update(teacher, 0, 0.1);
    
    // ダイナミクスレベルの模倣
    student->hierarchical_imitation_update(teacher, 1, 0.05);
    
    // 意図レベルの模倣
    student->hierarchical_imitation_update(teacher, 2, 0.02);
}
```

## トラブルシューティング

### よくあるエラーと解決方法

#### 1. ライブラリが見つからない

```
error: cannot find -lcrlGRU
```

**解決方法:**
```bash
# ライブラリパスを確認
ls -la $HOME/local/lib/libcrlGRU*

# CMakeに明示的にパスを指定
cmake .. -DCMAKE_PREFIX_PATH=$HOME/local
```

#### 2. LibTorchのバージョン不一致

```
undefined reference to `c10::Error::Error(c10::SourceLocation, std::string)'
```

**解決方法:**
```bash
# LibTorchのバージョンを確認
cat $HOME/local/libtorch/build-version

# crlGRUを再ビルド
rm -rf build && mkdir build && cd build
cmake .. && make clean && make -j8
```

#### 3. 実行時のライブラリエラー

```
dyld: Library not loaded: @rpath/libtorch.dylib
```

**解決方法:**
```bash
# macOS
export DYLD_LIBRARY_PATH=$HOME/local/libtorch/lib:$DYLD_LIBRARY_PATH

# Linux
export LD_LIBRARY_PATH=$HOME/local/libtorch/lib:$LD_LIBRARY_PATH

# または実行ファイルにRPATHを追加
install_name_tool -add_rpath $HOME/local/libtorch/lib your_program  # macOS
patchelf --set-rpath $HOME/local/libtorch/lib your_program  # Linux
```

### デバッグテクニック

```cpp
// テンソルの状態確認
void debug_tensor(const torch::Tensor& t, const std::string& name) {
    std::cout << name << ":" << std::endl;
    std::cout << "  Shape: " << t.sizes() << std::endl;
    std::cout << "  Mean: " << t.mean().item<double>() << std::endl;
    std::cout << "  Std: " << t.std().item<double>() << std::endl;
    std::cout << "  Has NaN: " << t.isnan().any().item<bool>() << std::endl;
}

// パフォーマンス測定
class Timer {
    std::chrono::high_resolution_clock::time_point start;
public:
    Timer() : start(std::chrono::high_resolution_clock::now()) {}
    
    double elapsed() {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double>(end - start).count();
    }
};

// 使用例
Timer timer;
network->forward(input);
std::cout << "Forward pass: " << timer.elapsed() << " seconds" << std::endl;
```

## よくある質問

### Q1: GPUは使えますか？

**A:** はい、LibTorchがCUDA対応版の場合は自動的にGPUを使用します。

```cpp
if (torch::cuda::is_available()) {
    auto device = torch::Device(torch::kCUDA);
    auto input = input.to(device);
    auto network = network->to(device);
}
```

### Q2: メモリ使用量を削減するには？

**A:** 以下の方法があります：

1. バッチサイズを小さくする
2. 隠れ層のサイズを削減する
3. SOM機能を無効化する（`config.enable_som_extraction = false`）
4. グラディエントを無効化する（推論時）

```cpp
torch::NoGradGuard no_grad;  // 推論時のメモリ節約
auto output = network->forward(input);
```

### Q3: 学習が収束しない

**A:** 以下を試してください：

1. 学習率の調整（`config.learning_rate`）
2. 自由エネルギーの温度パラメータ調整（`config.beta`）
3. ネットワーク構造の変更（層数、隠れ層サイズ）
4. 正規化の追加

### Q4: カスタムモジュールを追加するには？

**A:** crlgru名前空間内でクラスを定義し、必要なインターフェースを実装します：

```cpp
namespace crlgru {
    class MyCustomModule {
    public:
        torch::Tensor forward(const torch::Tensor& input) {
            // カスタム処理
            return output;
        }
    };
}
```

## 参考資料

- [APIリファレンス](./API_REFERENCE_JP.md)
- [チュートリアル](./TUTORIAL_JP.md)
- [LibTorchインストールガイド](./LIBTORCH_INSTALL_JP.md)
- [サンプルコード](../examples/)

---

**サポート**: 問題が解決しない場合は、GitHubのIssuesページでお問い合わせください。

**Happy Coding with crlGRU! 🤖✨**
