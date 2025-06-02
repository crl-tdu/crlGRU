# crlGRU利用プロジェクト例

このディレクトリは、crlGRUをsubmoduleとして利用する親プロジェクトの例です。

## 🚀 基本的な使用方法

### Method 1: 直接submodule追加（推奨）

```cmake
# 親プロジェクトのCMakeLists.txt
cmake_minimum_required(VERSION 3.18)
project(MySwarmProject)

# crlGRUの設定（ビルド前に設定）
set(CRLGRU_BUILD_TESTS OFF CACHE BOOL "Disable crlGRU tests in submodule")
set(CRLGRU_BUILD_SHARED OFF CACHE BOOL "Use static linking for submodule")

# submodule追加（重複チェック付き）
if(NOT TARGET crlGRU)
    add_subdirectory(external/crlGRU EXCLUDE_FROM_ALL)
endif()

# プロジェクト定義
add_executable(my_swarm_app src/main.cpp)

# crlGRU依存追加
target_link_libraries(my_swarm_app PRIVATE crlGRU)
```

### Method 2: 条件付きビルド（キャッシュ活用）

```cmake
# プリビルド済みライブラリ検索
find_library(CRLGRU_LIBRARY
    NAMES crlGRU libcrlGRU
    PATHS ${CMAKE_CURRENT_SOURCE_DIR}/external/crlGRU/lib
    NO_DEFAULT_PATH
)

if(CRLGRU_LIBRARY)
    # プリビルド済みを使用
    add_library(crlGRU STATIC IMPORTED)
    set_target_properties(crlGRU PROPERTIES
        IMPORTED_LOCATION ${CRLGRU_LIBRARY}
        INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}/external/crlGRU/include
    )
    message(STATUS "Using pre-built crlGRU library: ${CRLGRU_LIBRARY}")
else()
    # ソースからビルド
    message(STATUS "Building crlGRU from source")
    set(CRLGRU_BUILD_TESTS OFF)
    add_subdirectory(external/crlGRU EXCLUDE_FROM_ALL)
endif()
```

## 🔧 Git Submodule セットアップ

```bash
# crlGRUをsubmoduleとして追加
git submodule add https://github.com/crl-tdu/crlGRU.git external/crlGRU

# 初期化とクローン
git submodule update --init --recursive

# submoduleを最新に更新
git submodule update --remote external/crlGRU
```

## 📊 ビルドオプション制御

crlGRUは以下のオプションでビルド動作を制御できます：

| オプション | デフォルト(main) | デフォルト(sub) | 説明 |
|------------|------------------|-----------------|------|
| `CRLGRU_BUILD_SHARED` | `ON` | `OFF` | 共有ライブラリビルド |
| `CRLGRU_BUILD_STATIC` | `ON` | `ON` | 静的ライブラリビルド |
| `CRLGRU_BUILD_TESTS` | `ON` | `OFF` | テストビルド |
| `CRLGRU_INSTALL` | `ON` | `OFF` | インストール設定 |

## 🎯 利用例

### C++コード例

```cpp
#include <crlgru/core/fep_gru_network.hpp>
#include <crlgru/optimizers/spsa_optimizer.hpp>
#include <crlgru/utils/spatial_transforms.hpp>

int main() {
    // FEP-GRUネットワーク作成
    crlgru::config::FEPGRUNetworkConfig config;
    config.layer_sizes = {64, 128, 64};
    config.cell_config.input_size = 64;
    
    auto network = std::make_shared<crlgru::core::FEPGRUNetwork>(config);
    
    // SPSA最適化器設定
    auto params = network->parameters();
    crlgru::optimizers::SPSAOptimizer<double>::Config opt_config;
    opt_config.learning_rate = 0.01;
    
    auto optimizer = std::make_shared<crlgru::optimizers::SPSAOptimizer<double>>(
        params, opt_config);
    
    // マルチエージェントシミュレーション
    for (int step = 0; step < 1000; ++step) {
        auto input = torch::randn({1, 10, 64});
        auto [output, prediction, free_energy] = network->forward(input);
        
        // 最適化ステップ
        auto objective = [&]() -> double {
            return free_energy.item<double>();
        };
        
        optimizer->step(objective, step);
    }
    
    return 0;
}
```

## 📦 パッケージマネージャ対応

### vcpkg対応 (将来)

```json
{
  "name": "crlgru",
  "version": "1.0.0",
  "dependencies": [
    "libtorch"
  ]
}
```

### Conan対応 (将来)

```python
from conan import ConanFile

class CrlgruConan(ConanFile):
    name = "crlgru"
    version = "1.0.0"
    
    def requirements(self):
        self.requires("libtorch/1.13.1")
```

## 🔍 トラブルシューティング

### よくある問題

1. **LibTorchが見つからない**
   ```cmake
   set(CMAKE_PREFIX_PATH "/path/to/libtorch")
   ```

2. **submoduleが空**
   ```bash
   git submodule update --init --recursive
   ```

3. **ビルドエラー**
   ```bash
   # クリーンビルド
   rm -rf build external/crlGRU/build*
   ```

### デバッグオプション

```cmake
# デバッグ情報を有効化
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_VERBOSE_MAKEFILE ON)
```
