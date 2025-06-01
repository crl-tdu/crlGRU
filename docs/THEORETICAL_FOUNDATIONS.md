# crlGRU 理論的基盤

## 概要

このドキュメントは、crlGRUライブラリの理論的基盤となる数学的枠組みを詳述します。自由エネルギー原理、階層的模倣学習、SPSA最適化、極座標空間表現などの核心理論について、数式とその物理的・生物学的意味を明確に説明します。

## 1. 変分自由エネルギー原理

### 1.1 基本定式

Karl Fristonの自由エネルギー原理に基づき、各エージェント$i$の自由エネルギー$F_t^{(i)}$は以下で定義されます：

$$
F_t^{(i)} = D_{KL}[q_{\phi^{(i)}}(s_t^{(i)}) \| p(s_t^{(i)} \mid o_t^{(i)})] - \mathbb{E}_{q_{\phi^{(i)}}}[\log p(o_t^{(i)} \mid s_t^{(i)})]
$$

ここで：
- $q_{\phi^{(i)}}(s_t^{(i)})$: エージェント$i$の内部状態$s_t^{(i)}$に関する変分事後分布
- $p(s_t^{(i)} \mid o_t^{(i)})$: 観測$o_t^{(i)}$が与えられた時の真の事後分布
- $p(o_t^{(i)} \mid s_t^{(i)})$: 生成モデルの尤度
- $\phi^{(i)}$: エージェント$i$の内部モデルパラメータ

### 1.2 期待自由エネルギー

行動選択のための期待自由エネルギー$G_t^{(i)}$：

$$
G_t^{(i)}(u_{t:t+T-1}^{(i)}) = \mathbb{E}_{q_{\phi^{(i)}}}[F_{t+1:t+T}^{(i)}] + D_{KL}[q_{\phi^{(i)}}(s_{t+T}^{(i)}) \| p(s_{t+T}^{(i)})]
$$

ここで：
- $u_{t:t+T-1}^{(i)}$: 時刻$t$から$t+T-1$までの行動系列
- $T$: 予測ホライズン

### 1.3 ガウス分布仮定での解析解

実装効率化のため、ガウス分布仮定を採用：

$$
q_{\phi^{(i)}}(s_t^{(i)}) = \mathcal{N}(\mu_t^{(i)}, \Sigma_t^{(i)})
$$

$$
p(o_t^{(i)} \mid s_t^{(i)}) = \mathcal{N}(g(s_t^{(i)}), \tau^{-1}I)
$$

この仮定下で、自由エネルギーの解析的計算が可能：

$$
F_t^{(i)} = \frac{1}{2}[\tau \|o_t^{(i)} - g(\mu_t^{(i)})\|^2 + \text{tr}(\Sigma_t^{(i)})] - \frac{1}{2}\log|\Sigma_t^{(i)}| + \text{const}
$$

## 2. 階層的模倣学習メカニズム

### 2.1 3レベル階層構造

階層的模倣学習は以下の3つのレベルで実行されます：

#### レベル1：パラメータ模倣
$$
\theta_i^{(t+1)} = \theta_i^{(t)} + \alpha \cdot \tau_j \cdot (\theta_j^{(t)} - \theta_i^{(t)})
$$

ここで：
- $\theta_i^{(t)}$: エージェント$i$の時刻$t$でのパラメータ
- $\tau_j$: エージェント$j$への信頼度重み
- $\alpha$: パラメータ学習率

#### レベル2：ダイナミクス模倣
$$
h_i^{(t+1)} = h_i^{(t)} + \beta \cdot I(s_i, s_j) \cdot (h_j^{(t)} - h_i^{(t)})
$$

ここで：
- $h_i^{(t)}$: エージェント$i$の隠れ状態
- $I(s_i, s_j)$: 相互情報量重み
- $\beta$: ダイナミクス学習率

#### レベル3：意図模倣
$$
g_i^{(t+1)} = g_i^{(t)} + \gamma \cdot M(g_i, g_j) \cdot (g_j^{(t)} - g_i^{(t)})
$$

ここで：
- $g_i^{(t)}$: エージェント$i$の意図状態
- $M(g_i, g_j)$: 意図類似度メトリック
- $\gamma$: 意図学習率

### 2.2 信頼度重み計算

信頼度重み$\tau_j$は性能履歴と距離に基づいて計算：

$$
\tau_j = \exp\left(-\frac{d_{ij}}{d_{max}}\right) \cdot \frac{1}{N}\sum_{k=1}^{N} p_j^{(t-k)}
$$

ここで：
- $d_{ij}$: エージェント間距離
- $d_{max}$: 最大影響距離
- $p_j^{(t-k)}$: エージェント$j$の過去の性能

## 3. メタ評価関数

### 3.1 多目的統合評価

メタ評価関数は複数の目的を統合：

$$
J^{(i)}(\hat{y}_{t+1:t+T}^{(i)}) = \sum_{k=1}^K w_k^{(i)} J_k^{(i)}(\hat{y}_{t+1:t+T}^{(i)})
$$

ここで：
- $\hat{y}_{t+1:t+T}^{(i)}$: 予測状態系列
- $w_k^{(i)}$: 目的$k$の重み
- $K$: 評価項目数

### 3.2 評価項目の詳細

#### 目標達成度
$$
J_{goal}^{(i)} = -\|\hat{p}_{t+T}^{(i)} - g_t^{(i)}\|^2
$$

#### 衝突回避度
$$
J_{collision}^{(i)} = -\sum_{r,\phi} w_{r,\phi} \hat{m}_{r,\phi}^{(i)}(t+T) \cdot \mathbb{I}(r < r_{safe})
$$

#### 群凝集度
$$
J_{cohesion}^{(i)} = -\|\hat{p}_{t+T}^{(i)} - \mu_{neighbors}(t+T)\|^2
$$

#### 整列度
$$
J_{alignment}^{(i)} = \hat{v}_{t+T}^{(i)} \cdot \nu_{neighbors}(t+T)
$$

### 3.3 適応的重み調整

重みは性能履歴に基づいて動的調整：

$$
w_k^{(i)}(t+1) = w_k^{(i)}(t) + \eta \frac{\partial P^{(i)}}{\partial w_k^{(i)}}
$$

ここで：
- $P^{(i)}$: 総合性能指標
- $\eta$: 重み学習率

## 4. SPSA最適化アルゴリズム

### 4.1 勾配推定

同時摂動確率近似（SPSA）による勾配推定：

$$
\hat{\nabla} J^{(i)}(\theta_t^{(i)}) = \frac{J^{(i)}(\theta_t^{(i)} + c_k \Delta_k) - J^{(i)}(\theta_t^{(i)} - c_k \Delta_k)}{2c_k} \Delta_k^{-1}
$$

ここで：
- $\Delta_k$: 同時摂動ベクトル、$\Delta_k \sim \text{Bernoulli}(\pm 1)$
- $c_k$: 摂動幅、$c_k = \frac{c}{k^\gamma}$
- $c, \gamma$: SPSA調整パラメータ

### 4.2 パラメータ更新

推定勾配を用いたパラメータ更新：

$$
\theta_{t+1}^{(i)} = \theta_t^{(i)} - a_k \hat{\nabla} J^{(i)}(\theta_t^{(i)})
$$

ここで：
- $a_k$: 学習率、$a_k = \frac{a}{(A+k)^\alpha}$
- $a, A, \alpha$: SPSA調整パラメータ

### 4.3 収束条件

以下の条件で最適化を終了：

$$
\|\hat{\nabla} J^{(i)}(\theta_t^{(i)})\| < \epsilon_{grad} \quad \text{または} \quad |J^{(i)}(\theta_t^{(i)}) - J^{(i)}(\theta_{t-1}^{(i)})| < \epsilon_{obj}
$$

## 5. 極座標空間表現

### 5.1 座標変換

カルテシアン座標から極座標への変換：

$$
r = \sqrt{(x - x_{self})^2 + (y - y_{self})^2}
$$

$$
\theta = \arctan2(y - y_{self}, x - x_{self})
$$

### 5.2 空間離散化

極座標空間をリング×セクターで離散化：

$$
\text{ring\_idx} = \min\left(\lfloor r / (r_{max} / N_{rings}) \rfloor, N_{rings} - 1\right)
$$

$$
\text{sector\_idx} = \lfloor (\theta + \pi) / (2\pi / N_{sectors}) \rfloor \bmod N_{sectors}
$$

### 5.3 偏在解像度

生物学的妥当性を考慮した偏在解像度：

$$
\rho(r, \theta) = \rho_0 \exp\left(-\frac{r^2}{2\sigma_r^2}\right) \left(1 + A \cos(\theta - \theta_{pref})\right)
$$

ここで：
- $\rho_0$: 基準解像度
- $\sigma_r$: 距離減衰パラメータ
- $A$: 角度バイアス強度
- $\theta_{pref}$: 優先方向

## 6. SOM統合理論

### 6.1 競合学習

Self-Organizing Map (SOM)の競合学習則：

$$
w_{winner}(t+1) = w_{winner}(t) + \alpha(t) \cdot h(\|r_{winner} - r_j\|) \cdot (x(t) - w_j(t))
$$

### 6.2 近傍関数

ガウシアン近傍関数：

$$
h(d) = \exp\left(-\frac{d^2}{2\sigma(t)^2}\right)
$$

$$
\sigma(t) = \sigma_0 \exp\left(-\frac{t}{\tau}\right)
$$

## 7. 予測的符号化

### 7.1 予測誤差

階層的予測誤差の計算：

$$
\epsilon_l^{(t)} = x_l^{(t)} - \hat{x}_l^{(t)}
$$

$$
\hat{x}_l^{(t)} = f_l(\mu_{l+1}^{(t)})
$$

### 7.2 精度重み付き更新

精度を考慮した状態更新：

$$
\mu_l^{(t+1)} = \mu_l^{(t)} + \alpha \Pi_l \epsilon_l^{(t)}
$$

ここで：
- $\Pi_l$: レベル$l$の精度行列
- $\alpha$: 学習率

## 8. 相互情報量計算

### 8.1 定義

2つの状態間の相互情報量：

$$
I(X;Y) = \int\int p(x,y) \log\frac{p(x,y)}{p(x)p(y)} dx dy
$$

### 8.2 ガウス近似での解析解

ガウス分布仮定での相互情報量：

$$
I(X;Y) = \frac{1}{2}\log\frac{|\Sigma_X||\Sigma_Y|}{|\Sigma_{XY}|}
$$

ここで：
- $\Sigma_X, \Sigma_Y$: 周辺共分散行列
- $\Sigma_{XY}$: 結合共分散行列

## 9. 数値安定性と実装考慮事項

### 9.1 数値安定化テクニック

- **ゼロ除算回避**: $\frac{1}{\sigma^2 + \epsilon}$（$\epsilon = 10^{-8}$）
- **対数安定化**: $\log(\sigma^2 + \epsilon)$
- **勾配クリッピング**: $\|\nabla\| > \theta_{clip}$時のスケーリング

### 9.2 計算複雑度

- **FEP計算**: $O(d^2)$（$d$: 状態次元）
- **SPSA勾配推定**: $O(2 \cdot f_{eval})$（$f_{eval}$: 目的関数評価コスト）
- **極座標変換**: $O(N)$（$N$: エージェント数）
- **SOM更新**: $O(M)$（$M$: SOMノード数）

---

## 参考文献

1. Friston, K. (2010). The free-energy principle: a unified brain theory? Nature Reviews Neuroscience, 11(2), 127-138.
2. Spall, J. C. (1992). Multivariate stochastic approximation using a simultaneous perturbation gradient approximation. IEEE Transactions on Automatic Control, 37(3), 332-341.
3. Kohonen, T. (2001). Self-organizing maps. Springer Science & Business Media.
4. Rao, R. P., & Ballard, D. H. (1999). Predictive coding in the visual cortex: a functional interpretation of some extra-classical receptive-field effects. Nature Neuroscience, 2(1), 79-87.
