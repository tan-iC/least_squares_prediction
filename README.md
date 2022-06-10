
# 最小二乗推定法の実装

- バッチ型最小二乗推定法と逐次型最小二乗推定法の2種類の観測値推定
- 定数値のベクトル$\bm{x} = [x_0, x_1, ...x_n]^T$を推定する

## k番目の観測方程式

$$ z(k) = \bm{H}(k)\bm{x} + w(k) $$

## k番目の誤差共分散

$$R(k) = E[w(k)w(k)^T]$$

今回は特に $$ \bm{H}(k) = [1, k, k^2]$$,$$\bm{x} = [x_0, x_1, x_2]^T $$

## バッチ型

- 行列計算で一度に処理

1. 観測方程式 $$ \bm{z} = \bm{H}\bm{x} + \bm{w} $$

1. 誤差共分散 $$ \bm{R} = \bold{diag}[R(0), R(1), ...]$$

1. 推定値 $$ \hat{\bm{x}} = [\bm{H}^{T}\bm{R}^{-1}\bm{H}]^{-1}\bm{H}^{T}\bm{R}^{-1}\bm{z} $$

## 逐次型

- k番目の推定値をもとにk+1番目の推定値を求める
- $\hat{\bm{x}}(0) = [0, 0, ..., 0]^T$, $\bm{P}(0) = \bold{diag}[10^a, 10^a, ..., 10^a]$と初期化する

1. 観測予測誤差共分散 $$\bm{S}(k+1) = \bm{H}(k+1)\bm{P}(k)\bm{H}^T(k+1)+\bm{R}(k+1)$$

1. フィルタゲイン $$\bm{W}(k+1) = \bm{P}(k)\bm{H}^T(k+1)\bm{S}^{-1}(k+1)$$

1. k+1番目の推定値 $$\hat{\bm{x}}(k+1) = \hat{\bm{x}}(k) + \bm{W}(k+1) [z(k+1) - \bm{H}(k+1)\hat{\bm{x}}(k)]$$

1. 推定誤差共分散 $$\bm{P}(k+1) = \bm{P}(k)-\bm{W}(k+1)\bm{S}(k+1)\bm{W}^T(k+1)$$

## リポジトリ内容

- ```src```: ソースコード

- ```data```: 観測データ

- ```result```: 逐次型最小二乗法での推定値の推移をプロットしたグラフ

- ```README.md```: このファイル
