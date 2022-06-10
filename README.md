
# 最小二乗推定法の実装

- バッチ型最小二乗推定法と逐次型最小二乗推定法の2種類の観測値推定
- 定数値のベクトル$\boldsymbol{x} = [x_0, x_1, ...x_n]^T$を推定する

## k番目の観測方程式

$$ z(k) = \boldsymbol{H}(k)\boldsymbol{x} + w(k) $$

## k番目の誤差共分散

$$R(k) = E[w(k)w(k)^T]$$

今回は特に $$ \boldsymbol{H}(k) = [1, k, k^2]$$,$$\boldsymbol{x} = [x_0, x_1, x_2]^T $$

## バッチ型

- 行列計算で一度に処理

1. 観測方程式 $$ \boldsymbol{z} = \boldsymbol{H}\boldsymbol{x} + \boldsymbol{w} $$

1. 誤差共分散 $$ \boldsymbol{R} = \bold{diag}[R(0), R(1), ...]$$

1. 推定値 $$ \boldsymbol{\hat{x}} = [\boldsymbol{H}^{T}\boldsymbol{R}^{-1}\boldsymbol{H}]^{-1}\boldsymbol{H}^{T}\boldsymbol{R}^{-1}\boldsymbol{z} $$

## 逐次型

- k番目の推定値をもとにk+1番目の推定値を求める
- $\boldsymbol{\hat{x}}(0) = [0, 0, ..., 0]^T$, $\boldsymbol{P}(0) = \bold{diag}[10^a, 10^a, ..., 10^a]$と初期化する

1. 観測予測誤差共分散 $$\boldsymbol{S}(k+1) = \boldsymbol{H}(k+1)\boldsymbol{P}(k)\boldsymbol{H}^T(k+1)+\boldsymbol{R}(k+1)$$

1. フィルタゲイン $$\boldsymbol{W}(k+1) = \boldsymbol{P}(k)\boldsymbol{H}^T(k+1)\boldsymbol{S}^{-1}(k+1)$$

1. k+1番目の推定値 $$\boldsymbol{\hat{x}}(k+1) = \{\boldsymbol{x}}(k) + \boldsymbol{W}(k+1) [z(k+1) - \boldsymbol{H}(k+1) \boldsymbol{ \hat{x} } (k)]$$

1. 推定誤差共分散 $$\boldsymbol{P}(k+1) = \boldsymbol{P}(k)-\boldsymbol{W}(k+1)\boldsymbol{S}(k+1)\boldsymbol{W}^T(k+1)$$

## リポジトリ内容

- ```src```: ソースコード

- ```data```: 観測データ

- ```result```: 逐次型最小二乗法での推定値の推移をプロットしたグラフ

- ```README.md```: このファイル
