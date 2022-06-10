import  numpy   as np
import  pandas  as pd
import  matplotlib
matplotlib.use('Agg')
import  numpy.linalg        as la
import  matplotlib.pyplot   as plt


###
### 読み込み
###
df  = pd.read_csv('../data/input_data.csv')
# print(df)

k   = df['k'].values
z   = df['z'].values

#
# print(k)
'''
[-15 -14 -13 -12 -11 -10  -9  -8  -7  -6  -5  -4  -3  -2  -1   0   1   2
   3   4   5   6   7   8   9  10  11  12  13  14  15]
'''

#
# print(z)
'''
[162.1746 139.5805 113.8133  94.3372  74.7258  59.3817  41.4117  26.5951
  20.1832   8.8816   1.8636  -5.0213  -5.8861  -5.7711  -4.9332  -1.9845
   2.0593  12.3849  17.9044  30.1826  41.1677  55.7128  74.2944  93.7607
 112.6638 134.9818 162.7143 188.961  219.6236 248.9036 281.3082]
'''

# k = [odd, even]
w_aves  = np.array([0, 0])
w_vars  = np.array([1, 4])

# [3 x 1]
X_dim   = 3


def show(obj):
    print(f'type:{type(obj)}\nshape:{obj.shape}')

def init_H(k, X_dim, detail=False):
    H = []
    ls = k.tolist()

    for k in ls:

        #
        tmp = []
        for i in range(X_dim):
            tmp.append(k**i)

        H.append(tmp)

    H = np.array(H)

    if detail:
        show(H)

    return H

def init_R(k, w_vars, detail=False):

    ls  = k.tolist()    
    R   = []

    for k in ls:
        tmp = 0
        
        #
        if (k % 2) != 0:
            tmp = w_vars[0]
        else:
            tmp = w_vars[1]
        
        R.append(tmp)

    R = np.diag(np.array(R))

    if detail:
        show(R)

    return R

###
### z(k) = x_0 + k * x_1 + k ** 2 * x_2 + w(k)
###
### Z[k x 1] = H[k x 3] * X[3 x 1] + W[k x 1]
###

###
### A @ B       := 行列積演算子
### la.inv(A)   := 逆行列計算
### A.T         := 転置行列計算
###

###
### バッチ型
###
def batch(z, k, w_vars, X_dim, detail=False):
    ###
    ### x = [H^T * R^-1 * H]^-1 * H^T * R^-1 * z
    ###

    ### 初期化
    H = init_H(k, X_dim, detail=detail)
    R = init_R(k, w_vars, detail=detail)

    if detail:
        show(H)
        show(R)

    ### 計算の実行
    x = la.inv(H.T @ la.inv(R) @ H) @ H.T @ la.inv(R) @ z

    return x


###
### 逐次型
###
def sequential(z, k, w_vars, X_dim, num):

    X_ls    = []

    # H, Rの初期化
    H = init_H(k, X_dim)
    R = init_R(k, w_vars)

    # X, Pの初期化
    X = np.array([0.0] * X_dim)
    P = np.eye(X_dim) * (10**num) 

    # 逐次処理
    for k in range(len(H)): 

        # Hのk番目を行列計算できる形で取り出す
        # print(f'H[k].shape:{H[k].shape}')
        # print(f'H[k].shape:{H[k].reshape(-1,3).shape}')
        Hk  = H[k].reshape(-1,3)

        # S, Wの更新
        S   = Hk @ P @ Hk.T + R[k][k]
        W   = P @ Hk.T @ la.inv(S)

        # X, Pの更新
        X   = X + W @ (z[k] - Hk @ X)
        P   = P - W @ S @ W.T

        X_ls.append(X)

    return X_ls, P

###
### (1) 実行と出力
###
ans_batch   = batch(z, k, w_vars, X_dim)
print(ans_batch)

ans_seq     = sequential(z, k, w_vars, X_dim, 6)[0]
print(ans_seq[-1])


###
### (2) 初期共分散の値の変更
###
for i in range(-1, 8):
    ans_seq = sequential(z, k, w_vars, X_dim, i)[0]
    print(f'{pow(10, i)}: ans={ans_seq[-1]}')

    ### 初期化
    plt.figure()
    plt.ylim(-5, 5)

    ### 真値
    plt.hlines(4, 0, 31, color="black")
    plt.hlines(1, 0, 31, color="black")
    plt.hlines(-3, 0, 31, color="black")

    ### 予測値
    plt.plot(ans_seq)
    plt.xticks([i for i in range(0, 30+2, 2)], \
                [i for i in range(-15, 15+1, 2)])
    plt.grid()
    plt.title(f'P(0)=diag(10^{i}, ...)')
    plt.legend(['x0^', 'x1^', 'x2^', 'x0', 'x1', 'x2'])
    plt.xlabel("k")
    plt.ylabel("Predicted")
    plt.savefig(f'../result/seq_{pow(10, i)}.png')
