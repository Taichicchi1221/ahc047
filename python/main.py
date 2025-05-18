# author:  Taichicchi
# created: 18.05.2025 19:00:00

import sys

import numpy as np

N, M, L = 36, 12, 10**6


class Input:
    def __init__(self, S, P):
        self.S = S
        self.P = P

    def __str__(self):
        return f"S: {self.S}, P: {self.P}"


class Output:
    def __init__(self, C, A):
        self.C = C
        self.A = A

    def __str__(self):
        return f"C: {self.C}, A: {self.A}"

    def print(self):
        for i in range(len(self.C)):
            row = [self.C[i]] + list(map(str, self.A[i]))
            print(" ".join(row))


def greedy(input: Input):
    # 最も点数が高い文字列を探す
    best_idx = max(range(N), key=lambda i: input.P[i])
    best_str = input.S[best_idx]

    # best_strの長さでループを作り、残りの状態は適当な文字で埋める
    C = []
    A = []

    # best_strを割当
    for i in range(len(best_str)):
        C.append(best_str[i])

    # 残りの状態を適当な文字（例えば 'a'）で埋める
    while len(C) < M:
        C.append("a")

    # 遷移確率の初期化
    for i in range(M):
        row = [0] * M
        if i < len(best_str) - 1:
            row[i + 1] = 100  # 次の文字へ必ず遷移
        elif i == len(best_str) - 1:
            row[0] = 100  # ループに戻す
        else:
            row[i] = 100  # 余った状態は自分自身に留まる（影響しない）
        A.append(row)

    return Output(C, A)


def compute_word_probability(word, L, C, A):
    """
    word: 部分一致判定対象（list of char）
    L: 文字列長
    C: 各状態の割り当て文字（list of char, len=M）
    A: 遷移確率行列（M x M, 各行は合計100）
    """
    M = len(C)
    word = list(word)
    len_w = len(word)
    states = {}  # (マッチ長, 状態番号) -> 状態ID
    n = 0
    for j in range(M):
        states[(0, j)] = n
        n += 1
        for i in range(len_w - 1):
            if word[i] == C[j]:
                states[(i + 1, j)] = n
                n += 1

    # 遷移行列 X
    X = np.zeros((n, n))
    for (matchlen, u), j in states.items():
        for v in range(M):
            # マッチ進捗を計算
            next_ = word[:matchlen] + [C[v]]
            s = 0
            while next_[s:] != word[: len(next_) - s]:
                s += 1
            # wordが完成していない場合だけ遷移
            if len(next_) - s != len_w:
                i2 = states[(len(next_) - s, v)]
                X[i2, j] += A[u][v] / 100.0

    # 行列累乗 Y = X^(L-1)
    power = L - 1
    Y = np.eye(n)
    while power:
        if power % 2:
            Y = np.dot(X, Y)
        X = np.dot(X, X)
        power //= 2

    # 初期状態（C[0]とword[0]が一致していればマッチ長1、なければ0）
    init = states[(1, 0)] if C[0] == word[0] else states[(0, 0)]
    # "一度もマッチしなかった確率"の総和を計算し、1から引く
    ret = 1.0 - Y[:, init].sum()
    return max(0.0, min(1.0, ret))


def compute_score(input: Input, output: Output):
    total_score = 0.0
    for s, p in zip(input.S, input.P):
        prob = compute_word_probability(s, L, output.C, output.A)
        total_score += p * prob
    return round(total_score)


if __name__ == "__main__":
    input()  # 1行目を読み飛ばす
    S, P = [], []
    for _ in range(N):
        s, p = input().split()
        S.append(s)
        P.append(int(p))

    input = Input(S, P)

    output = greedy(input)
    output.print()

    score = compute_score(input, output)
    print(f"score {score}", file=sys.stderr)
