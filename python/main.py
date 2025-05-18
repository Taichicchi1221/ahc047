# author:  Taichicchi
# created: 18.05.2025 19:00:00


from collections import Counter

import numpy as np


class LLM:
    def __init__(self, N, M, L, S_list, P_list):
        self.N = N
        self.M = M
        self.L = L
        self.S_list = S_list
        self.P_list = P_list
        self.C = []  # 状態ごとの文字
        self.A = []  # 遷移確率行列（整数パーセント）

    def greedy_design(self):
        # 1. 文字割り当て: Sの先頭文字の頻度順でM個選ぶ
        freq = Counter(s[0] for s in self.S_list)
        most_common = [c for c, _ in freq.most_common()]
        # 'a'-'f'全てカバー
        for c in "abcdef":
            if c not in most_common:
                most_common.append(c)
        # M=12個まで繰り返し使う
        self.C = (most_common * ((self.M + 5) // 6))[: self.M]

        # 2. 遷移確率: 単純に均等
        self.A = []
        base = 100 // self.M
        for i in range(self.M):
            row = [base] * self.M
            # 端数調整（最初の要素にまとめる）
            row[0] += 100 - sum(row)
            self.A.append(row)

    def print_model(self):
        for i in range(self.M):
            row = [self.C[i]] + list(map(str, self.A[i]))
            print(" ".join(row))


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


class Evaluator:
    def __init__(self, model):
        self.model = model

    def compute_score(self):
        N, M, L, S_list, P_list = (
            self.model.N,
            self.model.M,
            self.model.L,
            self.model.S_list,
            self.model.P_list,
        )
        C = self.model.C
        A = self.model.A
        total_score = 0.0
        for S, P in zip(S_list, P_list):
            prob = compute_word_probability(S, L, C, A)
            total_score += P * prob
        return round(total_score)


# ====== 使用例（入力は標準入力から） ======
if __name__ == "__main__":
    import sys

    input = sys.stdin.readline

    N, M, L = map(int, input().split())
    S_list = []
    P_list = []
    for _ in range(N):
        s, p = input().split()
        S_list.append(s)
        P_list.append(int(p))

    model = LLM(N, M, L, S_list, P_list)
    model.greedy_design()
    model.print_model()  # 出力フォーマット

    # 評価（スコア計算）
    evaluator = Evaluator(model)
    print("score ", evaluator.compute_score(), file=sys.stderr)
