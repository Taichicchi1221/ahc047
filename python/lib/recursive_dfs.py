import sys

sys.setrecursionlimit(10**6)

try:
    import pypyjit

    pypyjit.set_param("max_unroll_recursion=-1")
except ModuleNotFoundError:
    pass


def dfs(c, p):
    # global変数設定
    global G

    # 行きがけ処理

    for g in G[c]:
        if g == p:
            continue
        dfs(g, c)

        # 帰りがけ処理
