from collections import deque


def topological_sort(G):
    n = len(G)
    ret = []
    degrees = [0] * n

    for i in range(n):
        for e in G[i]:
            degrees[e] += 1

    # 次数が0の点をキューに入れる
    que = deque()
    for i in range(n):
        if degrees[i] == 0:
            que.append(i)

    while que:
        node = que.popleft()
        ret.append(node)

        for e in G[node]:
            degrees[e] -= 1
            if degrees[e] == 0:
                que.append(e)
    return ret
