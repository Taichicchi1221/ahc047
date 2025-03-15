from heapq import heappush, heappop

INF = 1 << 64


def dijkstra(adj, s=0, g=None):  # (ad, j始点, 終点(Noneのときは全点経路))
    n = len(adj)
    dist = [INF] * n
    hq = [(0, s)]  # (distance, node)
    dist[s] = 0
    seen = [False] * n  # ノードが確定済みかどうか
    while hq:
        d, v = heappop(hq)  # ノードを pop する
        seen[v] = True

        if g is not None and v == g:
            break

        if d != dist[v]:
            continue

        for cost, to in adj[v]:  # ノード v に隣接しているノードに対して
            if seen[to]:
                continue

            if dist[v] + cost < dist[to]:
                dist[to] = dist[v] + cost
                heappush(hq, (dist[to], to))
    return dist
