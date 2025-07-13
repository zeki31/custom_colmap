from collections import defaultdict


class UnionFind:
    def __init__(self, n):
        self.par = list(range(n))
        self.rank = [1] * n

    def root(self, x):
        if self.par[x] == x:
            return x
        self.par[x] = self.root(self.par[x])
        return self.par[x]

    def union(self, x, y):
        rx, ry = self.root(x), self.root(y)
        if rx == ry:
            return

        if self.rank[rx] < self.rank[ry]:
            self.par[rx] = ry
        else:
            self.par[ry] = rx
            if self.rank[rx] == self.rank[ry]:
                self.rank[rx] += 1

    def same(self, x, y):
        return self.root(x) == self.root(y)


def merge_pairs(pairs: set[tuple[int, int]]) -> set[tuple[int, ...]]:
    # 1) Group second‐elements by the first‐element
    groups = defaultdict(set[int])
    for a, b in pairs:
        groups[a].add(b)

    # 2) Build the merged tuples
    merged = {(a,) + bs for a, bs in groups.items()}

    return merged
