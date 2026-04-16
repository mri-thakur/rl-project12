"""Find a wall configuration giving optimal path = 36 steps."""
from collections import deque

size = 20; start = (1,1); goal = (18,18)
deltas = [(-1,0),(1,0),(0,-1),(0,1)]

def bfs(blocked):
    q = deque([start]); parent = {start: None}
    while q:
        s = q.popleft()
        if s == goal:
            p = []; c = s
            while c is not None: p.append(c); c = parent[c]
            p.reverse(); return p
        for dr, dc in deltas:
            ns = (s[0]+dr, s[1]+dc)
            if 0<=ns[0]<size and 0<=ns[1]<size and ns not in blocked and ns not in parent:
                parent[ns] = s; q.append(ns)
    return None

# Start with border walls
blocked = set()
for i in range(size):
    blocked.add((0,i)); blocked.add((size-1,i))
    blocked.add((i,0)); blocked.add((i,size-1))

# Strategy: block cells on the shortest path, but skip if it would
# disconnect the grid (check that path still exists after blocking)
while True:
    path = bfs(blocked)
    if path is None:
        print("BROKEN")
        break
    opt = len(path) - 1
    if opt >= 36:
        print(f"Optimal = {opt}")
        internal = sorted([c for c in blocked if 0<c[0]<19 and 0<c[1]<19])
        print(f"Internal walls ({len(internal)}):")
        print(internal)
        break
    # Try blocking cells on path (skip start, goal, and first/last few)
    added = False
    for cell in path[2:-2]:
        test = blocked | {cell}
        if bfs(test) is not None:
            blocked.add(cell)
            added = True
            break
    if not added:
        print(f"Stuck at optimal={opt}")
        break
