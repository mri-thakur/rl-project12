import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from gridworld import GridWorld
from collections import deque

env = GridWorld()
print(f"Blocked: {len(env.blocked)}, Valid: {len(env.get_all_states())}")

visited = set()
queue = deque([env.start])
visited.add(env.start)
parent = {env.start: None}
found = False

while queue:
    state = queue.popleft()
    if state == env.goal:
        path = []
        s = state
        while s is not None:
            path.append(s)
            s = parent[s]
        print(f"Path found, length={len(path)-1}")
        found = True
        break
    for action in range(4):
        dr, dc = env.action_deltas[action]
        ns = (state[0]+dr, state[1]+dc)
        if (0 <= ns[0] < 20 and 0 <= ns[1] < 20 and
                ns not in env.blocked and ns not in visited):
            visited.add(ns)
            parent[ns] = state
            queue.append(ns)

if not found:
    print("NO PATH")
