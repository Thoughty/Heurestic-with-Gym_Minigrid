
from search import graph as G, search as S

init_state = G.get_init_state(7, 2, seed=None)
frontier = G.PriorityQueue()

plan, num_explored = S.search(init_state, frontier)

reward = S.execute(init_state, plan)

print(f'The search explored {num_explored} nodes.')
print(f'The plan got the reward of {reward}.')

input('Enter to quit')
