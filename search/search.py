"""Version 1.2"""

from typing import Tuple, List
from copy import deepcopy
import time

from gym_minigrid.minigrid import MiniGridEnv
import numpy as np

from search import graph as G



def heuristic(state: G.MiniGridState) -> float:
    # TODO: 1
    image = state.obs['image'][:, :, 0]
    direction = state.obs['direction']
    distance = 0

    agent_po = np.where(image == 10)
    agent_x = agent_po[0]
    agent_y = agent_po[1]

    goal_po = np.where(image == 8)
    goal_x = goal_po[0]
    goal_y = goal_po[1]
    
    distance += (abs(agent_x - goal_x) + abs(agent_y - goal_y))
    return distance


def search(
        init_state: G.MiniGridState,
        frontier: G.DataStructure) -> Tuple[List[int], int]:
    # TODO: 2
    # this is a random plan
    root = G.SearchTreeNode(init_state, None, -1,  0)
    frontier.add(root, heuristic(root.state))
    num_explored_nodes = 0
    plan = []
    explore = []
    while not frontier.is_empty():
        node = frontier.remove() # mean node that will add to search trees
        if node.state.is_goal():
            path = node.get_path()
            for i in path:
               if i.action >= 0 :
                   plan.append(i.action)
            break
        explore.append(node.state) #path to the goal
        num_explored_nodes += 1

        for j in range(3):#move forward, left, right
            pathcost = j
            successorr = node.state.successor(j) #get successor that expanded from 3 sides of agent
            if not frontier.is_in(successorr) and successorr not in explore :
                X = G.SearchTreeNode(successorr,node,j,pathcost)
                frontier.add(X,heuristic(X.state)) #move to check heuristic func.
              
    return plan, num_explored_nodes


def execute(init_state: G.MiniGridState, plan: List[int], delay=0.5) -> float:
    env = deepcopy(init_state.env)
    env.render()
    sum_reward = 0
    for i, action in enumerate(plan):
        print(f'action no: {i} = {action}')
        time.sleep(delay)
        _obs, reward, done, _info = env.step(action)
        sum_reward += reward
        env.render()
        if done:
            break
    env.close()
    return sum_reward
