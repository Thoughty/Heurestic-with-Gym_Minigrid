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

    agent_postion = np.where(image == 10)
    AgentX = agent_postion[0]
    AgentY = agent_postion[1]


    goal_position = np.where(image == 8)
    GoalX = goal_position[0]
    GoalY = goal_position[1]

    distance += np.sqrt(((GoalX-AgentX)**2) + ((GoalY-AgentY)**2))
    return distance


def search(
        init_state: G.MiniGridState,
        frontier: G.DataStructure) -> Tuple[List[int], int]:
    # TODO: 2
    # this is a random plan
    root = G.SearchTreeNode(init_state, None, -1,  0)
    frontier.add(root, heuristic(root.state))
    explored_set = []
    num_explored_nodes = 0
    plan = []
    while not frontier.is_empty():
        if(frontier.is_empty()):
            break

        Choose = frontier.remove()

        if(Choose.state.is_goal()):
            track = Choose.get_path()
            for way in track:
                if(way.action >= 0):
                    plan.append(way.action)
            break

        explored_set.append(Choose.state)
        num_explored_nodes += 1

        for act in range(3):
            cost = act
            SuccessorState = Choose.state.successor(act)

            if((not frontier.is_in(SuccessorState)) and (SuccessorState not in explored_set)):
                SuccessorState = G.SearchTreeNode(SuccessorState, Choose, act, cost)
                frontier.add(SuccessorState, heuristic(SuccessorState.state))

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
