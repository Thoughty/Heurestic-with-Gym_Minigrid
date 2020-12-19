"""This module contains classes for graph fomulation."""

from __future__ import annotations
from typing import Any, List
from copy import deepcopy
import hashlib
import abc
from queue import Queue as Queue_, PriorityQueue as PriorityQueue_

from gym.spaces.discrete import Discrete
from gym_minigrid.envs.crossing import CrossingEnv
from gym_minigrid.wrappers import FullyObsWrapper
from gym_minigrid.minigrid import MiniGridEnv, Wall


def get_init_state(
        size, num_corssings, seed=None):
    env = CrossingEnv(
        size=size, num_crossings=num_corssings, obstacle_type=Wall, seed=seed)
    env = FullyObsWrapper(env)
    obs = env.reset()
    return MiniGridState(obs, env, False, 0)


class MiniGridState:
    """
    This is just a wrapper class of the MiniGridEnv.
    
    Since MiniGridEnv does not implement __hash__ or __eq__, this makes 
    it hard for us to check duplicate state (for the Graph algorithm).
    """

    def __init__(
            self, obs: dict[str, Any], env: MiniGridEnv,
            done: bool, reward: int = 0) -> None:
        self._obs = obs
        self.env = env
        self.done = done
        self.reward = reward

    @property
    def obs(self) -> dict[str, Any]:
        """
        Return what agent can see.
        
        For this assignment, we use `FullyObsWrapper`, so the agent can see
        the whole environment.
        """
        s = {k: v for k, v in self._obs.items()}
        s['direction'] = self.env.agent_dir
        return s

    @property
    def action_space(self) -> Discrete:
        """Return all possible actions."""
        return self.env.action_space

    def successor(self, action: int) -> MiniGridState:
        """Return a resulting state of the action."""
        new_env = deepcopy(self.env)
        obs, reward, done, _info = new_env.step(action)
        return MiniGridState(obs, new_env, done, reward)
    
    def is_goal(self) -> bool:
        """
        Return a boolean whether we reach the goal.
        
        The goal is defined as `done` and `reward > 0`. `done` means the
        agent reaches the goal or "dead", while `reward` means how much
        score the agent gets (0 most of the time).
        """
        return self.done and self.reward > 0
    
    def __hash__(self) -> int:
        sample_hash = hashlib.sha256()
        to_encode = [
            self.env.grid.encode(), self.env.agent_pos, self.env.agent_dir]
        for item in to_encode:
            sample_hash.update(str(item).encode('utf8'))
        return sample_hash.hexdigest().__hash__()

    def __eq__(self, o: object) -> bool:
        if isinstance(o, MiniGridState):
            return self.__hash__() == o.__hash__()
        else:
            return False

class SearchTreeNode:
    """A node of a search tree."""

    def __init__(
            self, state: MiniGridState, parent: SearchTreeNode,
            action: int, path_cost: float) -> None:
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        # We don't really need to maintain this.
        # self.children = []

    def get_path(self) -> List[SearchTreeNode]:
        """
        Return a list of nodes from the root node to the current node.
        
        This method is useful for getting a solution.
        """
        cur_node = self
        path = [cur_node]
        while cur_node.parent is not None:
            cur_node = cur_node.parent
            path.append(cur_node)
        return list(reversed(path))


class DataStructure(abc.ABC):
    @abc.abstractmethod
    def is_empty(self) -> bool:
        """Return True if empty."""
        raise NotImplementedError()
    
    @abc.abstractmethod
    def add(self, node: SearchTreeNode, priority: float) -> None:
        """Add the node into the frontier."""
        raise NotImplementedError()
    
    @abc.abstractmethod
    def remove(self) -> SearchTreeNode:
        """Return a node from a frontier and remove it."""
        raise NotImplementedError()

    @abc.abstractmethod
    def get_nodes_with_state(
            self, state: MiniGridState) -> List[SearchTreeNode]:
        """Return a list of nodes from a frontier that contain the state."""
        raise NotImplementedError()

    @abc.abstractmethod
    def remove_nodes_with_state(
            self, state: MiniGridState) -> List[SearchTreeNode]:
        """Remove a list of nodes from a frontier that contain the state."""
        raise NotImplementedError()

    @abc.abstractmethod
    def is_in(self, state: MiniGridState) -> bool:
        """Return True if the state is in the frontier."""
        raise NotImplementedError()


class Stack(DataStructure):
    
    def __init__(self) -> None:
        super(Stack, self)
        self.stack = []

    def is_empty(self) -> bool:
        n = 0
        for __, status in self.stack:
            if status != 'D':
                n += 1
        return n == 0

    def add(self, node: SearchTreeNode, priority: float) -> None:
        # priority is ignored.
        # entry is (data, status)
        self.stack.append([node, ''])
    
    def remove(self) -> SearchTreeNode:
        if self.is_empty():
            return None
        node, status = self.stack.pop()
        while status == 'D':
            node, status = self.stack.pop()
        return node

    def is_in(self, state: MiniGridState) -> bool:
        for node, status in self.stack:
            if status != 'D' and state == node.state:
                return True
        return False

    def get_nodes_with_state(
            self, state: MiniGridState) -> List[SearchTreeNode]:
        d_nodes = []
        for entry in self.stack:
            if entry[1] != 'D' and entry[0].state == state:
                d_nodes.append(entry[0])
        return d_nodes

    def remove_nodes_with_state(
            self, state: MiniGridState) -> List[SearchTreeNode]:
        d_nodes = []
        for entry in self.stack:
            if entry[0].state == state:
                entry[-1] = 'D'
                d_nodes.append(entry[0])
        return d_nodes


class Queue(DataStructure):
    def __init__(self) -> None:
        super(Queue, self)
        self.queue = Queue_()

    def is_empty(self) -> bool:
        n = 0
        for __, status in self.queue.queue:
            if status != 'D':
                n += 1
        return n == 0

    def add(self, node: SearchTreeNode, priority: float) -> None:
        # priority is ignored.
        # entry is (data, status)
        self.queue.put([node, ''])
    
    def remove(self) -> SearchTreeNode:
        if self.is_empty():
            return None
        node, status = self.queue.get()
        while status == 'D':
            node, status = self.queue.get()
        return node

    def is_in(self, state: MiniGridState) -> bool:
        for node, status in self.queue.queue:
            if status != 'D' and state == node.state:
                return True
        return False

    def get_nodes_with_state(
            self, state: MiniGridState) -> List[SearchTreeNode]:
        d_nodes = []
        for entry in self.queue.queue:
            if entry[1] != 'D' and entry[0].state == state:
                d_nodes.append(entry[0])
        return d_nodes

    def remove_nodes_with_state(
            self, state: MiniGridState) -> List[SearchTreeNode]:
        d_nodes = []
        for entry in self.queue.queue:
            if entry[0].state == state:
                entry[-1] = 'D'
                d_nodes.append(entry[0])
        return d_nodes

class PriorityQueue(DataStructure):
    def __init__(self) -> None:
        super(PriorityQueue, self)
        self.running = 0
        self.pq = PriorityQueue_()

    def is_empty(self) -> bool:
        n = 0
        for __, __, __, status in self.pq.queue:
            if status != 'D':
                n += 1
        return n == 0

    def add(self, node: SearchTreeNode, priority: float) -> None:
        # the entry is (priority, running, data, status)
        self.pq.put([priority, self.running, node, ''])
        self.running += 1
    
    def remove(self) -> SearchTreeNode:
        # only need the node
        if self.is_empty():
            return None
        __, __, node, status = self.pq.get()
        while status == 'D':
            __, __, node, status = self.pq.get()
        return node

    def is_in(self, state: MiniGridState) -> bool:
        for __, __, node, status in self.pq.queue:
            if status != 'D' and state == node.state:
                return True
        return False

    def get_nodes_with_state(
            self, state: MiniGridState) -> List[SearchTreeNode]:
        d_nodes = []
        for entry in self.pq.queue:
            if entry[3] != 'D' and entry[2].state == state:
                d_nodes.append(entry[2])
        return d_nodes
    
    def remove_nodes_with_state(
            self, state: MiniGridState) -> List[SearchTreeNode]:
        d_nodes = []
        for entry in self.pq.queue:
            if entry[2].state == state:
                entry[-1] = 'D'
                d_nodes.append(entry[2
                ])
        return d_nodes
