import queue
from abc import abstractmethod

import numpy as np

from tree_solver.utils import SafePriorityQueue, hashable_state


class SearchTreeNode:
    def __init__(self, state, value, parent_node = None, action = None, metadata = None):
        self.state = state
        self.value = value
        self.parent_node = parent_node
        self.action = action
        self.metadata = metadata if metadata is not None else dict()


class Planner:
    """
    General Planner class.

    Manages the search tree, selects consecutive nodes to expand and returns
    the final solution once the problem is solved.
    """
    @abstractmethod
    def __init__(self, root_state):
        self.root_state = root_state

    @abstractmethod
    def add(self, node: SearchTreeNode):
        raise NotImplementedError()

    @abstractmethod
    def get(self):
        raise NotImplementedError()

    @abstractmethod
    def is_seen(self, state):
        raise NotImplementedError()

    @abstractmethod
    def get_solution_data(self, solving_node: SearchTreeNode, search_info: dict):
        raise NotImplementedError()


def get_solution_data(solving_node, search_info, root_node, seen_states):
    search_info['nodes_visited'] = len(seen_states)
    search_info['search_tree'] = root_node.state
    search_info['solving_node'] = solving_node

    if solving_node is None:
        return {'solved': False}, search_info

    action_path = []
    while solving_node is not None:
        action_path.append(solving_node.action)
        solving_node = solving_node.parent_node

    solution = {
        'solved': True,
        'action_path': action_path,
    }

    search_info['solution_length'] = len(action_path)

    return solution, search_info


class GreedyPlanner(Planner):
    """Basic planner that always selects the node with extreme priority."""
    def __init__(self, root_state):
        super().__init__(root_state)

        self.seen_states = set()

        self.nodes_queue = None
        self.create_priority_queue()

        self.root_node = SearchTreeNode(self.root_state, 0, None, None, metadata={'depth': 0})
        self.add(self.root_node)

    def create_priority_queue(self):
        self.nodes_queue: SafePriorityQueue = SafePriorityQueue()

    @abstractmethod
    def get_node_priority(self, node: SearchTreeNode) -> float:
        raise NotImplementedError()

    def add(self, node: SearchTreeNode):
        self.seen_states.add(hashable_state(node.state))

        if node.parent_node is not None:
            depth = node.parent_node.metadata['depth'] + 1
            node.metadata['depth'] = depth

        node_priority = self.get_node_priority(node)
        node.metadata['queue_priority'] = node_priority

        self.nodes_queue.put(data=node, key=node_priority)

    def get(self):
        if self.nodes_queue.empty():
            return None

        return self.nodes_queue.get()

    def is_seen(self, state):
        return hashable_state(state) in self.seen_states

    def get_solution_data(self, solving_node: SearchTreeNode, search_info):
        return get_solution_data(solving_node, search_info, self.root_node, self.seen_states)


class BestFSPlanner(GreedyPlanner):
    def get_node_priority(self, node: SearchTreeNode):
        return -node.value


class AstarPlanner(GreedyPlanner):
    def __init__(self, root_state, value_weight=1.0, depth_weight=1.0):
        self.value_weight = value_weight
        self.depth_weight = depth_weight

        super().__init__(root_state)

    def get_node_priority(self, node: SearchTreeNode):
        return -node.value * self.value_weight + node.metadata['depth'] * self.depth_weight


class BfsPlanner(GreedyPlanner):
    def create_priority_queue(self):
        self.nodes_queue = queue.Queue()

    def add(self, node: SearchTreeNode):
        self.seen_states.add(hashable_state(node.state))
        self.nodes_queue.put(node)

    def get_node_priority(self, node: SearchTreeNode) -> float:
        return 0.0    # BFS does not use priority queue