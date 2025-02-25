from envs.sokoban.env import SokobanEnv, pretty_print_sokoban_observation
from envs.sokoban.heuristics import simple_manhattan_distance_heuristic, ordering_heuristic
from tree_solver.planners import GreedyPlanner, SearchTreeNode, BestFSPlanner


def solve(env, heuristic, node_limit=10000):
    root_state = env.reset()
    planner = BestFSPlanner(root_state)
    nodes_visited = 0

    while nodes_visited < node_limit:
        current_node = planner.get()
        if nodes_visited == 0:
            print('Next instance:')
            pretty_print_sokoban_observation(current_node.state)

        if current_node is None:
            # No more nodes to explore
            return planner.get_solution_data(None, {'nodes_visited': nodes_visited})

        state = current_node.state
        if env.is_solved(state):
            # Found a solution
            return planner.get_solution_data(current_node, {'nodes_visited': nodes_visited})

        for action in env.get_actions(state):
            next_state = env.get_next_state(state, action)

            if planner.is_seen(next_state):
                # Already visited that state
                continue

            next_node = SearchTreeNode(
                state=next_state,
                value=heuristic(next_state),
                parent_node=current_node,
                action=action
            )
            planner.add(next_node)
            nodes_visited += 1

    return planner.get_solution_data(None, {'nodes_visited': nodes_visited})


if __name__ == '__main__':
    env = SokobanEnv()
    heuristic = ordering_heuristic

    n_trials = 20
    n_successes = 0
    budgets = 0

    for i in range(n_trials):
        solution, search_info = solve(env, heuristic)

        if solution['solved']:
            n_successes += 1
            budgets += search_info['nodes_visited']

        print(heuristic.__name__, f"{n_successes}/{i + 1} successes, average budget: {budgets / max(n_successes, 1)}")