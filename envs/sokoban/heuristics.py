from collections import defaultdict
from itertools import permutations

import numpy as np

from envs.sokoban.env import pretty_print_sokoban_observation, SokobanEnv


def boxes_on_goals_heuristic(observation):
    return -np.sum(observation[:, :, 1] * (1 - observation[:, :, 2]))


def simple_manhattan_distance_heuristic(state):
    agent_pos = np.where(state[:, :, 3] == 1)
    agent_pos = agent_pos[0][0], agent_pos[1][0]
    boxes = np.where(state[:, :, 2] == 1)
    boxes = list(zip(boxes[0], boxes[1]))
    goals = np.where(state[:, :, 1] == 1)
    goals = list(zip(goals[0], goals[1]))

    total_distance = 0
    for box in boxes:
        min_distance = int(1e6)
        for goal in goals:
            distance = abs(box[0] - goal[0]) + abs(box[1] - goal[1])
            min_distance = min(min_distance, distance)
        total_distance += min_distance

    # add agent distance to the closest box
    min_distance = int(1e6)
    for box in boxes:
        distance = abs(agent_pos[0] - box[0]) + abs(agent_pos[1] - box[1])
        min_distance = min(min_distance, distance)

    return -(total_distance + min_distance)


def ordering_heuristic(state):
    agent_pos = np.where(state[:, :, 3] == 1)
    agent_pos = agent_pos[0][0], agent_pos[1][0]
    boxes = np.where(state[:, :, 2] == 1)
    boxes = list(zip(boxes[0], boxes[1]))
    goals = np.where(state[:, :, 1] == 1)
    goals = list(zip(goals[0], goals[1]))

    best_distance = int(1e6)

    # choose the ordering of boxes and goals that minimizes the total distance
    for boxes_order in permutations(boxes):
        for goal_order in permutations(goals):
            total_distance = 0
            current_agent = agent_pos

            for box, goal in zip(boxes_order, goal_order):
                if box == goal:
                    continue

                total_distance += abs(current_agent[0] - box[0]) + abs(current_agent[1] - box[1]) - 1
                total_distance += abs(box[0] - goal[0]) + abs(box[1] - goal[1])
                current_agent = goal

            best_distance = min(best_distance, total_distance)

    return -best_distance


if __name__ == '__main__':
    env = SokobanEnv()
    obs = env.reset()
    pretty_print_sokoban_observation(obs)
    action_lookup = {'u': 0, 'd': 1, 'l': 2, 'r': 3}
    distances = defaultdict(list)

    while True:
        for heuristic in [boxes_on_goals_heuristic, simple_manhattan_distance_heuristic, ordering_heuristic]:
            h = heuristic(obs)
            print(heuristic.__name__, h)
            distances[heuristic.__name__].append(h)

        while True:
            a = input("Enter action: ")[0]
            if a in action_lookup:
                break
        action = action_lookup[a]
        obs, reward, done, info = env.step(action)
        pretty_print_sokoban_observation(obs)

        if done:
            print("Done!")
            break

    for heuristic, values in distances.items():
        print(heuristic, values)