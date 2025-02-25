import numpy as np


def manhattan_distance_heuristic(state):
    agent_pos = np.where(state[:, :, 2] == 1)

    return -(agent_pos[0].item() + agent_pos[1].item())