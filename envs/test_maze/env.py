import copy

import numpy as np
import gym
from gym import spaces


def get_small_board():
    # Define a small fixed layout (0: empty, 1: wall)
    return np.array([
        [0, 0, 1, 0, 1],
        [1, 0, 1, 0, 0],
        [0, 0, 0, 0, 1],
        [0, 1, 1, 0, 0],
        [0, 0, 1, 1, 0],
    ])


def get_large_board():
    # Define a large fixed layout (0: empty, 1: wall)
    return np.array([
        [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
        [0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0],
        [0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        [1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
        [0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0]
    ])


def get_random_board(shape=(10, 10)):
    # Sample a random maze layout
    rows, cols = shape[0] + 2, shape[1] + 2

    # Ensure dimensions are odd for proper maze generation
    assert rows % 2 == 1 and cols % 2 == 1

    # Initialize maze: all cells are walls
    maze = np.ones((rows, cols), dtype=int)

    def carve_passage(x, y):
        """Carves a passage starting from (x, y) using DFS."""
        directions = [(2, 0), (0, 2), (-2, 0), (0, -2)]
        np.random.shuffle(directions)  # Shuffle to create random paths

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 1 <= nx < rows - 1 and 1 <= ny < cols - 1:
                if maze[nx, ny] == 1:  # If it's a wall
                    maze[x + dx // 2, y + dy // 2] = 0  # Carve path in between
                    maze[nx, ny] = 0  # Carve path at (nx, ny)
                    carve_passage(nx, ny)

    # Start maze generation from a random odd cell
    start_x, start_y = np.random.choice(range(1, rows, 2)), np.random.choice(range(1, cols, 2))
    maze[start_x, start_y] = 0  # Starting cell
    carve_passage(start_x, start_y)

    return maze[1:-1, 1:-1]


class MazeEnv(gym.Env):
    def __init__(self, layout='large'):
        super(MazeEnv, self).__init__()

        self.layout = layout
        self.board = self.sample_board()

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=1, shape=(*self.board.shape, 3), dtype=np.float32)
        self.agent_state = None
        self.goal_state = [0, 0]

    def sample_board(self):
        if self.layout == 'small':
            return get_small_board()
        elif self.layout == 'large':
            return get_large_board()
        else:
            assert self.layout == 'random'
            return get_random_board()

    def reset(self):
        self.board = self.sample_board()

        while True:
            self.agent_state = [np.random.randint(0, len(self.board)),
                                np.random.randint(0, len(self.board[0]))]
            if self.board[tuple(self.agent_state)] == 0:
                break

        return self.get_state()

    @staticmethod
    def _get_raw_moves():
        return {
            0: (-1, 0),  # Up
            1: (0, 1),   # Right
            2: (1, 0),   # Down
            3: (0, -1)   # Left
        }

    @staticmethod
    def reverse_action(action):
        return (action + 2) % 4

        # moves = self._get_raw_moves()
        # for k, v in moves.items():
        #     if (v[0], v[1]) == (-moves[action][0], -moves[action][1]):
        #         return k

    def step(self, action):
        """Execute an action and update the state."""
        row, col = self.agent_state

        # Define moves: (up, right, down, left)
        moves = {
            0: (-1, 0),  # Up
            1: (0, 1),   # Right
            2: (1, 0),   # Down
            3: (0, -1)   # Left
        }

        # Compute new position
        if action not in moves:
            raise ValueError(f"Invalid action {action}")

        new_row, new_col = row + moves[action][0], col + moves[action][1]

        # Check if the move is within bounds and not into a wall
        if 0 <= new_row < self.board.shape[0] and 0 <= new_col < self.board.shape[1]:
            if self.board[new_row, new_col] == 0:  # Ensure it's not a wall
                self.agent_state = [new_row, new_col]

        # Check if the goal is reached
        done = np.array_equal(self.agent_state, self.goal_state)
        reward = 1 if done else 0

        return self.get_state(), reward, done, {}

    def get_state(self, agent_pos=None):
        """Return the current state as a 3D array."""
        state = np.zeros((*self.board.shape, 3), dtype=np.float32)

        # Layer 0: Empty cells
        state[:, :, 0] = (self.board == 0).astype(np.float32)

        # Layer 1: Walls
        state[:, :, 1] = (self.board == 1).astype(np.float32)

        if agent_pos is None:
            agent_pos = self.agent_state

        # Layer 2: Agent's position
        agent_row, agent_col = agent_pos
        state[agent_row, agent_col, 2] = 1.0

        return copy.deepcopy(state)

    @staticmethod
    def obs_to_state(self, obs):
        return np.where(obs[:, :, 2] == 1)[0][0], np.where(obs[:, :, 2] == 1)[1][0]

    def render(self, mode='human'):
        """Render the environment."""
        board_with_agent = self.board.copy()
        agent_row, agent_col = self.agent_state
        board_with_agent[agent_row, agent_col] = 2  # Represent agent as 2

        print("\n".join(" ".join(str(cell) for cell in row) for row in board_with_agent))
        print()

    def sample_state(self):
        while True:
            state = (np.random.randint(0, self.board.shape[0]), np.random.randint(0, self.board.shape[1]))
            if self.board[state] == 0:
                break
        return state


def pretty_print_maze_state(state):
    print(" ".join(["X"] * (state.shape[1] + 2)))
    for row in state:
        print(" ".join(["X"] + ["X" if cell[1] == 1 else "A" if cell[2] == 1 else " " for cell in row] + ["X"]))
    print(" ".join(["X"] * (state.shape[1] + 2)))


def generate_random_data(n_episodes=100, episode_length=100):
    env = MazeEnv()
    episodes = []

    for _ in range(n_episodes):
        state = env.reset()
        episode = []

        for i in range(episode_length):
            action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action)

            episode.append((state, action, reward))

            state = next_state

        episode.append((state, None, None))
        episodes.append(episode)

    return episodes


def generate_good_data(env, n_episodes=100):
    episodes = []

    for _ in range(n_episodes):
        current_state = copy.deepcopy(env.reset())
        episode = []
        prev_action = None

        while True:
            step_taken = False

            for action in np.random.permutation(env.action_space.n):
                if env.reverse_action(action) == prev_action:
                    continue

                next_state, reward, done, _ = env.step(action)

                if np.array_equal(next_state, current_state):
                    continue

                step_taken = True
                break

            if not step_taken:
                break

            episode.append((current_state, action, reward))
            current_state = copy.deepcopy(next_state)
            prev_action = action

        episode.append((current_state, None, None))
        episodes.append(episode)

    return episodes


if __name__ == '__main__':
    print(get_random_board((9, 15)))