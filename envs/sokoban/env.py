import gym
import numpy as np

from gym_sokoban.envs.sokoban_env import SokobanEnv as BaseSokobanEnv

from envs.PlanningEnv import PlanningEnv


class SokobanEnv(PlanningEnv):
    def __init__(self, dim_room=(12, 12), max_steps=int(1e6), num_boxes=4, num_gen_steps=None, reset=False):
        self._env = BaseSokobanEnv(dim_room, max_steps, num_boxes, num_gen_steps, reset)

        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = self._env.observation_space

    def step(self, action):
        assert 0 <= action <= 3
        obs, reward, done, info = self._env.step(action + 1, observation_mode="raw")

        return np.stack(obs, axis=-1), reward, done, info

    def reset(self):
        # arr_walls, arr_goals, arr_boxes, arr_player
        return np.stack(self._env.reset(render_mode="raw"), axis=-1)

    def render(self, mode="human"):
        return self._env.render(render_mode=mode)

    def is_solved(self, state):
        return np.array_equal(state[:, :, 1], state[:, :, 2])

    def get_actions(self, state):
        return np.arange(4)

    def restore_state(self, state):
        self._env.room_fixed = 1 - state[:, :, 0] + state[:, :, 1]
        # wall, floor, box_target, box_on_target, box, player
        self._env.room_state = 1 * (
                    (state[:, :, 0] == 0) & (state[:, :, 1] == 0) & (state[:, :, 2] == 0) & (state[:, :, 3] == 0)) + \
                               2 * ((state[:, :, 1] == 1) & (state[:, :, 2] == 0) & (state[:, :, 3] == 0)) + \
                               3 * ((state[:, :, 1] == 1) & (state[:, :, 2] == 1) & (state[:, :, 3] == 0)) + \
                               4 * ((state[:, :, 1] == 0) & (state[:, :, 2] == 1) & (state[:, :, 3] == 0)) + \
                               5 * (state[:, :, 3] == 1)
        self._env.box_mapping = None

        self._env.player_position = np.argwhere(state[:, :, 3] == 1)[0]
        self._env.num_env_steps = 0
        self._env.reward_last = 0
        self._env.boxes_on_target = np.sum(state[:, :, 1] * state[:, :, 2])

    def get_next_state(self, state, action):
        self.restore_state(state)
        return self.step(action)[0]


def pretty_print_sokoban_observation(observation):
    for row in observation:
        for cell in row:
            if cell[0] == 1:
                print("# ", end="")
            elif cell[2] == 1:
                if cell[1] == 0:
                    print("□ ", end="")
                else:
                    print("☒ ", end="")
            elif cell[3] == 1:
                if cell[1] == 0:
                    print("a ", end="")
                else:
                    print("@ ", end="")
            else:
                if cell[1] == 0:
                    print("  ", end="")
                else:
                    print("x ", end="")
        print()


if __name__ == '__main__':
    # Test state restoration
    env1 = SokobanEnv()
    env2 = SokobanEnv()

    while True:
        new_obs = env1.reset()

        for i in range(100):
            obs = new_obs
            action = env1.action_space.sample()
            new_obs, reward, done, info = env1.step(action)
            pretty_print_sokoban_observation(new_obs)

            sim_state = env2.get_next_state(obs, action)

            # Check if the two states are the same. If not, pretty_print both of them
            if not np.array_equal(new_obs, sim_state):
                print("Different states!")
                pretty_print_sokoban_observation(new_obs)
                pretty_print_sokoban_observation(sim_state)
                print(env1._env.room_fixed)
                print(env2._env.room_fixed)
                print(env1._env.room_state)
                print(env2._env.room_state)
                assert False

            if not np.array_equal(env1._env.room_fixed, env2._env.room_fixed):
                print("Different room_fixed!")
                print(env1._env.room_fixed)
                print(env2._env.room_fixed)
                assert False

            if not np.array_equal(env1._env.room_state, env2._env.room_state):
                print("Different room_state!")
                print(env1._env.room_state)
                print(env2._env.room_state)
                assert False

        print('Episode ok')




