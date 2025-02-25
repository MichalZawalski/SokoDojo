from abc import abstractmethod

import gym


class PlanningEnv(gym.Env):
    @abstractmethod
    def step(self, action):
        raise NotImplementedError()

    @abstractmethod
    def reset(self):
        raise NotImplementedError()

    @abstractmethod
    def render(self, mode="human"):
        raise NotImplementedError()

    @abstractmethod
    def is_solved(self, state):
        raise NotImplementedError()

    @abstractmethod
    def get_actions(self, state):
        raise NotImplementedError()

    @abstractmethod
    def get_next_state(self, state, action):
        raise NotImplementedError()
