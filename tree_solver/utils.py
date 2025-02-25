import queue

import numpy as np


class SafePriorityQueue:
    """Priority queue that uses a counter to ensure unique keys, sorts the elements as lowest-first."""
    def __init__(self):
        super().__init__()
        self.counter = 0
        self.queue = queue.PriorityQueue()

    def put(self, data, key):
        self.queue.put((key, self.counter, data))
        self.counter += 1

    def get(self):
        if self.queue.empty():
            return None

        return self.queue.get()[-1]

    def empty(self):
        return self.queue.empty()


def hashable_state(state):
    if isinstance(state, np.ndarray):
        return tuple(state.flatten())
    else:
        return state