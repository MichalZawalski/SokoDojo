from abc import abstractmethod

import torch


class ValueFunction:
    def __init__(self):
        pass

    @abstractmethod
    def predict_batch(self, state_embeddings_batch, target_embeddings_batch):
        raise NotImplementedError()

    def predict_single(self, state_embedding, target_embedding):
        return self.predict_batch(state_embedding.unsqueeze(0), target_embedding.unsqueeze(0))[0]


class DummyValueFunction(ValueFunction):
    def __init__(self):
        super(DummyValueFunction, self).__init__()

    def predict_batch(self, state_embeddings_batch, target_embeddings_batch):
        return torch.norm(state_embeddings_batch - target_embeddings_batch, dim=1)
