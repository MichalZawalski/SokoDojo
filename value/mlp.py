import torch

from value.base import ValueFunction


class MlpValue(torch.nn.Module):
    """Conditional value function trained with an imitation learning objective."""

    def __init__(self, embedding_dim=2, n_layers=2, hidden_dim=10):
        super(MlpValue, self).__init__()

        self.net = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim * 2, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1)
        )

    def forward(self, state_embeddings_batch, target_embeddings_batch):
        return self.net(torch.cat([state_embeddings_batch, target_embeddings_batch], dim=-1)).squeeze(dim=-1)

    def predict_batch(self, state_embeddings_batch, target_embeddings_batch):
        return self(state_embeddings_batch, target_embeddings_batch)

    def predict_single(self, state_embedding, target_embedding):
        return self.predict_batch(state_embedding.unsqueeze(0), target_embedding.unsqueeze(0))[0]