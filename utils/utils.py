# %%
import math

import numpy as np
import torch
import torch.nn as nn



# %%
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len, dropout=0.1):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float)
        position = position.unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 != 0:
            pe[:, 1::2] = torch.cos(position * div_term)[:, :-1]
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


def discretize_cont_var(var, min_val, scale):
    discrete_var = (var - min_val) // scale
    discrete_var[discrete_var < 0] = 0
    discrete_var[discrete_var > 10] = 9
    return discrete_var.to(torch.long)


def cosine_cyclical_annealing(n_iter, start=0.0, stop=1.0, n_cycle=4, ratio=0.5):
    """Cosine Cyclical Annealing with multiple cycles."""
    iterations = np.arange(n_iter)
    cycle_length = n_iter / n_cycle
    up_length = cycle_length * ratio

    annealing_values = np.zeros(n_iter)

    for i in range(n_iter):
        cycle_idx = i % cycle_length
        if cycle_idx < up_length:
            # Cosine schedule for the up phase
            annealing_values[i] = (
                start + (stop - start) * (1 - np.cos(np.pi * cycle_idx / up_length)) / 2
            )
        else:
            # Hold at stop
            annealing_values[i] = stop

    return iterations, annealing_values


def compute_grad_norm(model):
    total_norm = 0
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm**0.5
    return total_norm



