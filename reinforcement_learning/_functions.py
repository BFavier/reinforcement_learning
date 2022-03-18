import copy
from os import environ
import torch
from ._templates import Agent, Environement, Interpreter
from typing import Tuple


def batchify(data: Tuple[torch.Tensor], batch_size: int, n_batches: int, shuffle: bool = True):
    """
    yields 'n_batches' batches of size 'batch_size' of the given data

    Parameters
    ----------
    data : tuple of torch.Tensor
        tensors of shape (N, *) to batchify
    batch_size : int
        size of the batches
    n_batches : int
        number of batches to yield
    shuffle : bool
        if True, the data are shuffled before beeing batched
    """
    n = n_batches*batch_size
    N = max(d.shape[0] for d in data)
    indexes = torch.randperm(n) if shuffle else torch.arange(n, dtype=torch.long)
    for i in range(n_batches):
        if i*batch_size > N:
            break
        yield tuple(d[indexes[i*batch_size:(i+1)*batch_size]] for d in data)


def train_loop(agent: Agent, environement: Environement, Interpreter: Interpreter,
               optimizer: torch.optim.Optimizer, learning_rate: 1.0E-3,
               n_epochs: int = 100, n_updates: int = 100,
               n_batchs: int = 10, batch_size: int = 100):
    agent.train()
    states = environement.initial_state()
    for update in range(n_updates):
        print(f"\t\tUpdate {update}:")
        reference = copy.deepcopy(agent)
        reference.eval()
        Q = reference(states)
        actions, new_states = reference.play(states, Q)
        for epoch in range(n_epochs):
            for data in batchify(states, batch_size, n_batchs):
                pass


def play_against(agent: Agent, environement: Environement, player_starts: bool = True):
    """
    A small loop to play against a trained agent
    """
    player_turn = player_starts
    states = environement.initial_state()
    game_is_over = environement.game_is_over(states)
    while not game_is_over[0]:
        if player_turn:
            environement.draw(states[0])
            
