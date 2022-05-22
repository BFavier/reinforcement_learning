from os import environ
import torch
import torch.nn.functional as F
from ._templates import Agent
from typing import List, Tuple
import matplotlib.pyplot as plt


def batchify(*data, batch_size: int, n_batches: int, shuffle: bool = True) -> Tuple[torch.Tensor]:
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
    N = len(data[0])
    indexes = torch.randperm(N) if shuffle else torch.arange(N)
    for i in range(n_batches):
        if i*batch_size > N:
            break
        yield tuple(d[indexes[i*batch_size:(i+1)*batch_size]] for d in data)


def train_loop(agent: Agent, learning_rate: 1.0E-3, n_epochs: int = 100,
               n_updates: int = 100, n_batches: int = 10, batch_size: int = 100,
               epsilon: float = 0., n_replays: int = 10000):
    """
    train loop of the model
    """
    optimizer = torch.optim.Adam(agent.parameters(), lr=learning_rate)
    state_replays = None
    action_replays = None
    next_state_replays = None
    reward_replays = None
    loss_history = []
    try:
        for update in range(n_updates):
            loss_history.append([])
            print(f"\t\tUpdate {update}:")
            # making the model play against itself for one turn of each player
            print("playing ...")
            agent.eval()
            state = agent.interpreter.initial_state()
            states, actions, rewards, next_states = [], [], [], []
            with torch.no_grad():
                while not agent.interpreter.game_is_over(state):
                    # first player plays
                    action = agent.play(state, epsilon=epsilon)
                    next_state = agent.interpreter.apply(state, action)
                    next_state = agent.interpreter.change_turn(next_state)
                    reward = agent.interpreter.rewards(state, action, next_state)
                    # add the actions in the replay
                    states.append(state)
                    actions.append(action)
                    rewards.append(reward)
                    next_states.append(next_state)
                    # change turn
                    state = next_state
            # append new plays to replay
            replays = [state_replays, action_replays, next_state_replays, reward_replays]
            plays: list[list] = [states, actions, next_states, rewards]
            for replay, _list in zip(replays, plays):
                if replay is not None:
                    _list.insert(0, replay)
            state_replays = torch.cat(states)[-n_replays:]
            action_replays = torch.cat(actions)[-n_replays:]
            next_state_replays = torch.cat(next_states)[-n_replays:]
            reward_replays = torch.cat(rewards)[-n_replays:]
            # looping on epochs
            for epoch in range(n_epochs):
                optimizer.zero_grad()
                losses = []
                for state, action, reward, next_state in batchify(state_replays, action_replays, reward_replays, next_state_replays, n_batches=n_batches, batch_size=batch_size):
                    agent.train()
                    q = agent.q(state, action)
                    agent.eval()
                    enemy_q = agent.Q(next_state)
                    loss = F.mse_loss(q+reward.to(q.device), -agent.gamma*enemy_q)**0.5
                    loss.backward()
                    losses.append(loss.item())
                loss = sum(losses)/len(losses)
                print(f"Epoch {epoch}: loss = {loss:.3g}")
                loss_history[-1].append(loss)
                optimizer.step()
    except KeyboardInterrupt as e:
        pass
    return loss_history


def play_against(agent: Agent, player_starts: bool = True):
    """
    A small loop to play against a trained agent
    """
    # player_turn = player_starts
    # if player_turn:
    #     print(environment)
    # while True:
    #     if player_turn:
    #         while True:
    #             try:
    #                 action = Act.from_string(input("your action: "))
    #             except Exception as e:
    #                 continue
    #             break
    #         environment = environment.apply(action)
    #     else:
    #         environment = environment.change_turn()
    #         print(agent.Q(environment).unsqueeze(0))
    #         action, environment, _ = agent.play(environment)
    #         environment = environment.change_turn()
    #         print(f"agent action: {action}")
    #     print(environment)
    #     if environment.game_is_over():
    #         if environment.current_player_won() or environment.other_player_won():
    #             if player_turn:
    #                 print("You win !")
    #             else:
    #                 print("You lose ...")
    #         else:
    #             print("draw")
    #         break
    #     player_turn = not player_turn


def plot_loss(loss_history: List[List[float]]):
    f, ax = plt.subplots()
    i = 0
    for loss in loss_history:
        j = i + len(loss)
        ax.scatter(range(i, j), loss)
        i = j
    ax.set_yscale("log")
    ax.set_xlabel("epochs")
    ax.set_ylabel("loss")
    plt.show()
