import torch
from ._templates import Agent, Environment, Interpreter, Action
from typing import Type, List
import matplotlib.pyplot as plt


def batchify(data: torch.Tensor, batch_size: int, n_batches: int):
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
    N = len(data)
    for i in range(n_batches):
        if i*batch_size > N:
            break
        yield data[i*batch_size:(i+1)*batch_size]


def train_loop(agent: Agent, Env: Type[Environment], Inter: Type[Interpreter],
               optimizer: torch.optim.Optimizer, learning_rate: 1.0E-3,
               n_epochs: int = 100, n_updates: int = 100,
               n_batchs: int = 10, batch_size: int = 100,
               epsilon: float = 0., replay_history_size: int = 10000,
               loss_history: list = []):
    """
    train loop of the model
    """
    loss_history = list(loss_history)
    replay_history = Env()
    agent.train()
    try:
        for update in range(n_updates):
            print(f"\t\tUpdate {update}:")
            frozen_agent = agent.copy()
            frozen_agent.eval()
            # looping epochs
            update_loss = []
            for epoch in range(n_epochs):
                optimizer.zero_grad()
                replay_history = replay_history.sample(replay_history_size)
                environment = replay_history[:n_batchs*batch_size]
                # updating model parameters
                batch_losses = []
                for env in batchify(environment, batch_size, n_batchs):
                    # first player plays
                    action_A, environment_A, q_A = agent.play(env, epsilon=epsilon)
                    rewards_A = Inter.rewards(env, action_A, environment_A)
                    replay_history.extend(environment_A)
                    # second player plays
                    next_environment = environment_A.change_turn()
                    action_B, environment_B, q_B = agent.play(next_environment)
                    rewards_B = Inter.rewards(next_environment, action_B, environment_B)
                    replay_history.extend(environment_B)
                    # the actual reward is the difference of both rewards
                    next_environment = environment_B.change_turn()
                    rewards = rewards_A - rewards_B
                    with torch.no_grad():
                        # calculate the Q value of the final state
                        _, _, next_q = frozen_agent.play(next_environment)
                    # calculating loss
                    loss = torch.nn.functional.mse_loss(q_A, agent.gamma * next_q + rewards)
                    loss.backward()
                    batch_losses.append(loss.item())
                update_loss.append(sum(batch_losses) / len(batch_losses))
                print(f"Epoch {epoch}: loss = {update_loss[-1]:.3g}")
                optimizer.step()
            loss_history.append(update_loss)
    except KeyboardInterrupt as e:
        pass
    return loss_history


def play_against(agent: Agent, environment: Environment, Act: Type[Action], player_starts: bool = True):
    """
    A small loop to play against a trained agent
    """
    player_turn = player_starts
    states = environment.initial_state()
    game_is_over = environment.game_is_over(states)
    while not game_is_over[0]:
        if player_turn:
            print(environment)
            while True:
                try:
                    action = Act.from_string(input())
                except Exception as e:
                    continue
                break
            environment = environment.apply(action)
            if environment.game_is_over():
                print("You win !")
        else:
            environment.change_turn()
            action, environment, _ = agent.play(environment, epsilon=agent.epsilon)
            environment.change_turn()
            if environment.game_is_over():
                print("You lose ...")
        print(action)


def plot_loss(loss_history: List[List[float]]):
    f, ax = plt.subplots()
    for loss in loss_history:
        ax.scatter(range(len(loss)), loss)
    ax.set_yscale("log")
    ax.set_xlabel("epochs")
    ax.set_ylabel("loss")
    plt.show()
