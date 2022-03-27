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
            replay_history = replay_history.sample(replay_history_size)
            print(f"\t\tUpdate {update}: {len(replay_history)} replays")
            frozen_agent = agent.copy()
            frozen_agent.eval()
            # looping epochs
            update_loss = []
            for epoch in range(n_epochs):
                optimizer.zero_grad()
                environment = replay_history[:n_batchs*batch_size]
                # updating model parameters
                batch_losses = []
                for env in batchify(environment, batch_size, n_batchs):
                    # first player plays
                    action, new_environment, q = agent.play(env, epsilon=epsilon)
                    rewards = Inter.rewards(env, action, new_environment)
                    new_environment = new_environment.change_turn()
                    replay_history = replay_history.extend(new_environment[~new_environment.game_is_over()])
                    # second player plays
                    with torch.no_grad():
                        # calculate the Q value of the final state
                        _, _, next_q = frozen_agent.play(new_environment)
                        next_q = torch.where(new_environment.game_is_over().to(next_q.device),
                                             torch.zeros_like(next_q), next_q)
                    # calculating loss
                    loss = torch.nn.functional.mse_loss(q, rewards.to(next_q.device) - agent.gamma * next_q)**0.5
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
    if player_turn:
        print(environment)
    while True:
        if player_turn:
            while True:
                try:
                    action = Act.from_string(input("your action: "))
                except Exception as e:
                    continue
                break
            environment = environment.apply(action)
        else:
            environment = environment.change_turn()
            print(agent.Q(environment).unsqueeze(0))
            action, environment, _ = agent.play(environment)
            environment = environment.change_turn()
            print(f"agent action: {action}")
        print(environment)
        if environment.game_is_over():
            if environment.current_player_won() or environment.other_player_won():
                if player_turn:
                    print("You win !")
                else:
                    print("You lose ...")
            else:
                print("draw")
            break
        player_turn = not player_turn


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
