from reinforcement_learning.tic_tac_toe import Agent, Environment, Action
from reinforcement_learning import play_against
import torch
import pathlib

path = pathlib.Path(__file__).parent
agent = torch.load(path / "model.pty")

play_against(agent, Environment(), Action, player_starts=True)

import IPython
IPython.embed()