from reinforcement_learning.tic_tac_toe import Agent, Environment, Interpreter
import torch
import IPython

player = Agent()
states = Environment.initial_state()

Q = player.Q_with_symetries(states)
game_is_over = Environment.game_is_over(states)
running_states = states[~game_is_over]
actions = player.play(running_states, Q)
new_states = Environment.next_states(running_states, actions)

IPython.embed()