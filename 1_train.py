from reinforcement_learning import train_loop, plot_loss
from reinforcement_learning.tic_tac_toe import Agent, Environment, Interpreter
import torch
import pathlib

path = pathlib.Path(__file__).parent

agent, environment, interpreter = Agent(), Environment(), Interpreter()
optimizer = torch.optim.Adam(agent.parameters())

if torch.cuda.device_count() > 0:
    agent.to("cuda:0")

# training model
loss_history = train_loop(agent, Environment, Interpreter, optimizer,
                          learning_rate=1.0E-3, n_updates=1000, n_epochs=10,
                          n_batchs=1, batch_size=1000, epsilon=0.1)

# saving model
agent.to("cpu")
torch.save(agent, path / "model.pty")

# plotting loss
plot_loss(loss_history)
