import torch
import torch.nn as nn

class Critic(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(28, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 2*hidden_size)
        self.linear4 = nn.Linear(2*hidden_size, hidden_size)
        self.linear5 = nn.Linear(hidden_size, output_size)

    def forward(self, state, action):
        x = torch.cat([state.float(), action.squeeze(1).float()], 1)
        x = nn.functional.relu(self.linear1(x))
        x = nn.functional.relu(self.linear2(x))
        x = nn.functional.relu(self.linear3(x))
        x = nn.functional.relu(self.linear4(x))
        return self.linear5(x)


class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 2*hidden_size)
        self.linear4 = nn.Linear(2*hidden_size, hidden_size)
        self.linear5 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.float()
        x = nn.functional.relu(self.linear1(x))
        x = nn.functional.relu(self.linear2(x))
        x = nn.functional.relu(self.linear3(x))
        x = nn.functional.relu(self.linear4(x))
        return torch.tanh(self.linear5(x))
