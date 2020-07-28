import torch
import torch.nn as nn
from Actor_Critic import Actor, Critic
from utils import Memory

class DDPGagent:
    def __init__(self, hidden_size, env):
        self.num_states = env.observation_space.shape[0]
        self.num_actions = env.action_space.shape[0]

        self.Actor = Actor(input_size=self.num_states,
                           hidden_size=hidden_size,
                           output_size=self.num_actions).cuda()

        self.Actor_target = Actor(input_size=self.num_states,
                           hidden_size=hidden_size,
                           output_size=self.num_actions).cuda()

        self.Critic = Critic(input_size=self.num_states,
                           hidden_size=hidden_size,
                           output_size=self.num_actions).cuda()

        self.Critic_target = Critic(input_size=self.num_states,
                           hidden_size=hidden_size,
                           output_size=self.num_actions).cuda()


        for target_param, param in zip(self.Actor_target.parameters(), self.Actor.parameters()):
            target_param.data = param.data

        for target_param, param in zip(self.Critic_target.parameters(), self.Critic.parameters()):
            target_param.data = param.data

        self.Memory = Memory(30000)
        self.criterion = nn.MSELoss().cuda()
        self.actor_optimizer = torch.optim.Adam(self.Actor.parameters(), lr=1e-2)
        self.critic_optimizer = torch.optim.Adam(self.Critic.parameters(), lr=1e-1)

    def get_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).cuda()
        action = self.Actor.forward(state)
        action = action.detach().cpu().numpy()
        return action

    def update(self, batch_size):
        states, actions, rewards, next_states, _ = self.Memory.sample(batch_size)
        states = torch.tensor(states).cuda()
        actions = torch.tensor(actions).cuda()
        rewards = torch.tensor(rewards).cuda()
        next_states = torch.tensor(next_states).cuda()

        Q_Value = self.Critic.forward(states, action=actions)
        next_actions = self.Actor_target(next_states)
        next_Q = self.Critic_target.forward(next_states, next_actions.detach())
        Q_prime = rewards + 0.99 * next_Q
        critic_loss = self.criterion(Q_Value, Q_prime)
        policy_loss = -self.Critic.forward(states, self.Actor.forward(states)).mean()

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        for target_param, param in zip(self.Actor_target.parameters(), self.Actor.parameters()):
            target_param.data = (param.data * 1e-2 + target_param.data * (1.0 - 1e-2))

        for target_param, param in zip(self.Critic_target.parameters(), self.Critic.parameters()):
            target_param.data.copy_(param.data * 1e-2 + target_param.data * (1.0 - 1e-2))
