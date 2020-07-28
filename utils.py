import numpy
import random
from collections import deque

class Memory:
    def __init__(self, size):
        self.buffer = deque(maxlen=size)

    def push(self, state, action, reward, next_state, done):
        experience = (state, action, numpy.array([reward]), next_state,done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        State = []
        Action = []
        Reward = []
        Next_state = []
        Done = []
        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, action, reward, next_state, done = experience
            State.append(state)
            Action.append(action)
            Reward.append(reward)
            Next_state.append(next_state)
            Done.append(done)

        return State, Action, Reward, Next_state, Done

    def __len__(self):
        return len(self.buffer)

