import gym
from DDPG import DDPGagent

env = gym.make('BipedalWalker-v3')
agent = DDPGagent(hidden_size=128, env=env)
batch_Size = 32

for episode in range(5000):
    state = env.reset()
    episode_reward = 0

    for step in range(15000):
        action = agent.get_action(state)
        new_state, reward, done, _ = env.step(action.squeeze(0))
        env.render(mode='rgb')
        agent.Memory.push(state, action, reward, new_state, done)

        if(len(agent.Memory)) > batch_Size:
            agent.update(batch_Size)

        state = new_state
        episode_reward += reward

        if done:
            print("episode: {}, reward: {} \n"
                  .format(episode, episode_reward))
            break
