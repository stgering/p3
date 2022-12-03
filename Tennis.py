from unityagents import UnityEnvironment
import numpy as np
from ddpg_agent import Agent
import torch
from collections import deque
import matplotlib.pyplot as plt

env = UnityEnvironment(file_name='Tennis_Windows_x86_64/Tennis.exe',no_graphics=True)

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents 
num_agents = len(env_info.agents)

# size of each action
action_size = brain.vector_action_space_size

# size state space 
states = env_info.vector_observations
state_size = states.shape[1]

# init agent
agent = Agent(state_size, action_size, random_seed=2)

def ddpg(n_episodes=3000, target_score=0.5, print_every=10, window_len=100, updateAfterEvery=1, numUpdateCycles=1):
    scoreOverEpisodes = []
    averageScore = []
    scoreDeque = deque(maxlen=window_len)    
    actions = np.zeros([num_agents, action_size])

    for i_episode in range(1, n_episodes+1):                   
        env_info = env.reset(train_mode=True)[brain_name]      # reset the environment    
        states = env_info.vector_observations                  # get the current state (for each agent)
        agent.reset()                                          # reset noise
        scores = np.zeros(num_agents)                          # initialize the score (for each agent)
        t = 0

        while True:
            t += 1
            actions = agent.act(states)                        # select an action (for each agent)            
            env_info = env.step(actions)[brain_name]           # send all actions to tne environment
            next_states = env_info.vector_observations         # get next state (for each agent)
            rewards = env_info.rewards                         # get reward (for each agent)
            dones = env_info.local_done                        # see if episode finished
            
            for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
                agent.store(state, action, reward, next_state, done)   
                      
            if t % updateAfterEvery == 0:
                for i in range(numUpdateCycles):
                    agent.learnFromBufferSamples()                    

            scores += rewards                                  # update the score (for each agent)
            states = next_states                               # roll over states to next time step
            if np.any(dones):                                  # exit loop if episode finished
                break

        if i_episode % print_every == 0:    
            print('Score of episode {:04d}: {:1.4f} \t Window mean: {:1.4f}'.format(i_episode, np.max(scores), np.mean(scoreDeque)))
       
        scoreOverEpisodes.append(np.max(scores))
        scoreDeque.append(np.max(scores))
        averageScore.append(np.mean(scoreDeque))

        if np.mean(scoreDeque) >= target_score:
            # solved
            break

    torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')      # save final actor
    torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')    # save final critic

    print('Windowed mean score at episode {:04d}: \t Window mean: {:1.4f}'.format(i_episode, np.max(scores), np.mean(scoreDeque)))

    return scoreOverEpisodes, averageScore

scores, average = ddpg()
env.close()


# plot results
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores, label='instant')
plt.plot(np.arange(1, len(average)+1), average, label='average')
plt.ylabel('Score')
plt.xlabel('Episode #')
ax.legend(fontsize='large', loc='upper left')
plt.show()
fig.savefig('scores.png')