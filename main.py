# main function that sets up environments
# and performs training loop

from collections import deque
from maddpg_agent import Agent
import numpy as np
import os
import time
import torch
from unityagents import UnityEnvironment

N_EPISODES = 2000
SOLVED_SCORE = 0.5
CONSEC_EPISODES = 100
PRINT_EVERY = 10
ADD_NOISE = True


## Helper functions

def seeding(seed=1):
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_actions(states, add_noise):
    actions = [agent.act(states, add_noise) for agent in agents]
    # flatten action pairs into a single vector
    return np.reshape(actions, (1, num_agents*action_size))

def reset_agents():
    for agent in agents:
        agent.reset()

def learning_step(states, actions, rewards, next_states, done):
    for i, agent in enumerate(agents):
        agent.step(states, actions, rewards[i], next_states, done, i)

## Training loop

# start environment
env = UnityEnvironment(file_name='Tennis.app')

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like: \n{}\n'.format(states[0]))

# initialize agents
agents = [Agent(state_size, action_size, num_agents=1, random_seed=1) for i in range(num_agents)]

# initialize scoring
scores_window = deque(maxlen=CONSEC_EPISODES)
scores_all = []
moving_average = []
best_score = -np.inf
best_episode = 0
already_solved = False

for i_episode in range(1, N_EPISODES+1):
    env_info = env.reset(train_mode=True)[brain_name]      # reset the environment
    states = np.reshape(env_info.vector_observations, (1,num_agents*state_size)) # flatten states
    reset_agents()
    scores = np.zeros(num_agents)
    while True:
        actions = get_actions(states, ADD_NOISE)           # choose agent actions and flatten them
        env_info = env.step(actions)[brain_name]           # send both agents' actions to the environment
        next_states = np.reshape(env_info.vector_observations, (1, num_agents*state_size)) # flatten next states
        rewards = env_info.rewards                         # get rewards
        done = env_info.local_done                         # see if the episode finished
        learning_step(states, actions, rewards, next_states, done)  # perform the learning step
        scores += np.max(rewards)                          # update scores with best reward
        states = next_states                               # roll over states to next time step
        if np.any(done):                                   # exit loop if episode finished
            break

    ep_best_score = np.max(scores)                         # record best score for episode
    scores_window.append(ep_best_score)                    # add score to recent scores
    scores_all.append(ep_best_score)                       # add score to histor of all scores
    moving_average.append(np.mean(scores_window))          # recalculate moving average

    # save best score
    if ep_best_score > best_score:
        best_score = ep_best_score
        best_episode = i_episode

    # print results
    if i_episode % PRINT_EVERY == 0:
        print('Episodes {:0>4d}-{:0>4d}\tMax Reward: {:.3f}\tMoving Average: {:.3f}'.format(
            i_episode-PRINT_EVERY, i_episode, np.max(scores_all[-PRINT_EVERY:]), moving_average[-1]))

    # determine if environment is solved and keep best performing models
    if moving_average[-1] >= SOLVED_SCORE:
        if not already_solved:
            print('<-- Environment solved in {:d} episodes! \
            \n<-- Moving Average: {:.3f} over past {:d} episodes'.format(
                i_episode-CONSEC_EPISODES, moving_average[-1], CONSEC_EPISODES))
            already_solved = True
            # save weights
            torch.save(agent_0.actor_local.state_dict(), 'models/checkpoint_actor_0.pth')
            torch.save(agent_0.critic_local.state_dict(), 'models/checkpoint_critic_0.pth')
            torch.save(agent_1.actor_local.state_dict(), 'models/checkpoint_actor_1.pth')
            torch.save(agent_1.critic_local.state_dict(), 'models/checkpoint_critic_1.pth')
        elif ep_best_score >= best_score:
            print('<-- Best episode so far!\
            \nEpisode {:0>4d}\tMax Reward: {:.3f}\tMoving Average: {:.3f}'.format(
            i_episode, ep_best_score, moving_average[-1]))
            # save weights
            torch.save(agent_0.actor_local.state_dict(), 'models/checkpoint_actor_0.pth')
            torch.save(agent_0.critic_local.state_dict(), 'models/checkpoint_critic_0.pth')
            torch.save(agent_1.actor_local.state_dict(), 'models/checkpoint_actor_1.pth')
            torch.save(agent_1.critic_local.state_dict(), 'models/checkpoint_critic_1.pth')
        # stop training if model stops improving
        elif (i_episode-best_episode) >= 200:
            print('<-- Training stopped. Best score not matched or exceeded for 200 episodes')
            break
        else:
            continue
env.close()
