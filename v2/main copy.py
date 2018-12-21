# main function that sets up environments
# and performs training loop

from collections import deque
from maddpg_agent import Agent #, AgentTrainer
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
# MODEL_DIR = 'models/'


def seeding(seed=1):
    np.random.seed(seed)
    torch.manual_seed(seed)

def main():
    seeding()

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
    print('The state for the first agent looks like:', states[0])

    # initialize agents
    agent_0 = Agent(state_size, action_size, 1, random_seed=0)
    agent_1 = Agent(state_size, action_size, 1, random_seed=0)

    # n_episodes = 1000
    scores_window = deque(maxlen=CONSEC_EPISODES)
    scores_all = []
    moving_average = []

    for i_episode in range(1, N_EPISODES+1):
        env_info = env.reset(train_mode=True)[brain_name]      # reset the environment
        states = env_info.vector_observations
        states = np.reshape(states, (1,48))
        agent_0.reset()
        agent_1.reset()
        scores = np.zeros(num_agents)
        while True:
            action_0 = agent_0.act(states, ADD_NOISE)           # agent 1 chooses an action
            action_1 = agent_1.act(states, ADD_NOISE)           # agent 2 chooses an action
            actions = np.concatenate((action_0, action_1), axis=0).flatten()
            env_info = env.step(actions)[brain_name]           # send both agents' actions together to the environment
            next_states = env_info.vector_observations         # get next states
            next_states = np.reshape(next_states, (1, 48))     # combine each agent's state into one state space
            rewards = env_info.rewards                         # get reward
            done = env_info.local_done                         # see if episode finished
            agent_0.step(states, actions, rewards[0], next_states, done, 0) # agent 1 learns
            agent_1.step(states, actions, rewards[1], next_states, done, 1) # agent 2 learns
            scores += np.max(rewards)                                  # update the score for each agent
            states = next_states                               # roll over states to next time step

            if np.any(done):                                  # exit loop if episode finished
                break

        scores_window.append(np.max(scores))
        scores_all.append(np.max(scores))
        moving_average.append(np.mean(scores_window))

        if i_episode % PRINT_EVERY == 0:
            print('Episodes {:0>4d}-{:0>4d}\tMax Reward: {:.3f}\tMoving Average: {:.3f}'.format(
                i_episode-PRINT_EVERY, i_episode, np.max(scores_all[-PRINT_EVERY:]), np.mean(scores_window)))

        if np.mean(scores_window)>=SOLVED_SCORE:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.3f}'.format(
                i_episode-CONSEC_EPISODES, np.mean(scores_window)))
            torch.save(agent_0.actor_local.state_dict(), 'checkpoint_actor_0.pth')
            torch.save(agent_0.critic_local.state_dict(), 'checkpoint_critic_0.pth')
            torch.save(agent_1.actor_local.state_dict(), 'checkpoint_actor_1.pth')
            torch.save(agent_1.critic_local.state_dict(), 'checkpoint_critic_1.pth')
            break

    env.close()



if __name__=='__main__':
    main()
