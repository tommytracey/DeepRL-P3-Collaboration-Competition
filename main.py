# main function that sets up environments
# and performs training loop

from collections import deque
from maddpg import Agent, AgentTrainer
import numpy as np
import os
import time
import torch
from unityagents import UnityEnvironment

N_EPISODES = 3000
MAX_T = 1000
SOLVED_SCORE = 0.5
CONSEC_EPISODES = 100
PRINT_EVERY = 1
ADD_NOISE = True
MODEL_DIR = 'models/'

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
    # agent = AgentTrainer()
    agent_0 = Agent(state_size, action_size, num_agents=1, random_seed=1)
    agent_1 = Agent(state_size, action_size, num_agents=1, random_seed=1)

    # scoring placeholders
    mean_scores = []                               # list of mean scores from each episode
    min_scores = []                                # list of lowest scores from each episode
    max_scores = []                                # list of highest scores from each episode
    best_score = -np.inf
    scores_window = deque(maxlen=CONSEC_EPISODES)  # mean scores from most recent episodes
    moving_avgs = []                               # list of moving averages

    # training loop
    for i_episode in range(1, n_episodes+1):
        start_time = time.time()
        env_info = env.reset(train_mode=train_mode)[brain_name]    # reset environment
        states = np.reshape(env_info.vector_observations, (1,48))  # get current state for each agent
        scores = np.zeros(num_agents)                              # initialize score for each agent
        # agent.reset()
        agent_0.reset()
        agent_1.reset()

        for t in range(MAX_T):
            # actions = agent.act(states, add_noise=ADD_NOISE)
            action_0 = agent_0.act(states, ADD_NOISE)           # agent 0 chooses an action
            action_1 = agent_1.act(states, ADD_NOISE)           # agent 1 chooses an action
            actions = np.concatenate((action_0, action_1), axis=0)
            actions = np.reshape(actions, (1, 4))
            env_info = env.step(actions)[brain_name]            # send actions to environment
            next_states = np.reshape(env_info.vector_observations, (1,48))   # get next state
            rewards = env_info.rewards                          # get reward
            dones = env_info.local_done                         # see if episode has finished
            # save experience to replay buffer, perform learning step at defined interval
#             for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
#                 agent.step(state, action, reward, next_state, done, t)
            agent_0.step(states, actions, rewards[0], next_states, done, t, 0) # agent 0 learns
            agent_1.step(states, actions, rewards[1], next_states, done, t, 1) # agent 1 learns
            # agent.step(states, actions, rewards, next_states, dones, t)

            states = next_states
            scores += rewards
            if np.any(dones):                                   # exit loop when episode ends
                break

        duration = time.time() - start_time
        min_scores.append(np.min(scores))             # save lowest score for a single agent
        max_scores.append(np.max(scores))             # save highest score for a single agent
        mean_scores.append(np.mean(scores))           # save mean score for the episode
        scores_window.append(max_scores[-1])          # save mean score to window
        moving_avgs.append(np.mean(scores_window))    # save moving average

        if i_episode % PRINT_EVERY == 0:
            print('\rEpisode {} ({} sec)  -- \tMin: {:.1f}\tMax: {:.1f}\tMean: {:.1f}\tMov. Avg: {:.1f}'.format(\
                  i_episode, round(duration), min_scores[-1], max_scores[-1], mean_scores[-1], moving_avgs[-1]))

#         if train_mode and mean_scores[-1] > best_score:
#             torch.save(agent.actor_local.state_dict(), actor_path)
#             torch.save(agent.critic_local.state_dict(), critic_path)

#         if moving_avgs[-1] >= solved_score and i_episode >= consec_episodes:
#             print('\nEnvironment SOLVED in {} episodes!\tMoving Average ={:.1f} over last {} episodes'.format(\
#                                     i_episode-consec_episodes, moving_avgs[-1], consec_episodes))
#             if train_mode:
#                 torch.save(agent.actor_local.state_dict(), actor_path)
#                 torch.save(agent.critic_local.state_dict(), critic_path)
#             break


    env.close()
    # logger.close()

if __name__=='__main__':
    main()
