import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 6e-2              # for soft update of target parameters
LR_ACTOR = 1e-3         # learning rate of the actor
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
LEARN_EVERY = 1         # learning timestep interval
LEARN_NUM = 1           # number of learning passes
OU_SIGMA = 0.2          # Ornstein-Uhlenbeck noise parameter
OU_THETA = 0.13         # Ornstein-Uhlenbeck noise parameter
# EPSILON = 1.0           # explore->exploit noise process added to act step
# EPSILON_DECAY = 4e-3    # decay rate for noise process

eps_start = 6           # Noise level start
eps_end = 0             # Noise level end
eps_decay = 250         # Number of episodes to decay over from start to end

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, num_agents, random_seed):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            num_agents (int): number of agents
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.seed = random.seed(random_seed)
        self.eps = eps_start
        self.t_step = 0

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        self.agents = [Agent(state_size, action_size, num_agents, random_seed) for agent in range(num_agents)]

        # Noise process
        self.noise = OUNoise(action_size, random_seed)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)

    # def step(self, state, action, reward, next_state, done, agent_number):
    #     """Save experience in replay memory, and use random sample from buffer to learn."""
    #     self.t_step += 1
    #     # Save experience / reward
    #     self.memory.add(state, action, reward, next_state, done)
    #
    #     # Learn, if enough samples are available in memory and at interval settings
    #     if len(self.memory) > BATCH_SIZE and self.t_step % LEARN_EVERY == 0:
    #             for _ in range(LEARN_NUM):
    #                 experiences = self.memory.sample()
    #                 self.learn(experiences, GAMMA, agent_number)
    def get_actions(self, states, add_noise):
        """Returns actions for given state as per current policy."""
        # action_0 = agent_0.act(states, ADD_NOISE)           # agent 1 chooses an action
        # action_1 = agent_1.act(states, ADD_NOISE)           # agent 2 chooses an action
        # actions = np.concatenate((action_0, action_1), axis=0).flatten()

        # actions = [agent.act(state, add_noise) for i, state in enumerate(states)]
        # print('states shape = {}'.format(states.shape))
        actions = []
        for i, state in enumerate(states):
            print('i, state in enumerate(states) = {}, {}'.format(i, state))
            action = self.agents[i].act(state, add_noise)
            actions.append(action)
        actions = np.reshape(actions, (1,4))
        # print('actions = {}'.format(actions))
        return actions

    def act(self, state, add_noise):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        # actions = np.zeros((self.num_agents, self.action_size))
        # print('actions (zeros) = {}'.format(actions))
        self.actor_local.eval()
        with torch.no_grad():
            # for agent_num, state in enumerate(states):
                # print('agent_num = {}'.format(agent_num))
                # print('state = {}'.format(state))
            action = self.actor_local(state).cpu().data.numpy()
                # print('action = {}'.format(action))
                # actions[agent_num, :] = action
        # print('actions = {}'.format(actions))
        self.actor_local.train()
        if add_noise:
            action += self.eps * self.noise.sample()
        # print('actions (w/ noise) = {}'.format(actions))
        action = np.clip(action, -1, 1)
        # print('actions (clipped) = {}'.format(actions))
        return action

    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        self.t_step += 1
        # print('states shape = {}'.format(states.shape))
        # print('actions shape = {}'.format(len(actions)))
        # print('actions = {}'.format(actions))
        # print('rewards shape = {}'.format(len(rewards)))
        # print('rewards = {}'.format(rewards))
        # print('next_states shape = {}'.format(next_states.shape))
        # print('dones shape = {}'.format(len(dones)))

        # Save experience / reward
        # for i, state in enumerate(states):
        #     self.memory.add(states[i], actions, rewards[i], next_states[i], dones[i])
        self.memory.add(states, actions, rewards, next_states, dones)

        # Learn, if enough samples are available in memory and at interval settings
        if len(self.memory) > BATCH_SIZE and self.t_step % LEARN_EVERY == 0:
                for _ in range(LEARN_NUM):
                    experiences = self.memory.sample()
                    Agent.learn(experiences, GAMMA)

    # def reset(self):
    #     self.noise.reset()
    def reset(self):
        for agent in self.agents:
            noise.reset(agent)

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences
        # print('states shape = {}'.format(states.shape()))
        # print('actions shape = {}'.format(actions.shape()))
        # print('rewards shape = {}'.format(rewards.shape()))
        # print('next_states shape = {}'.format(next_states.shape()))
        # print('dones shape = {}'.format(dones.shape()))

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)

        # if agent_number == 0:
        #     actions_next = torch.cat((actions_next, actions[:,2:]), dim=1)
        # else:
        #     actions_next = torch.cat((actions[:,:2], actions_next), dim=1)

        # for i, agent in enumerate(agents):
        #     if i == 0:
        #         actions =



        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)

        if agent_number == 0:
            actions_pred = torch.cat((actions_pred, actions[:,2:]), dim=1)
        else:
            actions_pred = torch.cat((actions[:,:2], actions_pred), dim=1)

        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)

        # Update epsilon noise value
        self.eps = self.eps - (1/eps_decay)
        if self.eps < eps_end:
            self.eps=eps_end

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

# class AgentTrainer():
#     """Initiates some of the agent interactions."""
#
#     def __init__(self, state_size, action_size, num_agents, random_seed):
#         """Initialize multiple Agent objects.
#
#         Params
#         ======
#             state_size (int): dimension of each state
#             action_size (int): dimension of each action
#             num_agents (int): number of agents
#             random_seed (int): random seed
#         """
        # self.state_size = state_size
        # self.action_size = action_size
        # self.num_agents = num_agents
        # self.seed = random.seed(random_seed)
        # self.t_step = 0

        # self.agents = [Agent(state_size, action_size, num_agents, random_seed) for agent in range(num_agents)]

        # # Noise process
        # self.noise = OUNoise(action_size, random_seed)
        #
        # # Replay memory
        # self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)

    # def act(self, states, add_noise):
    #     """Returns actions for given state as per current policy."""
    #     # action_0 = agent_0.act(states, ADD_NOISE)           # agent 1 chooses an action
    #     # action_1 = agent_1.act(states, ADD_NOISE)           # agent 2 chooses an action
    #     # actions = np.concatenate((action_0, action_1), axis=0).flatten()
    #
    #     # actions = [agent.act(state, add_noise) for i, state in enumerate(states)]
    #     # print('states shape = {}'.format(states.shape))
    #     actions = []
    #     for i, state in enumerate(states):
    #         print('i, state in enumerate(states) = {}, {}'.format(i, state))
    #         action = self.agents[i].act(state, add_noise)
    #         actions.append(action)
    #     actions = np.reshape(actions, (1,4))
    #     # print('actions = {}'.format(actions))
    #     return actions

    # def reset(self):
    #     for i in self.agents:
    #         Agent.reset(self)

    # def step(self, states, actions, rewards, next_states, dones):
    #     """Save experience in replay memory, and use random sample from buffer to learn."""
    #     self.t_step += 1
    #     print('states shape = {}'.format(states.shape))
    #     print('actions shape = {}'.format(len(actions)))
    #     print('actions = {}'.format(actions))
    #     print('rewards shape = {}'.format(len(rewards)))
    #     print('rewards = {}'.format(rewards))
    #     print('next_states shape = {}'.format(next_states.shape))
    #     print('dones shape = {}'.format(len(dones)))
    #     # Save experience / reward
    #     # for i, state in enumerate(states):
    #     #     self.memory.add(states[i], actions, rewards[i], next_states[i], dones[i])
    #     self.memory.add(states, actions, rewards, next_states, dones)
    #
    #     # Learn, if enough samples are available in memory and at interval settings
    #     if len(self.memory) > BATCH_SIZE and self.t_step % LEARN_EVERY == 0:
    #             for _ in range(LEARN_NUM):
    #                 experiences = self.memory.sample()
    #                 Agent.learn(self, self.agents, experiences, GAMMA)
    #


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0.0, theta=OU_THETA, sigma=OU_SIGMA):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.size = size
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        return self.state

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
