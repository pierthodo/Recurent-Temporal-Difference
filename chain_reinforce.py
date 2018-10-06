import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import gym
from gym import spaces
from gym.utils import seeding
from gym.envs.registration import register

env.seed(0)
torch.manual_seed(0)

class YChain():
    def __init__(self, n=5):
        self.len_chain = n #length of one chain
        self.n = n*3 + 1 #length of MDP
        self.state = 0  # Start at beginning of the chain
        self.action_space = spaces.Discrete(2) #number of actions - 2 - [0: go left, 1: go right]
        self.pos_reward = 10
        self.neg_reward = -10
        self.observation_space = spaces.Discrete(self.n)
        self.seed()
        self.bottle_neck = n #bottleneck state - that connects the 3 chains

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        '''
        takes an action as an argument and returns the next_state, reward, done, info.
        '''
        assert self.action_space.contains(action)
        reward = 0
        done = False
        
        # deciding on the next chain to switch if in the bottleneck state
        if self.state == self.bottle_neck:
            if not action:
                self.state += 1
            else:
                self.state = self.len_chain * 2 + 1
                
        # keep moving forward in the chain if not in the bottleneck state irrespective of the action
        else:
            
            # if in next transition is terminal state, give out reward
            if (self.state == self.len_chain * 2 - 1) or (self.state == self.len_chain * 3 - 1):
                reward = (self.pos_reward if self.state == self.len_chain*2-1 else self.neg_reward)
                done = True
                self.state += 1
                
            else:
                self.state += 1
        return self.state, reward, done, {}

    def reset(self):
        '''
        transitions back to first state
        '''
        self.state = 0
        return self.state
    
register(
    id='YChain-v0',
    entry_point='ychain:YChain',
    timestep_limit=20000,
    reward_threshold=1,
)

class Policy(nn.Module):
    '''
    def __init__(self, dim_state, dim_actions):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(dim_state, 64)
        self.affine2 = nn.Linear(64, dim_actions)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)
    '''
    def __init__(self, dim_state, dim_actions):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(dim_state, dim_actions, bias=False)
        
        self.saved_log_probs = []
        self.rewards = []
        
    def forward(self, x):
        x = F.sigmoid(self.affine1(x))
        return x

def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(state)
    #m = Categorical(probs)
    m = torch.distributions.Bernoulli(probs)
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action))
    return action.type(torch.int).item()

def finish_episode():
    R = 0
    policy_loss = []
    rewards = []
    for r in policy.rewards[::-1]:
        R = r + gamma * R
        rewards.insert(0, R)
    rewards = torch.tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
    for log_prob, reward in zip(policy.saved_log_probs, rewards):
        policy_loss.append(-log_prob * (reward - 10) )
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()

    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]
    
def encode(state):
    temp_array = np.zeros(num_obs)
    temp_array[state] = 1
    return temp_array

def main():
    all_rewards = []
    for i_episode in range(num_episodes):
        state = env.reset()
        state = encode(state)
        done = False
        epi_reward = 0
        while not done:
            action = select_action(state)
            next_state, reward, done, _ = env.step(action)
            next_state = encode(next_state)
            policy.rewards.append(reward)
            state = next_state
            epi_reward += reward
        finish_episode()
        all_rewards.append(epi_reward)
    return all_rewards

env = YChain(n=20)
num_obs = env.observation_space.n
num_actions = env.action_space.n

policy = Policy(num_obs, 1)
optimizer = optim.Adam(policy.parameters(), lr=1e-1)
eps = np.finfo(np.float32).eps.item()

num_episodes = 5000
gamma = 0.99
runs_10 = []
if __name__ == '__main__':
    for i in range(1):
        runs_10.append(main())

'''
for i in range(num_obs):
    i_ = encode(i)
    action, p_b = select_action(i_)
    print(i, action, p_b.item())

'''
runs_10_np = np.array(runs_10)
mean_rew = runs_10_np.mean(axis=0)
plt.plot(runs_10_np[0])
plt.show()