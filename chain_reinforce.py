import numpy as np
import matplotlib.pyplot as plt
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from YChain import YChain

torch.manual_seed(0)
np.random.seed(0)

gamma = 0.99
eps = np.finfo(np.float32).eps.item()

class Policy(nn.Module):
    def __init__(self, dim_state, dim_actions):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(dim_state, dim_actions, bias=False)
        
        self.saved_log_probs = []
        self.rewards = []
        
    def forward(self, x):
        x = torch.sigmoid(self.affine1(x))
        return x

def select_action(state, policy):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(state)
    m = torch.distributions.Bernoulli(probs)
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action))
    return action.type(torch.int).item()

def finish_episode(policy, optimizer):
    R = 0
    policy_loss = []
    rewards = []
    for r in policy.rewards[::-1]:
        R = r + gamma * R
        rewards.insert(0, R)
    rewards = torch.tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
    for log_prob, reward in zip(policy.saved_log_probs, rewards):
        policy_loss.append(-log_prob * reward )
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    '''
    print("printing gradient")
    for param in policy.parameters():
        print(param.grad)
    '''
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]
    
def encode(state, num_obs):
    temp_array = np.zeros(num_obs)
    temp_array[state] = 1
    return temp_array

def experiment(env, num_episodes, policy, optimizer, num_obs):
    all_rewards = []
    for i_episode in range(num_episodes):
        state = env.reset()
        state = encode(state, num_obs)
        done = False
        epi_reward = 0
        while not done:
            action = select_action(state, policy)
            next_state, reward, done, _ = env.step(action)
            next_state = encode(next_state, num_obs)
            policy.rewards.append(reward)
            state = next_state
            epi_reward += reward
        finish_episode(policy, optimizer)
        all_rewards.append(epi_reward)
    return all_rewards


def main(chain_len, num_expt):
    env = YChain(n=chain_len)
    env.seed(0)
    num_obs = env.observation_space.n
    num_actions = env.action_space.n

    policy = Policy(num_obs, 1)
    optimizer = optim.SGD(policy.parameters(), lr=1e-2)

    num_episodes = 1000
    runs_ = []
    a_prob = []
    
    for i in range(num_expt):
        runs_.append(experiment(env, num_episodes, policy, optimizer, num_obs))
        a_ = []
        for s in range(num_obs):
            s = encode(s, num_obs)
            a_.append(policy(torch.from_numpy(s).float().unsqueeze(0)).item())
        a_prob.append(a_)
    
    return runs_, a_prob


if __name__ == '__main__':

    pkl_output_rew = []
    pkl_output_probs = []
    for c in range(3, 31):
        print(c)
        chain_len = c
        num_expt = 10
        
        runs_rew, probs = main(chain_len, num_expt)
        pkl_output_rew.append(runs_rew)
        pkl_output_probs.append(probs)

    with open('reinforce_outputs/reinforce_vanilla_'+'_expt_'+str(num_expt)+'_rewards_.pkl', 'wb') as f:
        pickle.dump(pkl_output_rew, f)

    with open('reinforce_outputs/reinforce_vanilla_probs'+'_expt_'+str(num_expt)+'_rewards_.pkl', 'wb') as f:
        pickle.dump(pkl_output_probs, f)