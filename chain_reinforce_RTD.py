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
        self.betas = []
        self.rewards = []
        
    def forward(self, x):
        x = torch.sigmoid(self.affine1(x))
        return x

def select_action(beta, prev_prob, state, begin, policy):
    beta = torch.tensor([beta], requires_grad=False)
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(state)
    
    if begin:
        prev_prob = probs
        
    prob_tilde = beta * probs + (1 - beta) * prev_prob
    m_tilde_prob = torch.distributions.Bernoulli(prob_tilde)
    action = m_tilde_prob.sample()
    policy.saved_log_probs.append(m_tilde_prob.log_prob(action))
    return action.type(torch.int).item(), prob_tilde


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
        policy_loss.append(-log_prob * reward)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward(retain_graph = True)
    '''
    print("printing Gradient")
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

def experiment(env, num_episodes, policy, optimizer, num_obs, beta):
    all_rewards = []
    for i_episode in range(num_episodes):
        prev_prob = torch.Tensor(np.array([0.0]))
        state = env.reset()
        state = encode(state, num_obs)
        beta_state = state.dot(beta)[0]
        done = False
        begin = True
        epi_reward = 0
        while not done:
            action, prev_prob = select_action(beta_state, prev_prob, state, False, policy)
            next_state, reward, done, _ = env.step(action)
            next_state = encode(next_state, num_obs)
            policy.rewards.append(reward)
            state = next_state
            beta_state = state.dot(beta)[0]
            epi_reward += reward
            begin = False
        finish_episode(policy, optimizer)
        all_rewards.append(epi_reward)
    return all_rewards


def main(chain_len, num_expt):
    env = YChain(n=chain_len)
    state = env.reset()
    env.seed(0)
    num_obs = env.observation_space.n
    num_actions = env.action_space.n

    beta = np.zeros((num_obs, 1))
    beta[0] = 1

    policy = Policy(num_obs, 1)
    optimizer = optim.SGD(policy.parameters(), lr=1e-1)

    num_episodes = 500
    runs_ = []
    a_prob = []
    
    for i in range(num_expt):
        runs_.append(experiment(env, num_episodes, policy, optimizer, num_obs, beta))
        a_ = []
        for s in range(num_obs):
            s = encode(s, num_obs)
            a_.append(policy(torch.from_numpy(s).float().unsqueeze(0)).item())
        a_prob.append(a_)
    
    
    return runs_, a_prob


if __name__ == '__main__':
    for c in range(3, 31):
        print(c)
        chain_len = c
        num_expt = 10

        runs_rew, probs = main(chain_len, num_expt)

        with open('reinforce_RTD_outputs/reinforce_RTD_chain_'+str(chain_len)+'_expt_'+str(num_expt)+'_rewards_.pkl', 'wb') as f:
            pickle.dump(runs_rew, f)

        with open('reinforce_RTD_outputs/reinforce_RTD_chain_probs_'+str(chain_len)+'_expt_'+str(num_expt)+'_rewards_.pkl', 'wb') as f:
            pickle.dump(probs, f)