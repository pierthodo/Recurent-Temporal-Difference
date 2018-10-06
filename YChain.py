import gym
from gym import spaces
from gym.utils import seeding
from gym.envs.registration import register

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