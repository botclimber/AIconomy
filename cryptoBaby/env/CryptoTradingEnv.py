import random
import json
import gym
from gym import spaces
import pandas as pd
import numpy as np

# observation space constants
DAYS = 100
CONSIDER_VARIABLES = 6

# constants
BALANCE = 10000

MAX_STEPS = 2e5

class CryptoTradingEnv(gym.Env):
    def __init__(self, extData):
        super(CryptoTradingEnv, self).__init__()
        
        # 0 - buy | 1 - hold | 2 - sell
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(DAYS, CONSIDER_VARIABLES), dtype=np.float32)
        pass

    def _next_observation(self):
        # Get the stock data points for the last 5 days and scale to between 0-1
        frame = np.array([
            self.df.loc[self.current_step: self.current_step +
                        DAYS, 'open'].values,
            self.df.loc[self.current_step: self.current_step +
                        DAYS, 'high'].values,
            self.df.loc[self.current_step: self.current_step +
                        DAYS, 'low'].values,
            self.df.loc[self.current_step: self.current_step +
                        DAYS, 'marketCap'].values,
            self.df.loc[self.current_step: self.current_step +
                        DAYS, 'volume'].values,
        ])

        # Append additional data and scale each value to between 0-1
        obs = np.append(frame, [[
            self.balance / MAX_ACCOUNT_BALANCE,
            self.max_net_worth / MAX_ACCOUNT_BALANCE,
            self.shares_held / MAX_NUM_SHARES,
            self.cost_basis / MAX_SHARE_PRICE,
            self.total_shares_sold / MAX_NUM_SHARES,
            self.total_sales_value / (MAX_NUM_SHARES * MAX_SHARE_PRICE),
        ]], axis=0)

        return obs

    def step(self, action):

        #return state, reward, done, {} 
        pass
    
    def reset(self):
        # Reset the state of the environment to an initial state
        self.balance = BALANCE

        # Set the current step to a random point within the data frame
        self.current_step = random.randint(0, len(self.df.loc[:, 'open'].values) - DAYS)

        return self._next_observation()
        pass

    def render(self):
        pass