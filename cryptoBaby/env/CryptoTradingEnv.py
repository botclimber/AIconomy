import random
import json
import gym
from gym import spaces
import pandas as pd
import numpy as np

# Constants
DAYS = 20
CONSIDER_VARIABLES = 6
BALANCE = 10000
MAX_STEPS = 2e5
TRADE_FEE = 0.001

class CryptoTradingEnv(gym.Env):
    def __init__(self, extData):
        super(CryptoTradingEnv, self).__init__()

        # 0 - buy | 1 - hold | 2 - sell
        self.action_space = spaces.Discrete(3)
        #self.observation_space = spaces.Box(
        #    low=0, high=np.inf, shape=(CONSIDER_VARIABLES, DAYS), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(DAYS * CONSIDER_VARIABLES + 3,), dtype=np.float32)
        
        self.df = extData
        self.balance = BALANCE
        self.holdings = 0
        self.current_step = 0

    def _next_observation(self):
        # Get the stock data points for the last DAYS
        observation = np.concatenate([
            self.df.loc[self.current_step:self.current_step + DAYS - 1, 'open'].values,
            self.df.loc[0:0 + DAYS - 1, 'high'].values,
            self.df.loc[self.current_step:self.current_step + DAYS - 1, 'low'].values,
            self.df.loc[self.current_step:self.current_step + DAYS - 1, 'close'].values,
            self.df.loc[self.current_step:self.current_step + DAYS - 1, 'volume'].values,
            self.df.loc[self.current_step:self.current_step + DAYS - 1, 'marketCap'].values,
            np.array([self.balance, self.holdings, self.balance + self.holdings * self.df.loc[self.current_step, 'open']], dtype=np.float32)
        ])
        # print(observation)

        '''
        frame = np.array([
            self.df.loc[self.current_step: self.current_step +
                        DAYS -1, 'open'].values,
            self.df.loc[self.current_step: self.current_step +
                        DAYS -1, 'high'].values,
            self.df.loc[self.current_step: self.current_step +
                        DAYS -1, 'low'].values,
            self.df.loc[self.current_step: self.current_step +
                        DAYS -1, 'close'].values,
            self.df.loc[self.current_step: self.current_step +
                        DAYS -1, 'volume'].values,
            self.df.loc[self.current_step: self.current_step +
                        DAYS -1, 'marketCap'].values,
            
        ])

        # Append additional data
        zeros = 
        obs = np.append(frame, [[
            self.balance,
            self.holdings,
            self.balance + self.holdings * self.df.loc[self.current_step, 'open'],
        ]], axis=0)

        return obs
        '''
        return observation

    def step(self, action):
        self.current_step += 1
        done = False

        if self.current_step > len(self.df.loc[:, 'open'].values) - DAYS:
            self.current_step = 0

        # Fetch the current price
        current_price = self.df.loc[self.current_step, 'close']

        if action == 0:  # Buy
            # Determine the number of coins that can be purchased
            coins_to_buy = self.balance / current_price
            self.balance -= coins_to_buy * current_price * (1 + TRADE_FEE)
            self.holdings += coins_to_buy

        elif action == 2:  # Sell
            # TODO: change line to sell chosen amount instead of everything DCA (Dollar Cost Averaging)
            self.balance += self.holdings * current_price * (1 - TRADE_FEE)
            self.holdings = 0

        # Calculate the current portfolio value
        current_value = self.balance + self.holdings * current_price

        # Calculate the reward as the change in portfolio value
        reward = current_value - BALANCE

        # Check if the episode is done
        if self.current_step == len(self.df) - 1:
            done = True

        # Update the observation and return the step
        obs = self._next_observation()
        return obs, reward, done, {}

    def reset(self):
        # Reset the state of the environment to an initial state
        self.balance = BALANCE
        self.holdings = 0

        # Set the current step to a random point within the data frame
        self.current_step = random.randint(0, len(self.df.loc[:, 'open'].values) - DAYS)

        return self._next_observation()

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance}')
        print(f'Profit: {self.balance - BALANCE}')

