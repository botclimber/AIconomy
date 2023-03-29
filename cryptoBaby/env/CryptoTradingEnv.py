import random
import json
import gym
from gym import spaces
import pandas as pd
import numpy as np

# Constants
DAYS = 20
CONSIDER_VARIABLES = 6
INIT_BALANCE = 5e4
MAX_STEPS = 2e5
TRADE_FEE = 0.001

class Wallet:

    def __init__(self):
        self.balance = INIT_BALANCE
        self.holdings = 0

class Transaction:
    transType = {0: "buy", 2: "sell"}

    def __init__(self, action, coinCurrentValue, earns, w):
        self.type = self.transType[action]
        self.coinCurrentValue = coinCurrentValue 
        self.earns = earns
        self.currentBalance = w.balance
        
    def __str__(self):
        print("Transaction Type (",self.type,") \n coin current value (",self.coinCurrentValue,") \n earns (",self.earns,") \n current balance (",self.currentBalance,")")
        

class CryptoTradingEnv(gym.Env):
    def __init__(self, extData):
        super(CryptoTradingEnv, self).__init__()

        # 0 - buy | 1 - hold | 2 - sell
        self.action_space = spaces.Discrete(3)
        #self.observation_space = spaces.Box(
        #    low=0, high=np.inf, shape=(CONSIDER_VARIABLES, DAYS), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(CONSIDER_VARIABLES + 3,), dtype=np.float32)
        
        self.df = extData

    def _next_observation(self):
        # Get the stock data points for the last DAYS
        observation = np.append(
            np.array([
            self.sample[0][self.current_step],
            self.sample[1][self.current_step],
            self.sample[2][self.current_step],
            self.sample[3][self.current_step],
            self.sample[4][self.current_step],
            self.sample[5][self.current_step],
            ], dtype=np.float32), 
            np.array([self.wallet.balance, self.wallet.holdings, self.wallet.balance + self.wallet.holdings * self.df.loc[self.current_step, 'open']], dtype=np.float32) )

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

        # Fetch the current price
        current_price = self.df.loc[self.current_step, 'close']

        if action == 0 and self.wallet.balance > 0:  # Buy
            # Determine the number of coins that can be purchased
            coins_to_buy = self.wallet.balance / current_price
            calc = coins_to_buy * current_price
            fee = calc * TRADE_FEE
            earns = calc - fee
            print(self.wallet.balance, current_price, coins_to_buy, calc, earns)
            self.wallet.balance -= earns + fee
            self.wallet.holdings += coins_to_buy
            self.transactions.append(Transaction(action, current_price, 0 - (earns + fee), self.wallet))

        elif action == 2 and self.wallet.holdings > 0:  # Sell
            # TODO: change line to sell chosen amount instead of everything DCA (Dollar Cost Averaging)
            earns = self.wallet.holdings * current_price * (1 - TRADE_FEE)
            self.wallet.balance += earns
            self.wallet.holdings = 0
            self.transactions.append(Transaction(action, current_price, earns, self.wallet))

        # Calculate the current portfolio value
        current_value = self.wallet.balance + self.wallet.holdings * current_price

        # Calculate the reward as the change in portfolio value
        reward = current_value - INIT_BALANCE

        # Check if the episode is done
        if self.current_step + 1 == DAYS:
            done = True

        # Update the observation and return the step
        obs = self._next_observation()
        return obs, reward, done, {}

    def getSample(self, randomStart):

        sample = np.array([
            self.df.loc[randomStart:randomStart + DAYS - 1, 'open'].values,
            self.df.loc[randomStart:randomStart + DAYS - 1, 'high'].values,
            self.df.loc[randomStart:randomStart + DAYS - 1, 'low'].values,
            self.df.loc[randomStart:randomStart + DAYS - 1, 'close'].values,
            self.df.loc[randomStart:randomStart+ DAYS - 1, 'volume'].values,
            self.df.loc[randomStart:randomStart + DAYS - 1, 'marketCap'].values,
        ])

        return sample


    def reset(self):
        # Reset the state of the environment to an initial state
        self.wallet = Wallet()
        self.transactions = []

        self.total_buys = 0
        self.total_sells = 0
        self.total_holds = 0

        # Set the current step to a random point within the data frame
        self.current_step = 0
        
        # get data sample from time horizon
        self.sample = self.getSample(random.randint(0, len(self.df.loc[:, 'open'].values) - DAYS))

        return self._next_observation()

    def render(self, mode='human', close=False):

        print("Transactions: \n")
        for x in self.transactions:
            print("\t\t", x.__str__())

        # Render the environment to the screen
        print(f'Step: {self.current_step}')
        print(f'Balance: {self.wallet.balance}')
        print(f'Profit: {self.wallet.balance - INIT_BALANCE}')
        

