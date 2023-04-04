import random
import json
import gym
from gym import spaces
import pandas as pd
import numpy as np

# Reward amplifier
REWARD_AMPLIFIER = 100

# time horizon of data that agent will analyze
DAYS = 92 # implicit DAYS - 2 so always add +2

# all considered variables from data file in the observation space
CONSIDER_VARIABLES = 6

# custom variables
CUSTOM_VARIABLES = 2

# initial balance
INIT_BALANCE = 5e4

# max steps in case of the agent gets stucked
MAX_STEPS = 2e5

# trade fee when buying or selling
#TRADE_FEE = 0.1


class Wallet:
    '''
     agent wallet control cash flow
    '''

    def __init__(self):
        self.balance = INIT_BALANCE
        self.holdings = 0

class Transaction:
    '''
    regist a transaction
    '''
    transType = {0: "buy", 1: "hold", 2: "sell"}

    def __init__(self, step, action, coinCurrentValue, earns, w, coinsAmount, reward):
        self.step = step
        self.type = self.transType[action]
        self.coinCurrentValue = coinCurrentValue 
        self.earns = earns
        self.coinsAmount = coinsAmount
        self.currentBalance = w.balance
        self.reward = reward
        
    def __str__(self):
        print("Transaction Type (",self.type,") \n coin current value (",self.coinCurrentValue,") \n earns (",self.earns,") \n coins earned (",self.coinsAmount,") \n current balance (",self.currentBalance,")")
        

class CryptoTradingEnv(gym.Env):

    def __init__(self, extData):
        super(CryptoTradingEnv, self).__init__()

        # 0 - buy | 1 - hold | 2 - sell
        self.action_space = spaces.Discrete(3)
        #self.observation_space = spaces.Box(
        #    low=0, high=np.inf, shape=(CONSIDER_VARIABLES, DAYS), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(CONSIDER_VARIABLES + CUSTOM_VARIABLES,), dtype=np.float64)
        
        self.df = extData

        # Set the current step to a random point within the data frame
        self.current_step = 0

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
            ], dtype=np.float64), 
            # custom variables
            np.array([self.wallet.balance, self.wallet.holdings], dtype=np.float64))

        return observation

    def  rewardSys(self, action, currentCoinPrice, valid):
        '''
        computes reward

        '''
        reward = 0.1

        if valid:
            if action == 1 and self.current_step > 0:
                if self.wallet.balance > 0:

                    if(currentCoinPrice < self.sample[3][self.current_step -1]):
                        reward = currentCoinPrice * REWARD_AMPLIFIER
                    #if (currentCoinPrice > self.sample[3][self.current_step -1]):
                    #    reward = - currentCoinPrice
                    #else:
                    #    reward = 0

                elif self.wallet.holdings > 0:

                    if(currentCoinPrice > self.sample[3][self.current_step -1]):
                        reward = currentCoinPrice * 2 * REWARD_AMPLIFIER
                    #elif (currentCoinPrice < self.sample[3][self.current_step -1]):
                    #    reward = - currentCoinPrice * 2
                    #else:
                    #    reward = 0
        else:
            reward = - REWARD_AMPLIFIER

        return reward

    def buy(self, current_price):
        coins_to_buy = self.wallet.balance / current_price
        earns = round(coins_to_buy * current_price, 8)

        self.wallet.balance = abs(round(self.wallet.balance - earns, 8))
        self.wallet.holdings += coins_to_buy

        return -earns

        #return earns

    def sell(self, current_price): 
        earns = self.wallet.holdings * current_price

        self.wallet.balance += earns
        self.wallet.holdings -= self.wallet.holdings 

        return earns
        

        #return earns

    def step(self, action):
        done = False
        valid = True

        # Fetch the current price
        current_price = self.sample[3][self.current_step]

        if action == 0 and self.wallet.balance > 0:  # Buy
            # Determine the number of coins that can be purchased
            earns = self.buy(current_price)

        elif action == 1: # Hold
            earns = 0
            
        elif action == 2 and self.wallet.holdings > 0:  # Sell
            # TODO: change line to sell chosen amount instead of everything DCA (Dollar Cost Averaging)
            earns = self.sell(current_price)

        else:
            earns = None
            valid = False

        # Calculate the reward as the change in portfolio value
        reward = self.rewardSys(action, current_price, valid)

        self.transactions.append(Transaction(self.current_step, action, current_price, earns, self.wallet, self.wallet.holdings, reward))

        # Check if the episode is done
        self.current_step += 1
        if self.current_step == DAYS -1:
            done = True

        # Update the observation and return the step
        obs = self._next_observation()
        return obs, reward, done, {"transactions": self.transactions, "balance": self.wallet.balance, "holdings": self.wallet.holdings, "profit": (self.wallet.balance - INIT_BALANCE)}

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

    def render(self, mode='human'):
        pass

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
        

