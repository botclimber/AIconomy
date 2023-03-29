from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO

from env.CryptoTradingEnv import CryptoTradingEnv

import pandas as pd

df = pd.read_csv('./data/IDNA.csv')
df = df.sort_values('timestamp')

# The algorithms require a vectorized environment to run
env = DummyVecEnv([lambda: CryptoTradingEnv(df)])

model = PPO("MlpPolicy", env, verbose=1)

model.load("dummy")


def render(info):
    transactions = info[0]["transactions"]
    balance = info[0]["balance"]
    holdings = info[0]["holdings"]
    profit = info[0]["profit"]

    print("Transactions: \n")
    for x in transactions:
        print("\t\t", x.__str__())

    # Render the environment to the screen
    print(f'Balance: {balance}')
    print(f'Holdings: {holdings}')
    print(f'Profit: {profit}')


obs = env.reset()

while True:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    
    if done:
        render(info)
        break
