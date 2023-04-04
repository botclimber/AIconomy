import gym
import json
import datetime as dt

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO

from env.CryptoTradingEnv import CryptoTradingEnv

import pandas as pd

df = pd.read_csv('./data/IDNA.csv')
df = df.sort_values('timestamp')

# The algorithms require a vectorized environment to run
env = DummyVecEnv([lambda: CryptoTradingEnv(df)])

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="log/ppo_train_log/")
model.learn(total_timesteps=1e6, reset_num_timesteps=False)

model.save("training/ppo_dummy")
