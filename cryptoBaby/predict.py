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

obs = env.reset()

for i in range(10):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()