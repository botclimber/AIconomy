from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO

from env.CryptoTradingEnv import CryptoTradingEnv

import pandas as pd

df = pd.read_csv('./data/IDNA.csv')
df = df.sort_values('timestamp')

# The algorithms require a vectorized environment to run
env = DummyVecEnv([lambda: CryptoTradingEnv(df)])

model = PPO("MlpPolicy", env, verbose=1)

model.load("training/ppo_dummy")


def render(info):
    transactions = info[0]["transactions"]
    balance = info[0]["balance"]
    holdings = info[0]["holdings"]
    profit = info[0]["profit"]


    transToHtml = "<table class='table'><thead><th>Action</th><th>Coin Price</th><th>Earns</th><th>Holding Coins</th><th>Current Balance</th><th>Step</th><th>Reward</th></thead>"
    for x in transactions:
        transToHtml += "<tr>"

        transToHtml += "<td>"+x.type+"</td>"
        transToHtml += "<td>"+str(x.coinCurrentValue)+"</td>"
        transToHtml += "<td>"+str(x.earns)+"</td>" if(x.earns != None) else "<td style='color:red'>No valid operation</td>"
        transToHtml += "<td>"+str(x.coinsAmount)+"</td>"
        transToHtml += "<td>"+str(x.currentBalance)+"</td>"
        transToHtml += "<td>"+str(x.step)+"</td>"

        if x.reward > 0:
            transToHtml += "<td style='color:green'>"+str(x.reward)+"</td>"
        elif x.reward < 0:
            transToHtml += "<td style='color:red'>"+str(x.reward)+"</td>"
        else:
            transToHtml += "<td style='color:orange'>"+str(x.reward)+"</td>"

        

        transToHtml += "</tr>"

    transToHtml += "</table>"

    #create html file
    file_html = open("predict_output.html", "w")

    #adding data
    file_html.write('''<html>
    <head>
    <title>Predict Output</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css" integrity="sha384-rbsA2VBKQhggwzxH7pPCaAqO46MgnOM80zW1RWuH61DGLwZJEdK2Kadq2F9CUG65" crossorigin="anonymous">
    </head> 
    <body>
    <h1>Predict Output</h1>
    <h2>Balance: {0}$</h2>
    <h2>holdings: {1} coins</h2>
    <h2>Profit: {2}$</h2>
    <h3>Transactions: </h3>
    {3}
    </body>
    </html>'''.format(round(balance,2), holdings, round(profit,2), transToHtml))

    file_html.close()

obs = env.reset()

while True:
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)
    
    if done:
        render(info)
        break
