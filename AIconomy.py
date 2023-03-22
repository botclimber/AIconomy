import numpy as np
import pandas as pd

# Create the environment
class EconomyEnv:
    def __init__(self, macro_data, micro_data, political_data):
        self.macro_data = macro_data
        self.micro_data = micro_data
        self.political_data = political_data
        self.current_state = None
        self.done = False

    # reset the environment to its initial state
    def reset(self):
        self.current_state = np.zeros(len(self.macro_data)+len(self.micro_data)+len(self.political_data))
        self.done = False
        return self.current_state

    # take a step in the environment
    def step(self, action):
        # execute the action
        self.current_state += action

        # get the reward
        rewards = self._calculate_rewards()

        # check if the episode is done
        done = self._check_done()

        return self.current_state, rewards, done

    # calculate the rewards based on the current state
    def _calculate_rewards(self):
        rewards = np.zeros(len(self.macro_data)+len(self.micro_data)+len(self.political_data))
        
        # calculate rewards for macroeconomic variables
        for i in range(len(self.macro_data)):
            rewards[i] = self.macro_data[i].calculate_reward(self.current_state[i])
            
        # calculate rewards for microeconomic variables
        for i in range(len(self.macro_data), len(self.macro_data)+len(self.micro_data)):
            rewards[i] = self.micro_data[i-len(self.macro_data)].calculate_reward(self.current_state[i])
            
        # calculate rewards for political variables
        for i in range(len(self.macro_data)+len(self.micro_data), len(self.macro_data)+len(self.micro_data)+len(self.political_data)):
            rewards[i] = self.political_data[i-len(self.macro_data)-len(self.micro_data)].calculate_reward(self.current_state[i])
            
        return rewards

    # check if the episode is done
    def _check_done(self):
        # check if any of the variables exceed their boundaries
        for i in range(len(self.current_state)):
            if self.current_state[i] > self.macro_data[i].max_value or self.current_state[i] < self.macro_data[i].min_value:
                return True
        
        return False

# Create the agent
class Agent:
    def __init__(self, env):
        self.env = env
        self.state_size = len(env.macro_data)+len(env.micro_data)+len(env.political_data)
        self.action_size = len(env.macro_data)+len(env.micro_data)+len(env.political_data)
        self.gamma = 0.95
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.memory = []

    # build the model
    def _build_model(self):
        model = keras.Sequential([
            keras.layers.Dense(64, activation="relu", input_shape=(self.state_size,)),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(self.action_size, activation="linear")
        ])
        model.compile(loss="mse", optimizer=keras.optimizers.Adam(lr=self.learning_rate))

        return model

    # take an action
    def act(self, state):
        action = self.model.predict(state)
        return action

    # store the experience
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # update the model
    def update_model(self):
        # sample a minibatch of experiences from the replay buffer
        minibatch = random.sample(self.memory, 32)
        
        # create empty lists for states, actions, rewards and next_states
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        
        # add the samples to the lists
        for state, action, reward, next_state, done in minibatch:
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
        
        # convert the lists to numpy arrays
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)
        
        # calculate the target q-values
        target_q_values = rewards + (1-dones)*self.gamma*np.amax(self.model.predict_on_batch(next_states), axis=1)
        
        # fit the model
        self.model.fit(states, target_q_values, epochs=1, verbose=0)

# Train the agent
def train(env, agent, episodes):
    # loop over the episodes
    for episode in range(episodes):
        # reset the environment
        state = env.reset()
        
        # loop over the steps
        done = False
        while not done:
            # choose an action
            action = agent.act(state)
            
            # take a step
            next_state, reward, done = env.step(action)
            
            # store the experience
            agent.remember(state, action, reward, next_state, done)
            
            # update the state
            state = next_state
            
        # update the model
        agent.update_model()

# Run the program
env = EconomyEnv(macro_data, micro_data, political_data)
agent = Agent(env)

train(env, agent, 1000)