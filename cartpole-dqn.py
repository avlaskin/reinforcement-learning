import gym
import keras
import random
import os

from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import model_from_json
import numpy as np

class MyAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = 4
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.01
        self.wins = 0
        self.model = self._build_model()
        self.futureModel = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action
    # def act(self, state):
    #     e = random.random()
    #     if e > self.epsilon:
    #         act_values = self.model.predict(state)
    #         return np.argmax(act_values[0])
    #     r = random.random()
    #     if r > 0.5:
    #         return 1
    #     return 0

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):

        self.futureModel.set_weights(self.model.get_weights())
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                # Not done means we have a potential solution, but we are not 100% sure about it
                # if we would be sure - gamma would be = 1
                # do not understand that
                #1. Using current model we predict movement for the next_state
                possibleActions = self.model.predict(next_state)
                target = (reward + self.gamma * np.amax(possibleActions[0]))

            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.futureModel.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.model.set_weights(self.futureModel.get_weights())

    def saveModel(self, fullStructureFileName, fullWeightsFileName):
        model_json = self.model.to_json()
        with open(fullStructureFileName, "w") as json_file:
            json_file.write(model_json)
        self.model.save_weights(fullWeightsFileName)

    def readModelFromFile(self, modelFile, modelWeightsFile):
        if os.path.isfile(modelFile) and os.path.isfile(modelWeightsFile):
          json_file = open(modelFile, 'r')
          loaded_model_json = json_file.read()
          json_file.close()
          self.model = model_from_json(loaded_model_json)
          # load weights into new model
          self.model.load_weights(modelWeightsFile)
          print("Loaded model from disk.")


env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = MyAgent(state_size, action_size)
agent.readModelFromFile("./rl-model.json", "./rl-model.h5")
agent.model.compile(loss='mse',
              optimizer=Adam(lr=agent.learning_rate))
episodes = 2000
batch_size = 32

state = env.reset()
print(state)
state = np.reshape(state, [1, 4])
# Cart Position, Cart Velocity, Pole Angle, Pole Velocity At Tip

for e in range(episodes):
    state = env.reset()
    state = np.reshape(state, [1, 4])
#         # time_t represents each frame of the game
#         # Our goal is to keep the pole upright as long as possible until score of 500
#         # the more time_t the more score
    old_wins = agent.wins
    for time_t in range(500):
        # turn this on if you want to render
        env.render()
        # Decide action

        action = agent.act(state)
        # Advance the game to the next frame based on the action.
        # Reward is 1 for every frame the pole survived
        next_state, reward, done, _ = env.step(action)
        reward = reward if not done else -10
        next_state = np.reshape(next_state, [1, 4])
        # Remember the previous state, action, reward, and done
        agent.remember(state, action, reward, next_state, done)
        # make next_state the new current state for the next frame.
        state = next_state
        if done:
            # print the score and break out of the loop
            # if time_t > 100 and agent.learning_rate > 0.001:
            #     agent.learning_rate = 0.001
            # if time_t > 495:
            #     agent.learning_rate = agent.learning_rate / 2
            # if time_t < 50 and agent.learning_rate < 0.05:
            #     agent.learning_rate = agent.learning_rate * 2
            if time_t > 400:
                agent.wins = agent.wins + 1
            print("episode: {}/{}, score: {} epsilon: {:.2} learningRate: {:.4} Wins: {}"
                  .format(e, episodes, time_t, agent.epsilon, agent.learning_rate,agent.wins ))
            break
    if len(agent.memory) > batch_size:
            agent.replay(batch_size)
    if agent.wins > 100 and old_wins < agent.wins:
        print("Save model...")
        agent.saveModel("./rl-model.json", ("./rl-model.h5"))
