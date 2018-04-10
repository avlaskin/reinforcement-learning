import gym
import numpy as np
import matplotlib.pyplot as plt

class CPRandomAgent:
    def __init__(self, state_size):
        self.weights = -1.0 + np.random.rand(4) * 2
        self.rewards = []

    def act(self, state):
        w = np.reshape(self.weights, [4, 1])
        res = state.dot(w)
        return 1 if res > 0.0 else 0

    def print_weights(self):
        for i in range(len(self.weights)):
            print("%f " % self.weights[i])

    def add_reward(self, reward):
        self.rewards.append(reward)

    def get_av(self):
        return np.average(self.rewards)


class Trainer:
    def __init__(self, env, itterations=1000, agents=1000, max_rounds=500):
        self.itterations = itterations
        self.agents = []
        self.env = env
        self.max_index = 0
        self.max_av = 0
        self.max_rounds = max_rounds
        state_size = self.env.observation_space.shape[0]
        action_size = env.action_space.n
        for i in range(agents):
            print(i)
            self.agents.append(CPRandomAgent(state_size))

    def play_episode(self, agent):
        i = 0
        state = env.reset()
        state = np.reshape(state, [1, 4])
        tt = 0
        for t in range(self.max_rounds):
            action = agent.act(state)
            state, reward, done, _ = env.step(action)
            # reward = reward if not done else -10
            if done:
                return t
        return self.max_rounds

    def play_all(self):
        l = len(self.agents)-1
        eps = []
        max = 0
        max_index = 0
        for i in range(l):
            agent = self.agents[i]
            for j in range(self.itterations):
                r = self.play_episode(agent)
                if r > max:
                    max = r
                    max_index = i
                eps.append(r)
                agent.add_reward(r)
            #agent.print_weights()
            print("Agent %d - Av: %f Max(%d) = %f " % (i, agent.get_av(), self.max_index, self.max_av))
            if agent.get_av() > self.max_av:
                self.max_av = agent.get_av()
                self.max_index = i
        print("Max reward: %f Index: %d" % (max, max_index))
        plt.plot(self.agents[i].rewards)
        plt.show()

env = gym.make('CartPole-v1')
trainer = Trainer(env, 200, 1000, 1000)
trainer.play_all()
