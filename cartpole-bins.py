import gym
import numpy as np
import matplotlib.pyplot as plt

class BinsTransformer:
    """This class makes bin-features for state space of cartpole."""

    def __init__(self):
        self.stateBins = []
        self.bins = [9, 9, 9, 9]
        self.mins = [-2.4, -3.6, -0.21, -3.6]
        self.maxes = [2.4,  3.6,  0.21,  3.6]
        for i in range(4):
            self.stateBins.append(np.linspace(self.mins[i], self.maxes[i], self.bins[i]))

    def state_to_int(self, state):
        """ Takes cartpole state and returns transformed features as Int."""
        nidx = []
        #value -> number
        for i in range(4):
            val = state[0, i]
            ni = np.digitize(val, self.stateBins[i])
            nidx.append(ni)

        #numbers -> String -> Int
        return int(''.join(str(x) for x in nidx))


class QLearningAgent:
    """ Q Learning Table model agent.
        Need to add save/load for the QTable for itterative training process.
    """
    def __init__(self, env):
        self.q_zise = [10000, 2]
        # Initial itialisation with zeros is much better for initial convergence.
        self.Q = np.zeros(self.q_zise)
        #np.random.uniform(low=-1,high=1,size=self.q_zise)
        self.transformer = BinsTransformer()
        self.alpha = 0.001
        self.env = env

    def predict(self, state):
        """ Takes state, returns Q for this state.
            Greedy max - will be the policy.
        """
        features = self.transformer.state_to_int(state)
        return self.Q[features]

    def epsilon_random_action(self, state, eps):
        """ Makes an action with exploration/exploitation dillema solved using epsilon greedy.
            Smaller epsilon more we stick to exploitation of Q-learning policy."""
        if np.random.random() < eps:
            return self.env.action_space.sample()
        p = self.predict(state)
        return np.argmax(p)

    def update(self, state, action, G):
        features = self.transformer.state_to_int(state)
        d_q = self.alpha*(G - self.Q[features, action])
        self.Q[features, action] += d_q


class Trainer:
    """ Trainer object that runs the expirement.
        It would be good find a better parameters for faster convergence.
    """
    def __init__(self, env, itterations=5000, max_rounds=500):
        self.itterations = itterations
        self.env = env
        self.eps = 0.1
        self.max_rounds = max_rounds
        self.q_agent = QLearningAgent(env)
        self.states = []
        self.gamma = 0.9

    def play_episode(self, agent):
        i = 0
        state = env.reset()
        state = np.reshape(state, [1, 4])
        tt = 0
        self.record_state(state)
        for t in range(self.max_rounds):
            prev_state = state
            action = agent.epsilon_random_action(state, self.eps)
            state, reward, done, _ = env.step(action)
            state = np.reshape(state, [1, 4])
            tt += reward
            if done and t < 400:
                reward = -400
            G = reward + self.gamma*np.max(agent.predict(state))
            agent.update(prev_state, action, G)
            # Some sorces keep training the model even after -
            # we reach the point of no return. Let's try that later.
            if done:
                return tt
            else:
                self.record_state(state)
        return tt

    def play_all(self):
        rewards = []
        m_rewards = []
        for j in range(self.itterations):
            r = self.play_episode(self.q_agent)
            rewards.append(r)
            #self.eps = 1.0 / (j+1)*(j+1)
            #self.eps = 1.0 / (j+1)
            self.eps = 1.0 / np.sqrt(j+1)
            if j % 500 == 1:
                print("AV reward: {} Iter: {}".format(np.mean(rewards[-500:]), j))
                m_rewards.append(np.mean(rewards[-500:]))
        plt.plot(m_rewards)
        plt.show()

    def record_state(self, state):
        self.states.append(state)

    def describe_state(self):
        print("Starting describe...",len(self.states))
        biner = BinsTransformer()
        self.states = np.array(self.states)
        self.states = np.reshape(self.states, [len(self.states), 4])
        for i in self.states:
            print("State: {} Int: {}".format(i, biner.state_to_int(i)))


if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    print(env.action_space)
    trainer = Trainer(env, itterations=200000)
    trainer.play_all()
