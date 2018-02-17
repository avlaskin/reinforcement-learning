from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from enum import IntEnum
from gridWorld import StandardGrid
from gridWorld import NegativeGrid
from gridWorld import GridWorldMove
from gridWorld import RandomPolicy

from printHelpers import print_values
from printHelpers import print_policy
from printHelpers import print_policy_beautifly

def max_dict(dict):
  # returns the argmax (key) and max (value) from a dictionary
  # put this into a function since we are using it so often
  max_key = None
  max_val = float('-inf')
  for k, v in dict.items():
    if v > max_val:
      max_val = v
      max_key = k
  return max_key, max_val

def eps_random_action(a, eps=0.1, allActions=(GridWorldMove.up, GridWorldMove.down, GridWorldMove.right, GridWorldMove.left)):
    if np.random.random() < (1 - eps):
        return a
    else:
        return np.random.choice(allActions)

class LinearSarsaState:
    def __init__(self, indexes = 25):
        self.weights = np.random.randn(indexes) / np.sqrt(indexes)

    def featureMapper(self, s, a):
        ## Four features and first is the bais
        features = [1,
        s[0] - 1 if a == GridWorldMove.up else 0,
        s[1] - 1.5 if a == GridWorldMove.up else 0,
        (s[0]*s[1] - 3)/3 if a == GridWorldMove.up else 0,
        (s[0]*s[0] - 2)/2 if a == GridWorldMove.up else 0,
        (s[1]*s[1] - 4.5)/4.5 if a == GridWorldMove.up else 0,
        1                      if a == GridWorldMove.up else 0,

        s[0] - 1 if a == GridWorldMove.down else 0,
        s[1] - 1.5 if a == GridWorldMove.down else 0,
        (s[0]*s[1] - 3)/3 if a == GridWorldMove.down else 0,
        (s[0]*s[0] - 2)/2 if a == GridWorldMove.down else 0,
        (s[1]*s[1] - 4.5)/4.5 if a == GridWorldMove.down else 0,
        1                      if a == GridWorldMove.down else 0,

        s[0] - 1 if a == GridWorldMove.right else 0,
        s[1] - 1.5 if a == GridWorldMove.right else 0,
        (s[0]*s[1] - 3)/3 if a == GridWorldMove.right else 0,
        (s[0]*s[0] - 2)/2 if a == GridWorldMove.right else 0,
        (s[1]*s[1] - 4.5)/4.5 if a == GridWorldMove.right else 0,
        1                      if a == GridWorldMove.right else 0,
        s[0] - 1 if a == GridWorldMove.left else 0,
        s[1] - 1.5 if a == GridWorldMove.left else 0,
        (s[0]*s[1] - 3)/3 if a == GridWorldMove.left else 0,
        (s[0]*s[0] - 2)/2 if a == GridWorldMove.left else 0,
        (s[1]*s[1] - 4.5)/4.5 if a == GridWorldMove.left else 0,
        1                      if a == GridWorldMove.left else 0
        ]
        return np.array(features)
    def predict(self, s, a):
        x = self.featureMapper(s, a)
        return self.weights.dot(x)

    def gradient(self, s, a):
        return self.featureMapper(s,a)

    def getQs(self, s):
      Qs = {}
      for a in (GridWorldMove.up, GridWorldMove.down, GridWorldMove.right, GridWorldMove.left):
        q_sa = self.predict(s, a)
        Qs[a] = q_sa
      return Qs

def test_td0_sarsa_approximated(alpha=0.1, gamma = 0.9, eps = 0.1, itterations = 20000):
    game = NegativeGrid(-0.5)
    print_values(game.rewards, game)
    allActions=(GridWorldMove.up, GridWorldMove.down, GridWorldMove.right, GridWorldMove.left)
    states = game.allStates()
    t = 1.0
    deltas = []
    model = LinearSarsaState()
    for j in xrange(itterations):
        ## Learning rate change
        if j % 1000 == 0:
            t += 10e-3

        if j % 1000 == 0:
            print(".")

        max_change = 0
        state = game.start
        game.position = state
        game_over = False
        Qs = model.getQs(state)
        action = max_dict(Qs)[0]
        action = eps_random_action(action, eps=0.5/t)
        alpha = alpha / 1.0001
        while not game_over:
            r = game.move(action)
            state2 = game.position
            game_over = game.gameOver()
            Qs2 = model.getQs(state2)
            ## Let's find possible next actions
            action2 = max_dict(Qs2)[0]
            action2 = eps_random_action(action2, eps=0.5/t) # epsilon-greedy
            ## At this stage we have all SARSA, let's calculate Q
            old_weights = model.weights.copy()
            model.weights += alpha*(r + gamma*model.predict(state2, action2) - model.predict(state, action))*model.gradient(state, action)
            max_change = max(max_change, np.abs(model.weights - old_weights).sum())
            action = action2
            state = state2
        deltas.append(max_change)

    policy = {}
    V = {}
    Q = {}
    for s in game.actions.keys():
      Qs = model.getQs(s)
      Q[s] = Qs
      a, max_q = max_dict(Qs)
      policy[s] = [0, 0, 0, 0]
      policy[s][a] = 1
      V[s] = max_q
    print("Policy:")
    print_policy_beautifly(policy, game)
    print("Values:")
    print_values(V, game)
    plt.plot(deltas)
    plt.show()

def play_game(game, policy, windy=False, eps=0.0, allActions=(GridWorldMove.up, GridWorldMove.down, GridWorldMove.right, GridWorldMove.left)):

    game.position = game.start
    state = game.position
    # initial state, arrived to start
    stateAndRewards = [(state, 0)]
    gameOver = False
    while not gameOver:
        action = allActions[np.argmax(policy[state])]
        action = eps_random_action(action, eps)
        reward = game.move(action)
        gameOver = game.gameOver()
        print(gameOver)
        state = game.position
        stateAndRewards.append((state,reward))
    return stateAndRewards

class LinearState:
    def __init__(self):
        self.weights = np.random.randn(4)
    def featureMapper(self, s):
        ## Four features and first is the bais
        features = [1, s[0] - 1, s[1] - 1.5, s[0]*s[1] - 3]
        return np.array(features)
    def predict(self, s):
        x = self.featureMapper(s)
        return self.weights.dot(x)
    def gradient(self, s):
        return self.featureMapper(s)

def test_td0_approximation_method(alpha=0.1, gamma = 0.9, eps = 0.1, itterations = 10000):
    game = StandardGrid()
    print_values(game.rewards, game)
    policy = {
    ## GridWorldMove.up, GridWorldMove.down, GridWorldMove.right, GridWorldMove.left
    (0, 0): [0, 0, 1 , 0],
    (0, 1): [1, 0, 0 , 0],
    (0, 2): [1, 0, 0 , 0],
    (1, 0): [0, 0, 1 , 0],
    (1, 2): [0, 0, 1 , 0],
    (2, 0): [0, 0, 1 , 0],
    (2, 1): [0, 0, 1 , 0], #(2, 1): [0, 0, 1 , 0],
    (2, 2): [0, 0, 1 , 0], #(2, 2): [0, 0, 1 , 0],
    (3, 2): [1, 0, 0 , 0]
    }
    print("Policy:")
    print_policy_beautifly(policy, game)
    states = game.allStates()
    # Let's create a state
    model = LinearState()
    deltas = [] ## Debug only
    for t in xrange(itterations):
        s_and_rs = play_game(game, policy, False, eps)
        biggest_change = 0
        alpha = alpha / 1.0001
        for i in xrange(len(s_and_rs) - 1):
            state, _ = s_and_rs[i]
            s_plus_one, reward = s_and_rs[i+1]
            if game.isTerminal(s_plus_one):
                target = reward
            else:
                target = reward + gamma*model.predict(s_plus_one)
            #This is for debug only
            old_weights = model.weights.copy()
            # This is gradient step similar to below
            #V[state] = V[state] + alpha *(reward + gamma*V[s_plus_one] - V[state])
            model.weights += alpha*(target - model.predict(state))*model.gradient(state)
            biggest_change = max(biggest_change, np.abs(old_weights - model.weights).sum())
        deltas.append(biggest_change)
    # We need to recreate the V
    V = {}
    for s in states:
      print(s)
      if s in game.actions:
        V[s] = model.predict(s)
      else:
        # terminal state or state we can't otherwise get to
        V[s] = 0

    print("Policy:")
    print_policy_beautifly(policy, game)
    print("Values:")
    print_values(V, game)
    plt.plot(deltas)
    plt.show()

def fetures_from_state(state):
    x = state[0]
    y = state[1]
    return [x-3, y-2, x*y - 6]

if __name__ == '__main__':
    ##test_td0_approximation_method(itterations = 50000)
    test_td0_sarsa_approximated()
