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

def test_td0_method(alpha=0.1, gamma = 0.9, eps = 0.1):
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
    (2, 1): [0, 0, 1 , 0],
    (2, 2): [0, 0, 1 , 0],
    (3, 2): [1, 0, 0 , 0]
    }
    print("Policy:")
    print_policy_beautifly(policy, game)
    V = {}
    states = game.allStates()
    for s in states:
        V[s] = 0
    for t in xrange(100000):
        s_and_rs = play_game(game, policy, False, eps)
        for i in xrange(len(s_and_rs) - 1):
            state, _ = s_and_rs[i]
            s_plus_one, reward = s_and_rs[i+1]
            V[state] = V[state] + alpha *(reward + gamma*V[s_plus_one] - V[state])
    print("Policy:")
    print_policy_beautifly(policy, game)
    print("Values:")
    print_values(V, game)

def test_td0_sarsa(alpha=0.1, gamma = 0.9, eps = 0.1):
    game = NegativeGrid()
    print_values(game.rewards, game)
    allActions=(GridWorldMove.up, GridWorldMove.down, GridWorldMove.right, GridWorldMove.left)
    states = game.allStates()
    Q = {}
    Q_visists = {}
    Q_visists_debug = {}
    for s in states:
        Q[s] = {}
        Q_visists[s] = {}
        Q_visists_debug[s] = 0
        for a in allActions:
            Q[s][a] = 0
            Q_visists[s][a] = 1.0
    t = 1.0
    deltas = []

    for j in xrange(100000):
        ## Learning rate change
        if j % 100 == 0:
            t += 10e-3

        if j % 1000 == 0:
            print(".")
        max_change = 0
        state = game.start
        game.position = state
        game_over = False
        action, _ = max_dict(Q[s])
        action = eps_random_action(action, eps=0.5/t)
        while not game_over:
            r = game.move(action)
            state2 = game.position
            game_over = game.gameOver()
            ## Let's find possible next actions
            action2, _ = max_dict(Q[state2])
            action2 = eps_random_action(action2, eps=0.5/t) # epsilon-greedy
            ## At this stage we have all SARSA, let's calculate Q
            betta = alpha / Q_visists[state][action]
            Q_visists[state][action] += 0.001 ## 0.005
            old_qsa = Q[state][action]
            Q[state][action] = Q[state][action] + betta*(r + gamma*Q[state2][action2] - Q[state][action])
            max_change = max(max_change, abs(Q[state][action] - old_qsa))
            Q_visists_debug[state] = Q_visists_debug.get(state,0) + 1
            action = action2
            state = state2
        deltas.append(max_change)
    plt.plot(deltas)
    plt.show()

    policy = {}
    V = {}
    for s in game.actions.keys():
      a, max_q = max_dict(Q[s])
      policy[s] = [0, 0, 0, 0]
      policy[s][a] = 1.0
      V[s] = max_q
    print("Policy:")
    print_policy_beautifly(policy, game)
    print("Values:")
    print_values(V, game)

def test_td0_q_learning(alpha=0.1, gamma = 0.9, eps = 0.5, delta_t = 10e-3, learning_decay=0.001):
    game = NegativeGrid()#StandardGrid()
    print_values(game.rewards, game)
    allActions=(GridWorldMove.up, GridWorldMove.down, GridWorldMove.right, GridWorldMove.left)
    states = game.allStates()
    Q = {}
    Q_visists = {}
    Q_visists_debug = {}
    for s in states:
        Q[s] = {}
        Q_visists[s] = {}
        Q_visists_debug[s] = 0
        for a in allActions:
            Q[s][a] = 0
            Q_visists[s][a] = 1.0
    t = 1.0
    deltas = []

    for j in xrange(100000):
        ## Learning rate change
        if j % 100 == 0:
            t += delta_t
        if j % 1000 == 0:
            print(".")
        max_change = 0
        state = game.start
        game.position = state
        game_over = False
        action = max_dict(Q[s])[0]
        while not game_over:
            ## The fact that we use e-greedy makes or actions non optimal during the play
            ## However this allows us to be off-policy
            ## That makes Q-LEarning - an OFF-Policy algorithm
            action = eps_random_action(action, eps=eps/t)
            r = game.move(action)
            state2 = game.position
            game_over = game.gameOver()
            ## Let's find possible next actions
            action2, q_max_a2 = max_dict(Q[state2])
            ## At this stage we have all Q-Learning params
            betta = alpha / Q_visists[state][action]
            Q_visists[state][action] += learning_decay
            old_qsa = Q[state][action]
            ### So we are taking Q* here despite not taking the best move as a future move.
            Q[state][action] = Q[state][action] + betta*(r + gamma*q_max_a2 - Q[state][action])
            max_change = max(max_change, abs(Q[state][action] - old_qsa))
            Q_visists_debug[state] = Q_visists_debug.get(state,0) + 1
            action = action2
            state = state2
        deltas.append(max_change)
    plt.plot(deltas)
    plt.show()

    policy = {}
    V = {}
    for s in game.actions.keys():
      a, max_q = max_dict(Q[s])
      policy[s] = [0, 0, 0, 0]
      policy[s][a] = 1.0
      V[s] = max_q
    print("Policy:")
    print_policy_beautifly(policy, game)
    print("Values:")
    print_values(V, game)

if __name__ == '__main__':
    ##test_td0_method()
    ##test_td0_sarsa()
    test_td0_q_learning()
