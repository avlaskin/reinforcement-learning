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

def getWindyAction(action, threshold=0.5):
    print(".")
    p = np.random.random()
    if p > threshold:
        return action
    else:
        actions = np.arange(4)
        actions = np.delete(actions, [action])
        return np.random.choice(actions)

def playGame(game, policy, gamma, windy=False, eps=0.0):
    game.position = game.start
    state = game.position
    # initial state, arrived to start
    stateAndRewards = [(state, 0)]
    gameOver = False
    count = 0
    while not gameOver:
        p = np.random.random()
        count+=1
        if count > 10000:
            print("Stuck!")
            print(game.position)
            break
        if p < eps:
            ## random Move
            action = np.random.choice(np.arange(4))
        else:
          actions = list(policy[state])
          for i in range(len(actions)):
            actions[i] = actions[i]*np.random.random()
          action = np.argmax(actions)
        if windy:
            action = getWindyAction(action)
        reward = game.move(action)
        gameOver = game.gameOver()
        state = game.position
        stateAndRewards.append((state,reward))

    G = 0
    stateAndReturns = []
    for s, r in reversed(stateAndRewards):
        stateAndReturns.append((s, G))
        G = r + gamma * G
    return list(reversed(stateAndReturns))

def generateRandomPolicy():
    n = 4.0
    p = {
        (0,0): [1/n, 1/n, 1/n, 1/n],
        (0,1): [1/n, 1/n, 1/n, 1/n],
        (0,2): [1/n, 1/n, 1/n, 1/n],

        (1,0): [1/n, 1/n, 1/n, 1/n],
        (1,1): [1/n, 1/n, 1/n, 1/n],
        (1,2): [1/n, 1/n, 1/n, 1/n],

        (2,0): [1/n, 1/n, 1/n, 1/n],
        (2,1): [1/n, 1/n, 1/n, 1/n],
        (2,2): [1/n, 1/n, 1/n, 1/n],

        (3,0): [0, 0, 0, 0],
        (3,1): [0, 0, 0, 0],
        (3,2): [1/n, 1/n, 1/n, 1/n],
    }
    return p

def generateRandomlyDeterministicPolicy(g):
    p = RandomPolicy.createRandomDeterministicPolicy(g.actions,g.width, g.height)
    return p

def testMonteCarloValuesEvaluation(windy=False):
    gamma = 0.9
    eps = 0.2
    game = StandardGrid()#NegativeGrid(-0.25)
    ## This is to make things real
    #generateRandomPolicy()
    policy = generateRandomlyDeterministicPolicy(game)
    V = {}
    returns = {}
    states = game.allStates()
    seenStates = {}
    for s in states:
        if s in game.actions.keys():
            returns[s] = []
            seenStates.update({s : 0})
        else:
            V[s] = 0

    for t in range(0,10000):

        sAr = playGame(game, policy, gamma, windy, eps)
        for s, g in sAr:
            if s in game.actions.keys():
                if seenStates[s] == 0:
                    returns[s].append(g)
                    V[s] = np.mean(returns[s])
                    seenStates.update({s : 1})

    print_values(V, game)

testMonteCarloValuesEvaluation()
