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

SMALL_ENOUGH = 1e-3



def generateRandomPolicy():
    ## State -> Array of Action probabilities
    n = 4
          # .  .  .  1
          # .  x  . -1
          # .  .  .  .
    ## Up, Down, Right, Left
    p = {
    (0,0): [0,  1/2,    0, 1/2],
    (0,1): [1/2, 1/2,   0,   0],
    (0,2): [1/2, 0, 1/2, 0],

    (1,0): [0, 0, 1/2, 1/2],
    (1,1): [0, 0, 0, 0],
    (1,2): [0, 0, 1/2, 1/2],

    (2,0): [  0, 1/3, 1/3, 1/3],
    (2,1): [1/3, 1/3, 1/3,   0],
    (2,2): [1/3,   0, 1/3, 1/3],

    (3,0): [0, 0, 0, 0],
    (3,1): [0, 0, 0, 0],
    (3,2): [1/2, 0, 0, 1/2],
    # This below seems like the more logical
    # as it does not imply our knowledge of the grid evironment
    # (0,0): [1/n, 1/n, 1/n, 1/n],
    # (0,1): [1/n, 1/n, 1/n, 1/n],
    # (0,2): [1/n, 1/n, 1/n, 1/n],
    #
    # (1,0): [1/n, 1/n, 1/n, 1/n],
    # (1,1): [1/n, 1/n, 1/n, 1/n],
    # (1,2): [1/n, 1/n, 1/n, 1/n],
    #
    # (2,0): [1/n, 1/n, 1/n, 1/n],
    # (2,1): [1/n, 1/n, 1/n, 1/n],
    # (2,2): [1/n, 1/n, 1/n, 1/n],
    #
    # (3,0): [0, 0, 0, 0],
    # (3,1): [0, 0, 0, 0],
    # (3,2): [1/n, 1/n, 1/n, 1/n],
      }
    return p

def generateDeterministicPolicy():
    ## State -> Array of Action probabilities
      # R  R  R  1
      # U  x  U -1
      # U  R  U  L
    ## Up, Down, Right, Left
    p = {
    (0,0): [0, 0, 1, 0],
    (0,1): [1, 0, 0, 0],
    (0,2): [1, 0, 0, 0],
    (1,0): [0, 0, 1, 0],
    (1,1): [1, 0, 0, 0],
    (1,2): [1, 0, 0, 0],
    (2,0): [0, 0, 1, 0],
    (2,1): [1, 0, 0, 0],
    (2,2): [1, 0, 0, 0],
    (3,0): [0, 0, 0, 0],
    (3,1): [0, 0, 0, 0],
    (3,2): [0, 0, 0, 1],
      }
    return p

def generateDeterministicPolicyTwo():
    ## State -> Array of Action probabilities
      # R  R  R  1
      # U  x  R -1
      # U  R  R  U
    ## Up, Down, Right, Left
    p = {
    (0,0): [0, 0, 1, 0],
    (0,1): [1, 0, 0, 0],
    (0,2): [1, 0, 0, 0],

    (1,0): [0, 0, 1, 0],
    (1,1): [0, 0, 0, 0],
    (1,2): [0, 0, 1, 0],

    (2,0): [0, 0, 1, 0],
    (2,1): [0, 0, 1, 0],
    (2,2): [0, 0, 1, 0],

    (3,0): [0, 0, 0, 0],
    (3,1): [0, 0, 0, 0],
    (3,2): [1, 0, 0, 0],
      }
    return p


def generateDeterministicPolicyRandomly(grid):
    ## State -> Array of Action probabilities
    ## By naute it is deterministic policy, where every step is very certain
    ## But by creation it is Random. We create it without knowledge of grid.
    p = {}
    allActions = [GridWorldMove.up,GridWorldMove.down,GridWorldMove.right,GridWorldMove.left]
    for i in range(grid.height):
        for j in range(grid.width):
            position = (j, i)
            if position in grid.actions.keys():
                print(position)
                element = np.random.choice(allActions)
                pp = [0,0,0,0]
                pp[element] = 1.0
                p.update({(j,i): pp})
    return p

def evaluateValueFunction(game, policy, gamma = 1.0):
    states = game.allStates()
    V = {}

    for s in states:
      V[s] = 0

    while True:
      biggest_change = 0

      for s in states:
          old_v = V[s]
          if s in game.actions.keys():
              new_v = 0
              for a in game.actions[s]:
                  game.setPosition(s)
                  reward = game.move(a)
                  p_as = policy[s] ## Array of positions
                  p_a = p_as[a] ## Action is a IntEnum, so we can use it as a index
                  new_v += p_a * (reward + gamma * V[game.position])
              V[s] = new_v
              biggest_change = max(biggest_change, np.abs(old_v - V[s]))
      if biggest_change < SMALL_ENOUGH:
        break
    return V

def itterateDetermenisticPolicy(game, value, policy, gamma = 1.0):
    states = game.allStates()
    newPolicy = policy
    bigNegative = -100000
    while True:
      policy_changed = False
      for s in states:
          if s in game.actions.keys():
              old_a = np.argmax(newPolicy[s])
              best_v = bigNegative
              best_a = 0
              for a in game.actions[s]:
                  game.setPosition(s)
                  reward = game.move(a)
                  ### THis is a super deterministic approach
                  new_v = reward + gamma * value[game.position]
                  if new_v > best_v and game.position != s:
                      best_a = a
                      best_v = new_v
              if best_v > bigNegative and old_a != best_a:
                  p = [0,0,0,0]
                  p[ best_a ] = 1
                  newPolicy.update({s: p})
                  policy_changed = True
                  ## print("Old a: {} New: {}".format(old_a,best_a))
      if policy_changed == False:
          break
    return newPolicy

def itteratePolicy(game, value, policy, gamma = 1.0, learningRate = 0.05):
    states = game.allStates()
    newPolicy = policy
    while True:
      policy_changed = False
      for s in states:
          if s in game.actions.keys():
              old_a = np.argmax(newPolicy[s])
              best_v = -10000
              best_a = 0
              for a in game.actions[s]:
                  game.setPosition(s)
                  reward = game.move(a)
                  p_as = newPolicy[s] ## Array of positions
                  p_a = p_as[a] ## Action is a IntEnum, so we can use it as a index
                  new_v = reward + gamma * value[game.position]
                  if new_v > best_v:
                      best_a = a
                      best_v = new_v
              if old_a != best_a:
                  probs = newPolicy[s]
                  for i in range(0,len(probs)):
                      if i == best_a:
                          probs[i] += learningRate
                      else:
                         probs[i] -= learningRate
                         if probs[i] < 0:
                           probs[i] = 0
                  ## And we normalise it here
                  summa = np.sum(probs)
                  if summa > 0:
                      for i in range(0,len(probs)):
                          probs[i] = probs[i] / summa
                  newPolicy.update({s: probs})
                  policy_changed = True
      if policy_changed == False:
          break
    return newPolicy

def testToEvaluateValue():
    gamma = 1
    g = StandardGrid()
    p1 = generateRandomPolicy()
    V1 = evaluateValueFunction(g,p1,gamma)
    print("========================================")
    print_policy(p1,g)
    print_policy_beautifly(p1,g)
    print("========================================")
    print_values(V1,g)
    #####
    g = StandardGrid()
    p2 = generateDeterministicPolicyTwo()
    V2 = evaluateValueFunction(g,p2,gamma)
    print("========================================")
    print_policy(p2,g)
    print_policy_beautifly(p2,g)
    print("========================================")
    print_values(V2,g)

def testForDetermenisticPolicyEvaluation():
    print("Standart Grid Deterministic Policy Testing!")
    gamma = 0.5
    g = StandardGrid()
    p2 = RandomPolicy.createRandomDeterministicPolicy(g.actions,g.width, g.height)
    print("Before itterative improvement:")
    print("========================================")
    print_policy_beautifly(p2,g)
    print_policy(p2,g)
    print("========================================")
    for i in range(0,100):
        V2 = evaluateValueFunction(g,p2,gamma=gamma)
        p2 = itterateDetermenisticPolicy(g,V2,p2,gamma=gamma)
    print("After improvement:")
    print_policy_beautifly(p2,g)
    print_policy(p2,g)
    print_values(V2,g)

def testForRandomPolicyEvaluation():
    print("Standart Grid Random Policy Testing!")
    gamma = 0.5
    g = StandardGrid()
    p = RandomPolicy.createRandomPolicy(g.actions,g.width, g.height)
    print("Before itterative improvement:")
    print("========================================")
    print_policy_beautifly(p,g)
    print_policy(p,g)
    for i in range(0,1000):
        v = evaluateValueFunction(g,p,gamma=gamma)
        p = itteratePolicy(g,v,p,gamma=gamma)
    print("After improvement:")
    print_policy_beautifly(p,g)
    print_policy(p,g)
    print_values(v,g)

def testForRandomPolicyEvaluationNegativeGrid():
    print("Negative Grid Random Policy Testing!")
    gamma = 0.7
    g = NegativeGrid(-0.50)
    p = RandomPolicy.createRandomPolicy(g.actions,g.width, g.height)
    print("Before itterative improvement:")
    print("========================================")
    print_policy_beautifly(p,g)
    print_policy(p,g)
    for i in range(0,1000):
        v = evaluateValueFunction(g,p,gamma=gamma)
        p = itteratePolicy(g,v,p,gamma=gamma)
    print("After improvement:")
    print_policy_beautifly(p,g)
    print_policy(p,g)
    print_values(v,g)

def testForValueItteration():
    # Last algorithm of the chapter, but the most efficient one
    # Itterating Values to get the policy automatically
    print("Negative Grid Value Itteration Testing!")
    gamma = 0.9
    bigNegative = -10000
    allActions = [GridWorldMove.up, GridWorldMove.down, GridWorldMove.right, GridWorldMove.left]
    g = NegativeGrid(-0.50)
    #p = generateDeterministicPolicyRandomly()
    p = generateDeterministicPolicyRandomly(g)
    print("Before itterative improvement:")
    print("========================================")
    print_policy_beautifly(p,g)
    print_policy(p,g)
    V = {}
    states = g.allStates()
    for s in states:
      if s in g.actions:
        V[s] = np.random.random()
      else:
        V[s] = 0
    convergenceMet = False
    while not convergenceMet:
      biggest_change = 0
      for s in states:
            old_v = V[s]
            # Policy already reduced to all possible actions
            if s in p:
              new_v = bigNegative
              for a in allActions:
                g.setPosition(s)
                r = g.move(a)
                v = r + gamma * V[g.position]
                if v > new_v:
                  new_v = v
              V[s] = new_v
              biggest_change = max(biggest_change, np.abs(old_v - V[s]))
      if biggest_change < SMALL_ENOUGH:
        convergenceMet = True
    # Building optimum policy
    for s in p.keys():
      best_a = None
      best_value = bigNegative
      # loop through all possible actions
      # to find the best current action for the state
      for a in allActions:
        g.setPosition(s)
        r = g.move(a)
        v = r + gamma * V[g.position]
        if v > best_value:
          best_value = v
          best_a = a
        p[s][a] = 0.0
      # Best gets all
      p[s][best_a] = 1.0
    print("After Value itteration improvement:")
    print("========================================")
    print_policy_beautifly(p,g)
    print_policy(p,g)

if __name__ == '__main__':
    #testForDetermenisticPolicyEvaluation()
    #testForRandomPolicyEvaluation()
    #testForRandomPolicyEvaluationNegativeGrid()
    testForValueItteration()
