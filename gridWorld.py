import random
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from enum import IntEnum

class GridWorldMove(IntEnum):
    up = 0
    down = 1
    right = 2
    left = 3

class GridWorld:
    def __init__(self, width, height, start):
        self.height = height
        self.width = width
        self.position = start
        self.start = start
        # Position on top left is (0,0), bottom rigth is (w,h)

    def set(self, rewards, actions):
        # rewards should be a dict of: (i, j): r (row, col): reward
        # actions should be a dict of: (i, j): A (row, col): list of possible actions
        self.rewards = rewards
        self.actions = actions

    def setPosition(self, p):
        self.position = p

    def isTerminal(self, s):
      return s not in self.actions

    def move(self, action):
      # check if legal move first
      if action in self.actions[self.position]:
        a, b = self.position
        if action == GridWorldMove.up:
          #self.position[1] -= 1
          self.position = (a, b-1)
        elif action == GridWorldMove.down:
          #self.position[1] += 1
          self.position = (a, b+1)
        elif action == GridWorldMove.right:
          #self.position[0] += 1
          self.position = (a+1, b)
        elif action == GridWorldMove.left:
          #self.position[0] -= 1
          self.position = (a-1, b)
      # return a reward (if any)
      return self.rewards.get(self.position, 0)

    def undoMove(self, action):
      # these are the opposite of what U/D/L/R should normally do
      if action == GridWorldMove.up:
        self.position[1] += 1
      elif action == GridWorldMove.down:
        self.position[1] -= 1
      elif action == GridWorldMove.right:
        self.position[0] -= 1
      elif action == GridWorldMove.left:
        self.position[0] += 1
      # raise an exception if we arrive somewhere we shouldn't be
      # should never happen
      assert(self.position in self.allStates())

    def gameOver(self):
      # returns true if game is over, else false
      # true if we are in a state where no actions are possible
      return self.position not in self.actions

    def allStates(self):
      # possibly buggy but simple way to get all states
      # either a position that has possible next actions
      # or a position that yields a reward
      return set(self.actions.keys()) | set(self.rewards.keys())

class RandomPolicy:
    def __init__(self, allowedActions, width, height):
        allActions = (0,1,2,3)
        self.width = width
        self.height = height
        self.policy = {}
        for i in range(0, height):
            for j in range(0, width):
                p = (j,i)
                if p in allowedActions.keys():
                    values = [0,0,0,0]
                    actions = allowedActions[ p ]
                    n = len(allowedActions[ p ])
                    for a in allActions:
                        for b in actions:
                            if a == b:
                                values[a] = 1.0/n
                    self.policy.update({p: values})
                else:
                    self.policy.update({p: [0,0,0,0]})

    def createRandomPolicy(allowedActions, width, height):
        policy = {}
        allActions = (0,1,2,3)
        for i in range(0, height):
            for j in range(0, width):
                p = (j,i)
                if p in allowedActions.keys():
                    values = [0,0,0,0]
                    actions = allowedActions[ p ]
                    n = len(allowedActions[ p ])
                    for a in allActions:
                        for b in actions:
                            if a == b:
                                values[a] = 1.0/n
                    policy.update({p: values})
                else:
                    policy.update({p: [0,0,0,0]})
        return policy

    def createRandomDeterministicPolicy(allowedActions, width, height):
        policy = {}
        for i in range(0, height):
            for j in range(0, width):
                p = (j,i)
                if p in allowedActions.keys():
                    values = [0,0,0,0]
                    actions = allowedActions[ p ]
                    c = np.random.choice(actions)
                    values[c] = 1
                    policy.update({p: values})
                else:
                    policy.update({p: [0,0,0,0]})
        return policy

def StandardGrid():
  # define a grid that describes the reward for arriving at each state
  # and possible actions at each state
  # the grid looks like this
  # x means you can't go there
  # s means start position
  # number means reward at that state
  # .  .  .  1
  # .  x  . -1
  # s  .  .  .
  g = GridWorld(4, 3, (2, 0))
  rewards = {(3, 0): 1, (3, 1): -1}
  actions = {
    (0, 0): (GridWorldMove.down, GridWorldMove.right),
    (0, 1): (GridWorldMove.up, GridWorldMove.down),
    (0, 2): (GridWorldMove.up, GridWorldMove.right),
    (1, 0): (GridWorldMove.left, GridWorldMove.right),
    (1, 2): (GridWorldMove.left, GridWorldMove.right),
    (2, 0): (GridWorldMove.left, GridWorldMove.down, GridWorldMove.right),
    (2, 1): (GridWorldMove.up, GridWorldMove.down, GridWorldMove.right),
    (2, 2): (GridWorldMove.left, GridWorldMove.right, GridWorldMove.up),
    (3, 2): (GridWorldMove.left, GridWorldMove.up),
  }
  g.set(rewards, actions)
  return g

def NegativeGrid(step_cost=-0.1):
  # in this game we want to try to minimize the number of moves
  # so we will penalize every move
  g = StandardGrid()
  g.rewards.update({
    (0, 0): step_cost,
    (0, 1): step_cost,
    (0, 2): step_cost,
    (1, 0): step_cost,
    (1, 2): step_cost,
    (2, 0): step_cost,
    (2, 1): step_cost,
    (2, 2): step_cost,
    (3, 2): step_cost,
  })
  return g
