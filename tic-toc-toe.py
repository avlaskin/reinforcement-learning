import random
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from enum import IntEnum

class TTTMove(IntEnum):
    empty = 0
    circle = 1
    cross = 2

class TTTGameOver(IntEnum):
    canPlay = 0
    circleWon = 1
    crossWon = 2
    draw = 3

class TickTockToe:

    def __init__(self, side=3):
        self.side = int(side)
        self.state = np.zeros((side*side,), dtype=np.int)
        self.isOver = TTTGameOver.canPlay

    def setCell(self,x,y,value):
        self.state[x + y*self.side] = value

    def getCell(self,x,y):
        return self.state[x + y*self.side]

    def isEmpty(self,x,y):
        if self.state[x + y*self.side] == TTTMove.empty:
            return True
        return False

    def clearBoard(self):
        for i in range(0, self.side):
            for j in range(0, self.side):
                self.state[i + i*self.side] = 0

    def isGameOver(self, force_recalculate = False):
        if not force_recalculate and self.isOver != TTTGameOver.canPlay:
            return self.isOver

        res = TTTGameOver.draw
        e = 0
        ## Rows checking
        for i in range(0, self.side):
            x = 0
            o = 0
            xx = 0
            oo = 0
            for j in range(0, self.side):
                e += (self.state[i + j*self.side] == TTTMove.empty)
                x += (self.state[i + j*self.side] == TTTMove.cross)
                o += (self.state[i + j*self.side] == TTTMove.circle)
                xx += (self.state[j + i*self.side] == TTTMove.cross)
                oo += (self.state[j + i*self.side] == TTTMove.circle)
            if x == self.side or xx == self.side:
                return TTTGameOver.crossWon
            if o == self.side or oo == self.side:
                return TTTGameOver.circleWon
        if e > 0 :
            res = TTTGameOver.canPlay
        ## Diagonal checking
        x = 0
        o = 0
        xx = 0
        oo = 0
        for i in range(0, self.side):
                x += (self.state[i + i*self.side] == TTTMove.cross)
                o += (self.state[i + i*self.side] == TTTMove.circle)
                xx += (self.state[(self.side - 1 - i) + i*self.side] == TTTMove.cross)
                oo += (self.state[(self.side - 1 - i) + i*self.side] == TTTMove.circle)
        if x == self.side or xx == self.side:
            return TTTGameOver.crossWon
        if o == self.side or oo == self.side:
            return TTTGameOver.circleWon
        self.isOver = res
        return res

    def description(self):
        str = ""
        for j in range(0,self.side):
            for i in range(0,self.side):
                if self.state[i + j*self.side] == TTTMove.empty:
                    str += (" |  ")
                elif self.state[i + j*self.side] == TTTMove.circle:
                    str += (" | O")
                else:
                    str += (" | X")
            str = str+" |\n"
        return str

    def getStateHash(self):
        deg = 0
        s = 0
        for i in range(0, self.side):
            for j in range(0, self.side):
                v = self.getCell(i,j)
                s += (3**deg) * v
                deg += 1
        return s

    def internalTest(self):
        self.clearBoard()
        for i in range(0, self.side):
            self.state[2 + i*self.side] = TTTMove.cross
        r1 = self.isGameOver()
        if r1 != TTTGameOver.crossWon:
            print("Test Failed!")
        self.clearBoard()
        for i in range(0, self.side):
            self.state[i + 1*self.side] = TTTMove.circle
        r1 = self.isGameOver()
        if r1 != TTTGameOver.circleWon:
            print("Test Failed!")
        self.clearBoard()
        for i in range(0, self.side):
            self.state[self.side - 1 - i + i*self.side] = TTTMove.circle
        if r1 != TTTGameOver.circleWon:
            print("Test Failed!")
        self.clearBoard()

def getAllStates(env, i=0, j=0):
  results = []
  for v in (TTTMove.empty, TTTMove.cross, TTTMove.circle):
    env.setCell(i,j,v) # if empty board it should already be 0
    if j == 2:
      # j goes back to 0, increase i, unless i = 2, then we are done
      if i == 2:
        # the board is full, collect results and return
        state = env.getStateHash()
        ended = (env.isGameOver(True) != TTTGameOver.canPlay)
        winner = 0
        if ended:
            winner = env.isGameOver()
        results.append((state, winner, ended))
      else:
        results += getAllStates(env, i + 1, 0)
    else:
      # increment j, i stays the same
      results += getAllStates(env, i, j + 1)
  return results

def generateAllStatesTwo(env):
    ## Not finished
    states = []
    deg = env.side * env.side
    lim = 3 ** deg
    for i in range(0, 3**lim):
        state.append(i)
    ## Here we would need to convert hash to board
    ## and then take the winner and the ending

class TTTPlayer(object):
    def __init__(self, type):
        self.type = type
    def takeMove(self, environment):
        print("Not implemented")
    def clearHistory(self):
        pass
    def replay(self, environment):
        pass
    def updateHistory(self, state):
        pass

class RandomPlayer(TTTPlayer):
    def __init__(self, type):
        self.type = type

    def takeMove(self, environment):
        while True:
            rx = np.random.randint(environment.side)
            ry = np.random.randint(environment.side)
            if environment.isEmpty(rx,ry):
                return (rx, ry)
    def clearHistory(self):
        pass
    def replay(self, environment):
        pass
    def updateHistory(self, state):
        pass

class RLPlayer(TTTPlayer):
    def __init__(self, type, side = 3, eps = 0.3, alpha = 0.5):
        self.type = type
        self.eps = eps
        self.alpha = alpha
        self.valueTable = np.zeros(3**(side*side))
        self.history = []

    def clearHistory(self):
        self.history = []

    def replay(self, environment):
        ''' Only run this function when the game is Over. '''

        winner = environment.isGameOver()
        if winner == TTTGameOver.canPlay or len(self.history) < 2:
            print(winner)
            print(len(self.history))
            print("Reply failed")
            return

        target = 1 ## Assuming that I won
        if winner != self.type:
            target = 0
        ## Updating all values backwards from the terminal state
        for p in reversed(self.history):
            delta = self.alpha * (target - self.valueTable[p])
            self.valueTable[p] += delta
            target = self.valueTable[p]

        self.clearHistory()

    def updateHistory(self, state):
        self.history.append(state)

    def initializeV(self, allStates):
        for state, winner, finished in allStates:
            if finished == True:
                if winner == self.type:
                    self.valueTable[state] = 1
                elif winner == TTTGameOver.draw:
                    self.valueTable[state] = 0.6
                else:
                    self.valueTable[state] = 0
            else:
                self.valueTable[state] = 0.5

    def showDecision(self, environment):
        possibleMoves = []
        res = ""
        for i in range(0, environment.side):
            for j in range(0, environment.side):
                if environment.isEmpty(j,i):
                    environment.setCell(j,i,self.type)
                    h = environment.getStateHash()
                    environment.setCell(j,i,TTTMove.empty)
                    v = self.valueTable[h]
                    res += " |%1.2f" % v
                else:
                    c = environment.getCell(j,i)
                    if c == TTTMove.circle:
                        res += " |  O "
                    else:
                        res += " |  X "
            res += " | \n"
        return res

    def takeMove(self, environment):
        ch = np.random.rand()
        possibleMoves = []
        move = None
        for i in range(0, environment.side):
            for j in range(0, environment.side):
                if environment.isEmpty(i,j):
                    possibleMoves.append((i,j))
        if ch < self.eps:
            r = np.random.choice(len(possibleMoves))
            move = possibleMoves[ r ]
        else:
            bestV = -1
            for i in range(0, len(possibleMoves)):
                x, y = possibleMoves[i]
                environment.setCell(x,y,self.type)
                h = environment.getStateHash()
                environment.setCell(x,y,TTTMove.empty)
                if self.valueTable[h] > bestV:
                    bestV = self.valueTable[h]
                    move = (x, y)
        return move

class Human(TTTPlayer):
    def __init__(self, type):
        self.type = type

    def takeMove(self, environment):
        i = 0
        j = 0
        move = input("Enter coordinates i,j for your next move (i,j=1..3): ")
        unpack = move.split(',')
        i = int(unpack[0])
        j = int(unpack[1])
        if i <= 3 and i > 0 and j <= 3 and j > 0 and environment.isEmpty(i-1, j-1):
            print("Correct Input!")
        else:
            print("Incorrect input.")
        i -= 1
        j -= 1
        if environment.isEmpty(i, j):
            return (i, j)
        else:
            return (0 , 0)

    def clearHistory(self):
        pass
    def replay(self, environment):
        pass
    def updateHistory(self, state):
        pass

class Game:
    def __init__(self, env, player1, player2):
        self.env = env
        self.players = [player1, player2]
        self.current = 0

    def playGame(self, verbose=False):
        while self.env.isGameOver() == TTTGameOver.canPlay:
            if verbose:
                print(self.env.description())
                try:
                    r = self.players[self.current].showDecision(self.env)
                    print(r)
                except AttributeError:
                    pass
            x, y = self.players[self.current].takeMove(self.env)
            self.env.setCell(x, y, self.players[self.current].type)
            self.players[self.current].updateHistory(self.env.getStateHash())
            self.current = (self.current + 1) % 2
        return self.env.isGameOver()

print("-------------MY STUFF------------------------")
s = 3
t = TickTockToe(s)
t.internalTest()

p1 = RandomPlayer(TTTMove.cross)
p2 = RandomPlayer(TTTMove.circle)
p3 = RLPlayer(TTTMove.circle)
p4 = RLPlayer(TTTMove.cross)
p5 = Human(TTTMove.circle)

ast = getAllStates(t)
t.clearBoard()
p3.initializeV(ast)
p4.initializeV(ast)

results = [0, 0, 0]

for i in range(0,25000):
    t = TickTockToe(s)
    g = Game(t, p4, p3)
    res = g.playGame()
    results[res-1] += 1
    p3.replay(t)
    p4.replay(t)
    t.clearBoard()
    if i % 499 == 0:
        print(i)
print("Circles Won: %d" % results[0])
print("Crosses Won: %d" % results[1])
print("Draws      : %d" % results[2])
p4.eps = 0.0 # No random moves
print("-------------------------------")
print("Human games:")
for i in range(0,4):
    print("Computer first!")
    t = TickTockToe(s)
    p5.type = TTTMove.circle
    g = Game(t, p4, p5)
    res = g.playGame(True)
    print(res)
    p4.replay(t)
    t.clearBoard()
    p5.type = TTTMove.cross
    print("Human first!")
    t1 = TickTockToe(s)
    g1 = Game(t1, p5, p3)
    res = g1.playGame(True)
    print(res)
    p3.replay(t1)
    t1.clearBoard()
