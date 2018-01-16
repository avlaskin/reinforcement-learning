import random
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

class Bandit:
    def __init__(self,average):
        self.mean = float(average)
        self.N = int(0)
        self.average = 100

    def pull(self):
        return np.random.randn() + self.mean

    def update(self, x):
        self.N = self.N + 1
        self.average = (1 - 1.0 / self.N) * self.average +  float(x) / self.N
        print("Av: %f Mean: %f Action: %f" % (self.average, self.mean, x))

class Experement:
    def __init__(self, m1,m2, m3, itterations, t):
        self.bs =  [Bandit(m1), Bandit(m2), Bandit(m3)]
        self.itterations = itterations
        self.data = np.empty(itterations)
        self.choice = np.empty(itterations)
        self.t = t

    def run(self, epsilon):
        self.data = np.empty(self.itterations)
        self.choice = np.empty(self.itterations)
        for i in range(0, self.itterations):
            r = np.random.randint(0,100)
            ch = 0.0
            x = float(0.0)
            best = self.chooseBest(r,epsilon,i)
            x = self.bs[best].pull()
            self.bs[best].update(x)
            self.choice[i] = best
            self.data[i] = x

    def chooseBest(self, r, epsilon, i):
        if self.t == 0:
            if r < epsilon:
                best = np.random.randint(0,3)
                return best
            else:
                best = np.argmax([b.average for b in self.bs])
                return best
        elif self.t == 1:
            return np.argmax([b.average + np.sqrt( 2 * np.log(i) / (b.N + 1)) for b in self.bs])
        else:
            return np.argmax([b.average for b in self.bs])

    def description(self):
        for i in range(0,len(self.bs)):
            print("Average : %f App. Average: %f" % (self.bs[i].mean, self.bs[i].average) )
N= 20000
e1 = Experement(1.0,2.0,3.0,N,0)
e2 = Experement(1.0,2.0,3.0,N,1)
e3 = Experement(1.0,2.0,3.0,N,2)

e1.run(10)
e2.run(0)
e3.run(0)

cumaverage1 = np.cumsum(e1.data) / (np.arange(N) + 1)
cumaverage2 = np.cumsum(e2.data) / (np.arange(N) + 1)
cumaverage3 = np.cumsum(e3.data) / (np.arange(N) + 1)

fig, ax = plt.subplots()
ax.plot(cumaverage1, color="r")
ax.plot(cumaverage2, color="g")
ax.plot(cumaverage3, color="b")
plt.xscale('log')
plt.show()
