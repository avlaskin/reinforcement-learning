from __future__ import print_function
import numpy as np

def print_values(V, g):
  for i in range(g.height):
    print("---------------------------")
    for j in range(g.width):
      v = V.get((j,i), 0)
      if v >= 0:
        print(" %.2f|" % v, end="")
      else:
        print("%.2f|" % v, end="") # -ve sign takes up an extra space
    print("")


def print_policy(P, g):
  for i in range(g.height):
    print("---------------------------")
    for j in range(g.width):
      a = P.get((j,i), [0 ,0, 0, 0])
      for y in a:
          print("  %.2f  - " % y, end="")
      print(" | ", end="")
    print("")

def print_policy_beautifly(P, g):
  beauty = {0: "U", 1: "D", 2: "R", 3: "L"}
  for i in range(g.height):
    print("---------------------------")
    for j in range(g.width):
      a = P.get((j,i), [0 ,0, 0, 0])
      y = np.argmax(a)
      if a[y] == 0:
          print(" x | " , end="")
      else:
          print(" %s | " % beauty[y] , end="")
    print("")
