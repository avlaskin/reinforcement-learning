# Reinforcement Learning - learning repository

Here there will be a bunch of experimental algorithms implemented using different methods of the modern reinforcement learning. Mainly influenced by Udemy course on RL.

## Environment

You can setup the environment using virtualenv and requirements file or using docker container for more reproducible and reliable results.

### Virtualenv

```
mkdir env
cd env
virtualenv .
source bin\activate
cp ../docker/requirements.txt ./
pip install -r requirements.txt
```

## Docker

Here we use technic to speedup docker rebuild. Run ``build.sh``` only once to make a new container and ```runs.sh``` as many times as you change the app.

## Bandit

Exploration/Exploitation dilema.

## Tic Toc Toe

Simple Time Difference method for this game.

## Cart Pole

*Random* is a linear solution approximation where we are trying to find a vector such as when  multiplied by state to give us a scalar value that can be used as decisiom boundary. More than zero - go right, less - go left. This solution is certainly full of assumtions but in practice still gives us good solution.

*Bins* is a table Q-learning method that converts state space into 10000 bins table. That allows us to use general table methods like Q-leaning to solve the cartpole. Distretisising the state space is a very useful technic and certainly can be applied to many different problems.

*DQN* is the early days attempt to solve cartpole problem using deep Q-Learning with neural network implemented in Keras. 
