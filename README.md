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
