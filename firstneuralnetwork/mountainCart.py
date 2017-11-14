# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 15:34:40 2017

@author: Gert-Jan
"""

import tensorflow as tf
import numpy as np
import tflearn
import gym
import random

from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import mean, median
from collections import Counter



GYM_NAME = 'MountainCar-v0'
env = gym.make(GYM_NAME)
n_obs = env.observation_space.shape
n_action = env.action_space.n
env.reset()

###Hyperparameters
learningRate = 1e-3 #0.001(1 -> 0.1 -> 0.01 -> 0.001)
goal_steps = 500
initial_games = 10000
score_requirement = 50

def some_random_games_first():
    for episode in range(5):
        env.reset()
        for t in range(goal_steps):
            env.render()
            action = env.action_space.sample() #take a random action
            observation, reward, done, info = env.step(action)
                #observation(object): representation of pixel data
                #reward(float): amount of reward achieved by the previous action
                #done(bool): episode is over/game over
                #info(dict): information for debugging
            if done:
                break


#some_random_games_first()

def initial_population():
    training_data = []
    scores = []
    accepted_scores = []
    for _ in range(initial_games): #episodes
        score = 0
        game_memory = []
        prev_observation = []
        for t in range(goal_steps): #steps
            #env.render()

            action = random.randrange(0, n_action)
            observation, reward, done, info = env.step(action)

            if len(prev_observation) > 0:
                game_memory.append([prev_observation, action])

            prev_observation = observation
            score += reward
            if done:
                #print("episode finished after {} steps".format(t + 1))
                break

        #wat gebeurt hier?

        accepted_scores.append(score)
        for data in game_memory:
            if data[1] == 1:
                output = [0,1]
            elif data[1] == 0:
                output = [1,0]

                training_data.append([data[0], output])

        env.reset()
        scores.append(score)

    training_data_save = np.array(training_data)
    np.save('saved.npy', training_data_save)

    return training_data

def neural_network_model(input_size):
    network = input_data(shape=[None, input_size, 1], name='input')

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 512, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, n_action, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=learningRate, loss='categorical_crossentropy', name='targets')

    model = tflearn.DNN(network, tensorboard_dir='log')

    return model

def train_model(training_data, model=False):
    #create a numpy array.
    X = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]), 1)
    y = [i[1] for i in training_data]

    if not model:
        model = neural_network_model(input_size= len(X[0]))

    model.fit({'input': X}, {'targets': y}, n_epoch=5, snapshot_step=500, show_metric=True, run_id='cart')

    return model
training_data = initial_population()
model = train_model(training_data)

scores = []
choices = []

for each_game in range(10):
    score = 0
    game_memory = []
    prev_obs = []
    env.reset()

    for _ in range(goal_steps): #500
        env.render()
        if len(prev_obs) == 0:
            action = random.randrange(0, n_action)
        else:
            action = np.argmax(model.predict(prev_obs.reshape(-1, len(prev_obs), 1)) [0])
        choices.append(action)

        new_observation, reward, done, info = env.step(action)
        prev_obs = new_observation
        game_memory.append([new_observation])
        score += reward
        if done:
            break

    scores.append(score)


