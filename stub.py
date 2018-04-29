# Imports.
import numpy as np
import numpy.random as npr
import pygame as pg
from collections import defaultdict


from SwingyMonkey import SwingyMonkey


class Learner(object):

    def __init__(self, iterations, epsilon):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.binsize = {'height' : 10, 'width': 20, 'vel' : 10 }
        self.iterations = iterations
        self.epsilon = epsilon
        self.Q = np.zeros((400/self.binsize['height'], 600/self.binsize['width'], int(100/self.binsize['vel']), 2))

    def reset(self, iterations, epsilon):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.epsilon = epsilon / 1.01

    def indices(self, state):
        print state['monkey']['vel']
        height = int((state['monkey']['top'] - state['tree']['top'] + 200) / self.binsize['height'])
        width = int(state['tree']['dist'] / self.binsize['width'])
        vel = int((state['monkey']['vel'] + 50) / self.binsize['vel'])
        if vel > 10 :
            vel = 8
        elif vel < 0:
            vel = 0
        return (height, width, vel)

    def action_callback(self, state):

        new_state = self.indices(state)

        self.iterations += 1

        if npr.random() < .1:
            if npr.random() < .5:
                new_action = 0
            else:
                new_action = 1
        else:
            new_action = np.argmax(new_state)

        if (self.last_state is None and
            self.last_action is None and
            self.last_reward is None):
            self.last_state = new_state
            self.last_action = new_action
            self.last_reward = 0

        last_state = self.last_state
        last_action = self.last_action
        last_reward = self.last_reward

        self.last_action = new_action
        self.last_state  = new_state

        self.Q[last_state[0]][last_state[1]][last_state[2]][last_action] += self.epsilon * (last_reward + (max(self.Q[new_state[0]][new_state[1]][new_state[2]])) - self.Q[last_state[0]][last_state[1]][last_state[2]][last_action])
        # print self.Q[last_state[0]][last_state[1]][last_state[2]][last_action] #self.Q.max()
        return self.last_action
        if (self.last_state is not None and
            self.last_action is not None and
            self.last_reward is not None):
            s = self.last_state
            a = self.last_action
            r = self.last_reward
            sn = new_state

            self.Q[s[0]][s[1]][s[2]][a] += self.epsilon * (r + (max(self.Q[sn[0]][sn[1]][sn[2]])) - self.Q[s[0]][s[1]][s[2]][a])

        self.last_action = new_action
        self.last_state = new_state

        print self.Q.max()
        return new_action

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''

        self.last_reward = reward


def run_games(learner, hist, iters = 100, t_len = 100):
    '''
    Driver function to simulate learning by having the agent play a sequence of games.
    '''
    for ii in range(iters):
        # Make a new monkey object.
        swing = SwingyMonkey(sound=False,                  # Don't play sounds.
                             text="Epoch %d" % (ii),       # Display the epoch on screen.
                             tick_length = t_len,          # Make game ticks super fast.
                             action_callback=learner.action_callback,
                             reward_callback=learner.reward_callback)

        # Loop until you hit something.
        while swing.game_loop():
            pass

        # Save score history.
        hist.append(swing.score)

        # Reset the state of the learner.
        learner.reset(5, 0.5)

    pg.quit()
    return


if __name__ == '__main__':

	# Select agent.
	agent = Learner(5, 0.4)

	# Empty list to save history.
	hist = []

	# Run games.
	run_games(agent, hist, 200, 10)

	# Save history.
	np.save('hist',np.array(hist))


