# Imports.
import numpy as np
import numpy.random as npr
import pygame as pg

from SwingyMonkey import SwingyMonkey


class Learner(object):

    def __init__(self, iterations, epsilon):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.binsize = {'height' : 10, 'width': 10, 'vel' : 5 }
        self.iterations = iterations
        self.epsilon = epsilon
        self.Q = np.zeros((400/binsize['height'], 600/binsize['width'], 25, 2))

    # def reset(self, iterations, epsilon):
    #     self.last_state  = None
    #     self.last_action = None
    #     self.last_reward = None
    #     self.binsize = {'height' : 10, 'width': 10, 'vel' : 5 }
    #     self.epsilon = epsilon

    def indices(self, state):
        height = int((state['tree']['top'] - state['monkey']['top']) / binsize['height'])
        width = int(state['tree']['dist'] / binsize['width'])
        vel = int(state['monkey']['vel'] / binsize['vel'])
        return height, width, vel

    def action_callback(self, state):

        # You might do some learning here based on the current state and the last state.


        new_state = self.indices(state)
        self.iterations += 1
        new_action = np.argmax(self.Q[new_state])

        if (self.last_state is not None and 
        self.last_action is not None and 
        self.last_reward is not None):
            last_state = self.last_state
            last_action = self.last_action
            last_reward = self.last_reward

        self.last_action = new_action
        self.last_state  = new_state

        self.Q[s][a] += self.epsilon * (last_reward + (max(self.Q[new_state])) - self.Q[s][a])

        return self.last_action

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
        learner.reset()
    pg.quit()
    return


if __name__ == '__main__':

	# Select agent.
	agent = Learner()

	# Empty list to save history.
	hist = []

	# Run games.
	run_games(agent, hist, 20, 10)

	# Save history.
	np.save('hist',np.array(hist))


