# Imports.
import numpy as np
import numpy.random as npr
import pygame as pg
import sys
from sklearn.neural_network import MLPRegressor as mlp

from SwingyMonkey import SwingyMonkey

class Learner(object):

    def __init__(self, epsilon, alpha, gamma, size):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.first = True
        self.X = np.zeros((size,4))
        self.Y = np.zeros(size)
        self.nn = mlp(activation = 'logistic', warm_start = True)
        self.iteration = 0
        self.size = size


    def reset(self, epsilon, alpha):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.epsilon = epsilon
        self.alpha = alpha

    def X_factory(self, state):

        height = state['monkey']['top'] - state['tree']['top'] 
        width = state['tree']['dist'] 
        vel = state['monkey']['vel'] 
        return np.array((height, width, vel))

    def max_action(self, h, w, v):
    	swing = self.nn.predict((h,w,v,0))
    	jump = self.nn.predict((h,w,v,1))
    	return np.argmax((swing, jump))

    def decay(self, initial, lam, time):
    	return initial * exp(-lam*time)

    def action_callback(self, state):

        # Observe initial state/new state depending on first or not
        height, width, vel = self.X_factory(state)
        new_action = 0
        if (self.last_state is None and
        self.last_action is None and
        self.last_reward is None):
    		self.last_action = new_action
    		self.last_state = height, width, vel
        	#check if first instance
        	if self.first:
        		self.first = False
        		#initialize values and neural net
        		self.X[0] = np.array((height, width, vel, 0))
        		self.Y[0] = 0
        		self.nn.fit(self.X, self.Y)
        	return new_action
        	


        prob = npr.random()
        # Do epsilon greedy here to explore
        if prob < self.epsilon:
            coin_toss = npr.random()
            if coin_toss <= .5:
                # choose a random action, note they might be None
                new_action = 1
        else:
        	new_action = self.max_action(height, width, vel)

        # Use last state and last reward to update values
        sh, sw, sv = self.last_state
        index = self.iteration % self.size
        self.X[index] = np.array((sh,sw,sv,self.last_action))
        self.Y[index] = self.last_reward + self.gamma*(self.max_action(sh,sw,sv))
        self.nn.fit(self.X, self.Y)

        #update global variables
        self.last_action = new_action
        self.last_state = height, width, vel
        self.iteration += 1
        return new_action

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''

        self.last_reward = reward


def run_games(learner,iters = 300, t_len = 30):
    '''
    Driver function to simulate learning by having the agent play a sequence of games.
    '''
    scores = np.zeros(iters)
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
        #################################################
        #With max, seems to do better without the below:
        #################################################
        # #TODO: Reflect Negative Reward
        # Q_target = learner.last_reward
        # # Calculate Q_delta = Q_target - Q(s,a)
        # sh,sw,sv = learner.last_state
        # a = learner.last_action
        # Q_delta = Q_target - learner.Q[(sh,sw,sv,a)]
        # # Add alpha*Q_delta to Q(s,a) and update
        # learner.Q[(sh,sw,sv,a)] = learner.Q[(sh,sw,sv,a)] + (learner.alpha * Q_delta)
        # # Reset the state of the learner.

        scores[ii] = swing.score
        print learner.epsilon
        sh, sw, sv = learner.last_state
        index = learner.iteration % learner.size
        learner.X[index] = np.array((sh,sw,sv,learner.last_action))
        learner.Y[index] = learner.last_reward + learner.gamma*(learner.max_action(sh,sw,sv))
        learner.nn.fit(learner.X, learner.Y)
        learner.reset(learner.decay(.2, .015, ii), learner.decay(.2, .015, ii))

    pg.quit()

	return np.mean(scores[-200:]), np.max(scores[-200:])



if __name__ == '__main__':

  # Select agent.
  agent = Learner(float(sys.argv[1]),float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]))
  # Run games.
  avg,maxi = run_games(agent, sys.argv[5], 10)
  f = open('neural_net.csv', "a")
  s = sys.argv[1] + ','+ sys.argv[2] + ',' sys.argv[3] + ',' + sys.argv[4] + ',' + sys.argv[5] + ','+ str(avg) + ',' + str(maxi)+'\n'
  f.write(s)
  f.close()

