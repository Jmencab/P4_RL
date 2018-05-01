# Imports.
import numpy as np
import numpy.random as npr
import pygame as pg
from collections import defaultdict
from pprint import pprint
from math import exp


from SwingyMonkey import SwingyMonkey


class Learner(object):

    def __init__(self,epsilon,alpha,gamma):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.binsize = {'height' : 20, 'width': 30, 'vel' : 10 }
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma  = gamma
        self.Q  = dict()

    def reset(self, epsilon, alpha):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.epsilon = epsilon
        self.alpha = alpha

    def closest(self, height, width, vel, act):
        MAX = 5
        def dist(s, t):
            a, b, c, d = s
            e, f, g, h = t
            if d == h:
                return (a-e)**2 + (b-f)**2 + (c-g)**2
            else:
                return np.inf
        states = self.Q.keys()
        min_dist = np.inf
        value = []
        for state in states:
            d = dist(state, (height, width, vel, act))
            if d <= min_dist and d < MAX:
                min_dist = d
                a,b,c,d = state
                value.append(self.Q[state])
        print value
        if len(value) == 0:
            return 0
        else:
            return self.alpha * (sum(value)/len(value))

    def mass_update(self, Q_delta, sh, sw, sv, a):
        MAX = 2
        for i in range(MAX):
            for j in range(MAX):
                for k in range(MAX):
                    self.Q[(sh+i,sw+j,sv+k,a)] = self.Q[(sh+i,sw+j,sv+k,a)] + (self.alpha/(i+j+k+2) * Q_delta)
                    self.Q[(sh+i,sw+j,sv-k,a)] = self.Q[(sh+i,sw+j,sv-k,a)] + (self.alpha/(i+j+k+2) * Q_delta)
                    self.Q[(sh-i,sw+j,sv+k,a)] = self.Q[(sh-i,sw+j,sv+k,a)] + (self.alpha/(i+j+k+2) * Q_delta)
                    self.Q[(sh+i,sw-j,sv+k,a)] = self.Q[(sh+i,sw-j,sv+k,a)] + (self.alpha/(i+j+k+2) * Q_delta)
                    self.Q[(sh-i,sw-j,sv+k,a)] = self.Q[(sh-i,sw-j,sv+k,a)] + (self.alpha/(i+j+k+2) * Q_delta)
                    self.Q[(sh-i,sw+j,sv-k,a)] = self.Q[(sh-i,sw+j,sv-k,a)] + (self.alpha/(i+j+k+2) * Q_delta)
                    self.Q[(sh+i,sw-j,sv-k,a)] = self.Q[(sh+i,sw-j,sv-k,a)] + (self.alpha/(i+j+k+2) * Q_delta)
                    self.Q[(sh-i,sw-j,sv-k,a)] = self.Q[(sh-i,sw-j,sv-k,a)] + (self.alpha/(i+j+k+2) * Q_delta)

    def indices(self, state):

        height = int((state['monkey']['top'] - state['tree']['top']) / self.binsize['height'])
        width = int(state['tree']['dist'] / self.binsize['width'])
        vel = int(state['monkey']['vel'] / self.binsize['vel'])
        return height, width, vel

    def decay(self, initial, lam, time):
        return initial * exp(-lam*time)

    def action_callback(self, state):
        # Observe initial state/new state depending on first or not
        h,w,v = self.indices(state)
        # Need to observe value first, add to dictionary
        # This is like the base case
        if (self.last_state is None and
        self.last_action is None and
        self.last_reward is None):
            self.last_state = (h,w,v)
            if not self.Q.get((h,w,v,0)):
                self.Q[(h,w,v,0)] = 0
            if not self.Q.get((h,w,v,1)):
                self.Q[(h,w,v,1)] = 0
            action = np.argmax((self.Q.get((h,w,v,0), self.Q.get((h,w,v,1)))))
            self.last_action = action
            return action

        #Figure out what action to take
        swing_Q = self.Q.get((h,w,v,0))
        if not swing_Q:
            if len(self.Q) == 0:
                swing_Q = 0
                self.Q[(h,w,v,0)] = 0
            else:
                swing_Q, self.Q[(h,w,v,0)] = self.closest(h,w,v,0), self.closest(h,w,v,0)
        jump_Q = self.Q.get((h,w,v,1))
        if not jump_Q:
            if len(self.Q) == 0:
                jump_Q = 0
                self.Q[(h,w,v,1)] = 0
            else:
                jump_Q, self.Q[(h,w,v,1)] = self.closest(h,w,v,1), self.closest(h,w,v,1)
        new_action = 0
        prob = npr.random()
        # Do epsilon greedy here to explore
        if prob < self.epsilon:
            coin_toss = npr.random()
            if coin_toss <= .5:
                # choose a random action, note they might be None
                new_action = 1
        else:
            if jump_Q > swing_Q:
                new_action = 1

        # Get last state and last reward
        sh, sw, sv = self.last_state
        r = self.last_reward
        a = self.last_action
        # Calculate Q_target = r + gamma*max[Q(s',A)]
        Q_target = r + self.gamma*max(self.Q[(h,w,v,0)], self.Q[(h,w,v,1)])
        # Calculate Q_delta = Q_target - Q(s,a)
        Q_delta = Q_target - self.Q[(sh,sw,sv,a)]
        # Add alpha*Q_delta to Q(s,a) and update
        self.Q[(sh,sw,sv,a)] = self.Q[(sh,sw,sv,a)] + (self.alpha * Q_delta)
        self.mass_update(Q_delta, sh, sw, sv, a)
        #update global variables
        self.last_state = (h,w,v)
        self.last_action = new_action
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

        #TODO: Reflect Negative Reward
        Q_target = learner.last_reward
        # Calculate Q_delta = Q_target - Q(s,a)
        sh,sw,sv = learner.last_state
        a = learner.last_action
        Q_delta = Q_target - learner.Q[(sh,sw,sv,a)]
        # Add alpha*Q_delta to Q(s,a) and update
        learner.Q[(sh,sw,sv,a)] = learner.Q[(sh,sw,sv,a)] + (learner.alpha * Q_delta)
        # Reset the state of the learner.

        scores[ii] = swing.score
        learner.reset(learner.decay(1, .015, ii), learner.decay(1, .015, ii))

    print "-------------------------"
    print "R:"
    print learner.last_reward
    print "S:"
    print learner.last_state
    print "A:"
    print learner.last_action
    print "Q:"
    pprint(learner.Q)
    pg.quit()

    # pipe results of games
    return


if __name__ == '__main__':

  # Select agent.
  agent = Learner(.9,.9, 1)
  # Run games.
  run_games(agent, 400, 10)
