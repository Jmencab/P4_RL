import SwingyMonkey
import numpy as np

# self.screen_width  = 600
# self.screen_height = 400
# self.gravity       = npr.choice([1,4])
# self.tree_gap      = 200

# example
x = { 'score': 0,
  'tree': { 'dist': 100,
            'top': 300,
            'bot': 100},
  'monkey': { 'vel': 0,
              'top': 100,
              'bot': 0}}

BINSIZE = 10

def make_statespace () :
  return np.zeros((60, 40, 40))

def indices (state):
  tree = state['tree']['top']
  monkey = state['monkey']['top']
  dist = state['tree']['dist']
  return int(tree/BINSIZE), int(monkey/BINSIZE), int(dist/BINSIZE)

ind = indices (x)
a = make_statespace ()
print a[ind[0]][ind[1]][ind[2]]

