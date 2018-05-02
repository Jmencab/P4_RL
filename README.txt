Run stub.py in the usual way:
python stub.py

Run neural_net.py with the following arguments:
python neural_net.py epsilon alpha gamma size epochs

Where:
-0 < epsilon <= 1.0 is a float used in epsilon-greedy
-0 < alpha <= 1.0 is a float is the learning parameter in Q-learning
-0 < gamma <= 1.0 is a float used as the discount factor
-100 <= size <= 400 is and int that is the length of the running history of states and Q-scores put into the neural net
-epochs is an int that specifies the number of iterations of the game 