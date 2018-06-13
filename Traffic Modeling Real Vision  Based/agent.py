import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.alpha = .09
        self.gamma = 1.0

    def select_action(self, state,i_episode):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        epsilon = 1.0/i_episode
        policy_s  = np.ones(self.nA)*epsilon/self.nA
        policy_s[np.argmax(self.Q[state])] = 1 - epsilon + (epsilon / self.nA)
         
        action  = np.random.choice(np.arange(self.nA), p =policy_s)
        
        
        return action

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        self.Q[state][action] += self.alpha*(reward +(self.gamma*(np.max(self.Q[next_state]))- self.Q[state][action]))
        
                 
         
                 
                 
        #self.Q[state][action] += 1