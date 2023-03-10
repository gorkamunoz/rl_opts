# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/lib_nbs/04_imitation_learning.ipynb.

# %% auto 0
__all__ = ['PS_imitation']

# %% ../nbs/lib_nbs/04_imitation_learning.ipynb 2
import numpy as np

# %% ../nbs/lib_nbs/04_imitation_learning.ipynb 4
class PS_imitation():
    def __init__(self, 
                 num_states: int, # Number of states 
                 eta: float, # Glow parameter of PS
                 gamma: float # Damping parameter of PS
                ):     
        '''Constructs a PS agent with two actions (continue and rotate) that performs imitation learning 
        in the search scenario. Instead of following a full trajectory of action-state tuples, the agent 
        is directly given the reward state (the step length in this case). The agent updates all previous
        continue actions and the current rotate action.        
        '''
        
        self.num_states = num_states        
        self.eta = eta
        self.gamma_damping = gamma
        
        # h-matrix
        self.h_matrix = np.ones((2, self.num_states)).astype(float)
        # initiate glow matrix
        self.reset()
        
    def reset(self):
        '''Resets the glow matrix'''
        self.g_matrix = np.zeros((2, self.num_states)).astype(float)        
        
        
    def update(self, 
               length: int, # Step length rewarded
               reward: int = 1 # Value of the reward
              ):
        '''
        Updates the policy based on the imitation scheme (see paper for detailes)      
        NOTE: state is length-1 because counter starts in 0 
        (but in 0, agent has already performed a step of length 1 -- from the previous action "rotate").
        '''

        factor = 1 - self.eta
        # ---- Updating the CONTINUE part of g-matrix ---- 
        # Damping before adding up the traversed edges.
        self.g_matrix[0, :length-1] *= (factor**np.arange(1,length))
        # Set to one all previous states (adding up the traversed edges)
        self.g_matrix[0, :length-1] += 1   
        # Multiply by eta**x all previous states
        self.g_matrix[0, :length-1] *= (factor**np.arange(1,length))[::-1]
        # Multiply the rest of the matrix by number of steps don
        self.g_matrix[0, length-1:] *= factor**length

        # ---- Updating the TURN part of g-matrix ---- 
        self.g_matrix[1, :] *= factor**length
        self.g_matrix[1, length-1] += 1

        # Apply damping
        if self.gamma_damping > 0:
            for _ in range(length):
                self.h_matrix -= self.gamma_damping*(self.h_matrix - 1.)

        # Apply reward
        self.h_matrix += self.g_matrix*reward