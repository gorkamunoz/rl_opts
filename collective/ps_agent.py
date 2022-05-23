import numpy as np

class PSAgent():
	"""Projective Simulation agent with two-layered network. Features: forgetting, glow, optional softmax rule. """
	
	def __init__(self, num_actions, num_percepts_list, gamma_damping, **kwargs):
		"""Initialize the basic PS agent. 
        Args:
            - num_actions: integer >=1, number of actions. 
            - num_percepts_list: list of integers >=1, not nested, representing the cardinality of each category/feature of percept space.
            - gamma_damping: float between 0 and 1, controls forgetting/damping of h-values
            
        **kwargs: 
            - eta_glow_damping: float between 0 and 1, controls the damping of glow; setting this to 1 effectively switches off glow
            - policy_type: string, 'standard' or 'softmax'; toggles the rule used to compute probabilities from h-values
            - beta_softmax: float >=0, probabilities are proportional to exp(beta*h_value). If policy_type != 'softmax', then this is irrelevant.
            - fixed_policy: list of lists, in case the user wants to fix a policy for the agent. This list contains, per percept, a list with the values of the probabilities for each action. 
                            Ex: Percept 0: fixed_policy[0] = [p(a0), p(a1), p(a2)] = [0.2, 0.3, 0.5], where a0, a1 and a2 are the three possible actions.
            
            """
            
            
		if 'eta_glow_damping' in kwargs and type(kwargs['eta_glow_damping']) is float:
			setattr(self, 'eta_glow_damping', kwargs['eta_glow_damping'])
		else:
			setattr(self, 'eta_glow_damping', 0.0)
        
		if 'policy_type' in kwargs and type(kwargs['policy_type']) is str:
			setattr(self, 'policy_type', kwargs['policy_type'])
		else:
			setattr(self, 'policy_type', 'standard')
            
		if 'beta_softmax' in kwargs and type(kwargs['beta_softmax']) is float:
			setattr(self, 'beta_softmax', kwargs['beta_softmax'])
		else:
			setattr(self, 'beta_softmax', 3)
            
		if 'fixed_policy' in kwargs and type(kwargs['fixed_policy']) is list:
			setattr(self, 'fixed_policy', kwargs['fixed_policy'])
		else:
			setattr(self, 'fixed_policy', None)
        
		self.num_actions = num_actions
		self.num_percepts_list = num_percepts_list
		self.gamma_damping = gamma_damping
		
        
		self.num_percepts = int(np.prod(np.array(self.num_percepts_list).astype(np.float64))) # total number of possible percepts
		
		
		self.h_matrix = np.ones((self.num_actions, self.num_percepts), dtype=np.float64) #Note: the first index specifies the action, the second index specifies the percept.
		
		self.g_matrix = np.zeros((self.num_actions, self.num_percepts), dtype=np.float64) #glow matrix, for processing delayed rewards
		
		
	def percept_preprocess(self, observation): # preparing for creating a percept
		"""Takes a multi-feature percept and reduces it to a single integer index.
        Input: list of integers >=0, of the same length as self.num_percept_list; 
        respecting the cardinality specified by num_percepts_list: observation[i]<num_percepts_list[i] (strictly)
        Output: single integer."""
		percept = 0
		for which_feature in range(len(observation)):
			percept += int(observation[which_feature] * np.prod(self.num_percepts_list[:which_feature]))
		return percept
		
	def deliberate(self, observation):
		"""Given an observation , this method chooses the next action and records that choice in the g_matrix.
        Arguments: 
            - observation: list of integers, as specified for percept_preprocess, 
            
        Output: action, represented by a single integer index."""        
		
		percept = self.percept_preprocess(observation) 
		action = np.random.choice(self.num_actions, p=self.probability_distr(percept)) #deliberate once
		self.g_matrix = (1 - self.eta_glow_damping) * self.g_matrix
		self.g_matrix[action, percept] += 1 #record latest decision in g_matrix
		return action	
    
	def learn(self, reward):
		"""
        Given a reward, this method updates the h matrix.
        
        Arguments:
            reward (float): given by the environment after the agent's interactions.
        """
		self.h_matrix =  self.h_matrix - self.gamma_damping * (self.h_matrix - 1.) + reward * self.g_matrix
        
        
	def probability_distr(self, percept):
		"""Given a percept index, this method returns a probability distribution over actions
        (an array of length num_actions normalized to unit sum) computed according to policy_type."""        
		if self.policy_type == 'standard':
			h_vector = self.h_matrix[:, percept]
			probability_distr = h_vector / np.sum(h_vector)
		elif self.policy_type == 'softmax':
			h_vector = self.beta_softmax * self.h_matrix[:, percept]
			h_vector_mod = h_vector - np.max(h_vector)
			probability_distr = np.exp(h_vector_mod) / np.sum(np.exp(h_vector_mod))
		return probability_distr
    
	def erase_memory(self):
		self.h_matrix = np.ones((self.num_actions, self.num_percepts), dtype=np.float64)
        
	def reset_g(self):
		self.g_matrix = np.zeros((self.num_actions, self.num_percepts), dtype=np.float64) #glow matrix, for processing delayed rewards

	def deliberate_fixed_policy(self, observation):
		"""Given an observation , this method chooses the next action according to the fixed policy specified as attribute of the class.
        Arguments: 
            - observation: list of integers, as specified for percept_preprocess, 
            
        Output: action, represented by a single integer index."""        
		
		percept = self.percept_preprocess(observation) 
		if self.fixed_policy:
			action = np.random.choice(self.num_actions, p=self.fixed_policy[percept]) #deliberate once
		else:
			print('No fixed policy was given to the agent. The action will be selected randomly.')
			action = np.random.choice(self.num_actions)
        
		self.g_matrix = (1 - self.eta_glow_damping) * self.g_matrix
		self.g_matrix[action, percept] += 1 #record latest decision in g_matrix
        
		return action       
