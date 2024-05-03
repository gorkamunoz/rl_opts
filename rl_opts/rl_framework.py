# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/lib_nbs/01_rl_framework.ipynb.

# %% auto 0
__all__ = ['TargetEnv', 'PSAgent', 'Forager']

# %% ../nbs/lib_nbs/01_rl_framework.ipynb 5
try:
    import torch
except:
    import warnings
    warnings.warn('torch is not installed, you will need it for the classic rl_opts.rl_framework version')
    
import numpy as np

from .utils import isBetween_c_Vec, coord_mod

# %% ../nbs/lib_nbs/01_rl_framework.ipynb 6
class TargetEnv():
    def __init__(self,
                 Nt,
                 L,
                 r,
                 lc,
                 agent_step = 1,
                 boundary_condition = 'periodic',
                 num_agents = 1,
                 high_den = 5,
                 destructive = False):
        
        """Class defining the foraging environment. It includes the methods needed to place several agents to the world.
        
        Parameters
        ----------
        Nt: int 
            Number of targets.
        L: int
            Size of the (squared) world.
        r: int 
            Radius with center the target position. It defines the area in which agent detects the target.
        lc: int
            Cutoff length. Displacement away from target (to implement revisitable targets by displacing agent away from the visited target).
        agent_step: int, optional 
            Displacement of one step. The default is 1.
        boundary_conditions: str, optional
            If there is an ensemble of agents, this is automatically set to periodic. The default is 'periodic'. 
        num_agents: int, optional 
            Number of agents that forage at the same time. The default is 1.
        high_den: int, optional
            Number of agents from which it is considered high density. Useful for the case with num_agents >> 1. The default is 5. 
        destructive: bool, optional
            True if targets are destructive. The default is False.
        """
        self.Nt = Nt
        self.L = L
        self.r = r
        self.lc = lc
        self.agent_step = agent_step 
        self.boundary_condition = (boundary_condition if num_agents == 1 else 'periodic')
        self.num_agents = num_agents
        self.high_den = high_den
        self.destructive_targets = destructive
        
        self.init_env()
        
        
    def init_env(self):
        """
        Environment initialization.
        """
        self.target_positions = torch.rand(self.Nt, 2)*self.L
        
        #store who is/was rewarded
        self.current_rewards = torch.zeros(self.num_agents)
        self.last_step_rewards = torch.zeros(self.num_agents)
        
        #signal whether agent has been kicked
        self.kicked = torch.zeros(self.num_agents)
        
        #set positions and directions of the agents
        self.current_directions = torch.zeros(self.num_agents)
        self.positions = torch.zeros(self.num_agents, 2)
        for ag in range(self.num_agents):
            self.current_directions[ag] = torch.rand(1)*2*np.pi
            self.positions[ag] = torch.rand(2)*(self.L) 
        self.previous_pos = self.positions.clone()
          
        
    def update_pos(self, change_direction, agent_index=0):        
        """
        Updates information of the agent depending on its decision.

        Parameters
        ----------
        change_direction : bool
            Whether the agent decided to turn or not.
        agent_index : int, optional
            Index of the given agent. The default is 0.
        """
        # Save previous position to check if crossing happened
        self.previous_pos[agent_index] = self.positions[agent_index].clone()
        
        if change_direction:
            self.current_directions[agent_index] = torch.rand(1)*2*np.pi
        
        #Update position
        self.positions[agent_index][0] = self.positions[agent_index][0] + self.agent_step*np.cos(self.current_directions[agent_index])
        self.positions[agent_index][1] = self.positions[agent_index][1] + self.agent_step*np.sin(self.current_directions[agent_index])
        
       
    def check_encounter(self, agent_index=0):
        """
        Checks whether the agent found a target, and updates the information accordingly.

        Parameters
        ----------
        agent_index : int, optional

        Returns
        -------
        True if the agent found a target.

        """
        
        encounters = isBetween_c_Vec(self.previous_pos[agent_index], self.positions[agent_index], self.target_positions, self.r)
        
        self.last_step_rewards[agent_index] = self.current_rewards[agent_index].clone()
        
        if sum(encounters) > 0: 
            
            #if there is more than 1 encounter, pick the closest to the agent.
            if sum(encounters) == 1:
                first_encounter = np.arange(len(self.target_positions))[encounters]
            else:
                # compute the distance from the previous position to each target            
                distance_previous_pos = np.sqrt((self.previous_pos[agent_index][0]- self.target_positions[:, 0])**2 + (self.previous_pos[agent_index][1] - self.target_positions[:, 1])**2)            
                
                # checking which encountered point is closer to previous position
                min_distance_masked = np.argmin(distance_previous_pos[encounters])
                first_encounter = np.arange(len(self.target_positions))[encounters][min_distance_masked]
            
            #if targets are destructive, remove the found target
            if self.destructive_targets:
                self.target_positions[first_encounter] = torch.rand(2)*self.L
            else:
                #----KICK----
                # If there was encounter, we reset direction and change position of particle to (pos target + lc)
                kick_direction = np.random.uniform(low = 0, high = 2*np.pi)
                self.positions[agent_index][0] = self.target_positions[first_encounter, 0] + self.lc*np.cos(kick_direction)
                self.positions[agent_index][1] = self.target_positions[first_encounter, 1] + self.lc*np.sin(kick_direction)
                self.kicked[agent_index] = 1
                #------------
                
            #...and we add the information that this agent got to the target
            self.current_rewards[agent_index] = 1
              
            return 1
        
        else: 
            self.kicked[agent_index] = 0
            self.current_rewards[agent_index] = 0
            return 0
        
    def check_bc(self, agent_index=0):
        """
        Updates position coordinates of agent agent_index to fulfill periodic boundary conditions.

        """
        self.positions[agent_index] = (self.positions[agent_index])%self.L
        

    def get_neighbors_state(self, focal_agent, visual_cone, visual_radius):
        """
        Gets the visual information of the agents surrounding the focal agent.

        Parameters
        ----------
        focal_agent : int 
            Index of focal agent.
        visual_cone : float 
            Angle (rad) of the visual cone in front of the agent.
        visual_radius : int/float
            Radius of the visual circular region around the agent. 
            
        Returns
        -------
        State: list
            [density of rewarded agents in front, same at the back].
            density values: 0 -- no rewarded agent, 1 -- low # of rewarded agents, 2 -- more than high_den rewarded agents.

        """
        mask_in_sight, mask_behind = self.get_agents_in_sight(focal_agent, visual_cone, visual_radius)
        
        num_success_infront = torch.sum(mask_in_sight * self.last_step_rewards)
        num_success_behind = torch.sum(mask_behind * self.last_step_rewards)
              
        return [np.argwhere((np.array([0, 1, self.high_den, self.num_agents]) - int(num_success_infront)) <= 0)[-1][0],
                np.argwhere((np.array([0, 1, self.high_den, self.num_agents]) - int(num_success_behind)) <= 0)[-1][0]]
        
    
    def get_agents_in_sight(self, focal_agent, visual_cone, visual_radius):
        """
        Get which agents are within the front visual cone of the focal agent.

        Parameters
        ----------
        focal_agent : int
            Index of focal agent.
        visual_cone : float
            Angle (rad) of the visual cone in front of the agent.
        visual_radius : int/float
            Radius of the visual circular region around the agent. 
            

        Returns
        -------
        mask_in_sight : torch.tensor of boolean values
            True at the indices of agents that are within the visual cone in front.
        mask_behind : torch.tensor of boolean values
            True at the indices of agents that are within the visual range, but outside the front cone.

        """
        
        y = coord_mod(self.previous_pos[:,1], self.positions[focal_agent,1], self.L)
        x = coord_mod(self.previous_pos[:,0], self.positions[focal_agent,0], self.L)
        
        mask_inside_radius = np.sqrt(x**2 + y**2) < visual_radius
        
        mask_in_sight = (np.abs((np.arctan2(y,x) + 2*np.pi) % (2*np.pi) - self.current_directions[focal_agent]) < visual_cone / 2 ) * mask_inside_radius
        mask_behind = mask_inside_radius ^ mask_in_sight
        
        mask_in_sight[focal_agent] = False
        mask_behind[focal_agent] = False
        
        return mask_in_sight, mask_behind
    

# %% ../nbs/lib_nbs/01_rl_framework.ipynb 8
class PSAgent():
    
    def __init__(self, num_actions, 
                 num_percepts_list, 
                 gamma_damping=0.0, 
                 eta_glow_damping=0.0, 
                 policy_type='standard', 
                 beta_softmax=3, 
                 initial_prob_distr=None, 
                 fixed_policy=None):
        """
        Base class of a Reinforcement Learning agent based on Projective Simulation,
        with two-layered network. This class has been adapted from https://github.com/qic-ibk/projectivesimulation

        Parameters
        ----------
        num_actions : int >=1
            Number of actions.
        num_percepts_list : list of integers >=1, not nested
            Cardinality of each category/feature of percept space.
        gamma_damping : float (between 0 and 1), optional
            Forgetting/damping of h-values at the end of each interaction. The default is 0.0.
        eta_glow_damping : float (between 0 and 1), optional
            Controls the damping of glow; setting this to 1 effectively switches off glow. The default is 0.0.
        policy_type : string, 'standard' or 'softmax', optional
            Toggles the rule used to compute probabilities from h-values. See probability_distr. The default is 'standard'.
        beta_softmax : float >=0, optional
            Probabilities are proportional to exp(beta*h_value). If policy_type != 'softmax', then this is irrelevant. The default is 3.
        initial_prob_distr : list of lists, optional
            In case the user wants to change the initialization policy for the agent. This list contains, per percept, a list with the values of the initial h values for each action. The default is None.
        fixed_policy : list of lists, optional
            In case the user wants to fix a policy for the agent. This list contains, per percept, a list with the values of the probabilities for each action. 
            Example: Percept 0: fixed_policy[0] = [p(a0), p(a1), p(a2)] = [0.2, 0.3, 0.5], where a0, a1 and a2 are the three possible actions. The default is None.

        """
        
        self.num_actions = num_actions
        self.num_percepts_list = num_percepts_list
        self.gamma_damping = gamma_damping
        self.eta_glow_damping = eta_glow_damping
        self.policy_type = policy_type
        self.beta_softmax = beta_softmax
        self.initial_prob_distr = initial_prob_distr
        self.fixed_policy = fixed_policy
        
        self.num_percepts = int(np.prod(np.array(self.num_percepts_list).astype(np.float64))) # total number of possible percepts
        
        self.h_matrix = np.ones((self.num_actions, self.num_percepts), dtype=np.float64) #Note: the first index specifies the action, the second index specifies the percept.
        
        self.g_matrix = np.zeros((self.num_actions, self.num_percepts), dtype=np.float64) #glow matrix, for processing delayed rewards
        
        #initialize h matrix with different values
        if self.initial_prob_distr:
            self.h_0 = np.ones((self.num_actions, self.num_percepts), dtype=np.float64)
            
            for percept_index, this_percept_prob_distr in enumerate(self.initial_prob_distr):
                self.h_0[:, percept_index] = this_percept_prob_distr
                
            self.h_matrix = np.copy(self.h_0)
            
        
    def percept_preprocess(self, observation):
        """
        Takes a multi-feature percept and reduces it to a single integer index.

        Parameters
        ----------
        observation : list of integers >=0, of the same length as self.num_percepts_list
            List that describes the observation. Each entry is the value that each feature takes in the observation.
            observation[i] < num_percepts_list[i] (strictly)

        Returns
        -------
        percept : int
            Percept index that corresponds to the input observation.

        """
        
        percept = 0
        for which_feature in range(len(observation)):
            percept += int(observation[which_feature] * np.prod(self.num_percepts_list[:which_feature]))
        return percept
    
    def deliberate(self, observation):
        """
        Given an observation , this method chooses the next action and records that choice in the g_matrix.

        Parameters
        ----------
        observation : list
            List that describes the observation, as specified in percept_preprocess.

        Returns
        -------
        action : int
            Index of the chosen action.

        """
        percept = self.percept_preprocess(observation) 
        action = np.random.choice(self.num_actions, p=self.probability_distr(percept)) #deliberate once
        self.g_matrix = (1 - self.eta_glow_damping) * self.g_matrix
        self.g_matrix[action, percept] += 1 #record latest decision in g_matrix
        return action
    
    def learn(self, reward):
        """
        Given a reward, this method updates the h matrix.

        Parameters
        ----------
        reward : float
            Value of the obtained reward.
        """
        if self.initial_prob_distr:
            self.h_matrix =  self.h_matrix - self.gamma_damping * (self.h_matrix - self.h_0) + reward * self.g_matrix
        else:
            self.h_matrix =  self.h_matrix - self.gamma_damping * (self.h_matrix - 1.) + reward * self.g_matrix
    
    def probability_distr(self, percept):
        """
        Given a percept index, this method returns a probability distribution over actions.

        Parameters
        ----------
        percept : int
            Index of the given percept.

        Returns
        -------
        probability_distr : np.array, length = num_actions
            Probability for each action (normalized to unit sum), computed according to policy_type.

        """
        
        if self.policy_type == 'standard':
            h_vector = self.h_matrix[:, percept]
            probability_distr = h_vector / np.sum(h_vector)
        elif self.policy_type == 'softmax':
            h_vector = self.beta_softmax * self.h_matrix[:, percept]
            h_vector_mod = h_vector - np.max(h_vector)
            probability_distr = np.exp(h_vector_mod) / np.sum(np.exp(h_vector_mod))
        return probability_distr
    
    def reset_g(self):
        """
        Resets the g_matrix.
        """
        self.g_matrix = np.zeros((self.num_actions, self.num_percepts), dtype=np.float64)
        
    def deliberate_fixed_policy(self, observation):
        """
        Given an observation , this method chooses the next action according to the fixed policy specified as attribute of the class.

        Parameters
        ----------
        observation : list
            List that describes the observation, as specified in percept_preprocess.

        Returns
        -------
        action : int
            Index of the chosen action.

        """
        percept = self.percept_preprocess(observation) 
        if self.fixed_policy:
            action = np.random.choice(self.num_actions, p=self.fixed_policy[percept])
        else:
            print('No fixed policy was given to the agent. The action will be selected randomly.')
            action = np.random.choice(self.num_actions)
    
        self.g_matrix = (1 - self.eta_glow_damping) * self.g_matrix
        self.g_matrix[action, percept] += 1 #record latest decision in g_matrix
    
        return action


# %% ../nbs/lib_nbs/01_rl_framework.ipynb 10
class Forager(PSAgent):
    
    def __init__(self, state_space, num_actions, visual_cone= np.pi, visual_radius=1.0, **kwargs):
        """
        This class extends the general `PSAgent` class and adapts it to the foraging scenario·

        Parameters
        ----------
        state_space : list
            List where each entry is the state space of each perceptual feature.
            E.g. [state space of step counter, state space of density of successful neighbours].
        num_actions : int
            Number of actions.
        visual_cone : float, optional
            Visual cone (angle, in radians) of the forager, useful in scenarios with ensembles of agents. The default is np.pi.
        visual_radius : float, optional
            Radius of the visual region, useful in scenarious with ensembles of agents. The default is 1.0.
        **kwargs : multiple
            Parameters of the class that defines the learning agent.

        """
        
        self.state_space = state_space
        self.visual_cone = visual_cone
        self.visual_radius = visual_radius
        
        num_states_list = [len(i) for i in self.state_space]
        
        super().__init__(num_actions, num_states_list, **kwargs)
        
        #initialize the step counter n
        self.agent_state = 0
    
    def act(self, action):
        """
        Agent performs the given action.

        Parameters
        ----------
        action : int (0, 1)
            1 if it changes direction, 0 otherwise
        """
        
        # If the agent changes direction   
        if action == 1:
            self.agent_state = 0
        else:
            self.agent_state += 1        
        

    
    def get_state(self, visual_perception=[0,0]):
        """
        Gets the total state of the agent, combining the internal perception (#steps in same direction)
        and the external information of the other agents.
                                                                             
        Parameters
        ----------
        visual_perception : list, optional 
            List with the visual perception of surrounding agents,
            [density rewarded agents in front, density of rewarded agents at the back]. 
            The default is [0,0], for when there is only one agent.

        Returns
        -------
        Final state: list
            [internal state, external visual information]

        """

        #state related to step counter
        internal_state = list(np.argwhere((self.state_space[0] - self.agent_state) <= 0)[-1])
        
        return internal_state + visual_perception
        
    
