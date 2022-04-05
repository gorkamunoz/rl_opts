
import numpy as np
from ps_agent import PSAgent


class Forager(PSAgent):
    ''' At every time step, the agent decides if it rotates or continues in the current direction'''
    
    def __init__(self, visual_cone, visual_radius, state_space, num_actions, gamma, **kwargs):        
        ''' Class constructor
        visual_cone (float): angle (rad) of visual cone in front of the agent.
        visual_radius (int): radius of visual region.
        state_space (list): list where each entry is the state space of each perceptual feature.
            E.g. [state space of step counter, state space of density of successful neighbours].
        
        '''
        
        self.visual_cone = visual_cone
        self.visual_radius = visual_radius
        self.state_space = state_space
        
        # num_states = int(np.prod(np.array([len(i) for i in self.state_space]).astype(np.float64)))
        num_states_list = [len(i) for i in self.state_space]
        
        super().__init__(num_actions, num_states_list, gamma, **kwargs)
        
        self.agent_state = 0
    
    def act(self, action):
        """
        Agent performs the given action.

        Parameters
        ----------
        action : (int: 0 ,1)
            1 if it changes direction, 0 otherwise
        """
        
        # If the agent changes direction   
        if action == 1:
            self.agent_state = 0
        
        self.agent_state += 1        
        

    
    def get_state(self, visual_perception):
        """
        Gets the total state of the agent,
        combining the internal perception (#steps in same direction) and the external information of the other agents.
                                                                             
        Parameters
        ----------
        visual_perception : (list) list with the visual perception of surrounding agents.
            [density rewarded agents in front, density of rewarded agents at the back]

        Returns
        -------
        Final state (list), of the form: [internal state, external visual information]

        """
        #state related to step counter
        internal_state = list(np.argwhere((self.state_space[0] - self.agent_state) <= 0)[-1])
        
        return internal_state + visual_perception
        
    