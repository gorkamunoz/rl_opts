import torch
import numpy as np


class TargetEnv():
    def __init__(self,
                 Nt = 1000,
                 L = 200,
                 at = 1,
                 ls = 2,
                 tau = 5,
                 agent_step = 1,
                 boundary_condition = 'periodic',
                 num_agents = 1,
                 high_den = 5):
        
        """Class defining the foraging environment. It includes the methods needed to incorporate
        several agents to the world. 
        Args:
            Nt: (int) number of targets 
            L: (int) size of the (squared) world
            at: (int) radius with center the target position. It defines the area in which agent detects the target.
            ls: (int) displacement away from target (to implement revisitable targets by kicking agent away from the visited target).
            tau: (int) time steps after which target is regenerated.
            agent_step: (int) displacement of one step.
            boundary_conditions: (str) default: periodic. If there is an ensemble of agents, this is automatically set to periodic.
            num_agents: (int) number of agents.
            high_den: (int) number of agents from which it is considered high density.
        """
        self.Nt = Nt
        self.L = L
        self.at = at
        self.ls = ls
        self.tau = tau
        self.agent_step = agent_step 
        self.boundary_condition = (boundary_condition if num_agents == 1 else 'periodic')
        self.num_agents = num_agents
        self.high_den = high_den
        
        self.init_env()
        
        
        
    def init_env(self):
        """
        Environment initialization.
        """
        self.target_positions = torch.rand(self.Nt, 2)*self.L
        
        #store who was/is rewarded
        self.current_rewards = torch.zeros(self.num_agents)
        self.last_step_rewards = torch.zeros(self.num_agents)
        
        #time counters for time delay with tau (for revisitable targets, instead of kick)
        self.time_counters = torch.zeros(self.num_agents, self.Nt)
        
        #set positions and directions of the agents
        self.current_directions = torch.zeros(self.num_agents)
        self.positions = torch.zeros(self.num_agents, 2)
        for ag in range(self.num_agents):
            self.current_directions[ag] = torch.rand(1)*2*np.pi
            self.positions[ag] = torch.rand(2)*(self.L / 4) #initial area is the lower left quadrant with size L/4.
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
        
        #update counters of visited targets (for the given agent)
        visited_targets = np.argwhere(self.time_counters[agent_index] != 0)
        self.time_counters[agent_index][visited_targets] = (self.time_counters[agent_index][visited_targets] + 1)%self.tau
        
        if change_direction:
            self.current_directions[agent_index] = torch.rand(1)*2*np.pi
            
        self.positions[agent_index][0] = (self.positions[agent_index][0] + self.agent_step*np.cos(self.current_directions[agent_index]))%self.L
        self.positions[agent_index][1] = (self.positions[agent_index][1] + self.agent_step*np.sin(self.current_directions[agent_index]))%self.L
        
        # self.check_bc()
        
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
        encounters = self.get_encounters(agent_index)
        self.last_step_rewards[agent_index] = self.current_rewards[agent_index].clone()
        
        if sum(encounters) > 0:   
            first_encounter = np.arange(self.Nt)[encounters]
            
            #----KICK----
            # If there was encounter, we reset direction and change position of particle to (pos target + ls)
            # self.current_directions[agent_index] = np.random.uniform(low = 0, high = 2*np.pi)
            # self.positions[agent_index][0] = (self.target_positions[first_encounter, 0] + self.ls*np.cos(self.current_directions[agent_index]))%self.L
            # self.positions[agent_index][1] = (self.target_positions[first_encounter, 1] + self.ls*np.sin(self.current_directions[agent_index]))%self.L
            #------------
            
            #----TIME DELAY----
            #If the target was visited in the previous tau steps, it does not get a reward:
            if self.time_counters[agent_index, first_encounter] != 0:
                return 0
            else:
                #Else, the time counter for the given target and agent starts.
                self.time_counters[agent_index, first_encounter] = 1
            
            #...and we add the information that this agent got to the target
            self.current_rewards[agent_index] = 1
            
            # self.check_bc()
            
            return 1
        
        else: 
            self.current_rewards[agent_index] = 0
            return 0

    def get_state(self, focal_agent, visual_cone, visual_radius):
        """
        Gets the visual information of the agents surrounding the focal agent.

        Parameters
        ----------
        focal_agent : (int) index of focal agent.
        visual_cone : (float) angle (rad) of the visual cone in front of the agent.
        visual_radius : (int/float) radius of the visual circular region around the agent. 
            
        Returns
        -------
        State: (list) [density of rewarded agents in front, same at the back].
                density values: 0 -- no rewarded agent, 1 -- low # of rewarded agents, 2 -- more than high_den rewarded agents.

        """
        mask_in_sight, mask_behind = self.get_agents_in_sight(focal_agent, visual_cone, visual_radius)
        
        num_success_infront = torch.sum(mask_in_sight * self.last_step_rewards)
        num_success_behind = torch.sum(mask_behind * self.last_step_rewards)
        
#        return [int(np.sign(num_success_infront)), int(np.sign(num_success_behind))] #checks only if there is any rewarded agent (no info about density).
        
        return [np.argwhere((np.array([0, 1, self.high_den, self.num_agents]) - int(num_success_infront)) <= 0)[-1][0],
                np.argwhere((np.array([0, 1, self.high_den, self.num_agents]) - int(num_success_behind)) <= 0)[-1][0]]
        
    
    def get_agents_in_sight(self, focal_agent, visual_cone, visual_radius):
        """
        Get which agents are within the front visual cone of the focal agent.

        Parameters
        ----------
        focal_agent : (int) index of focal agent.
        visual_cone : (float) angle (rad) of the visual cone in front of the agent.
        visual_radius : (int/float) radius of the visual circular region around the agent. 
            

        Returns
        -------
        mask_in_sight : (torch.tensor of boolean values)
            True at the indices of agents that are within the visual cone in front.
        mask_behind : (torch.tensor of boolean values)
            True at the indices of agents that are within the visual range, but outside the front cone.

        """
        
        y = self.coord_mod(self.previous_pos[:,1], self.positions[focal_agent,1], self.L)
        x = self.coord_mod(self.previous_pos[:,0], self.positions[focal_agent,0], self.L)
        
        mask_inside_radius = np.sqrt(x**2 + y**2) < visual_radius
        
        mask_in_sight = (np.abs((np.arctan2(y,x) + 2*np.pi) % (2*np.pi) - self.current_directions[focal_agent]) < visual_cone / 2 ) * mask_inside_radius
        mask_behind = mask_inside_radius ^ mask_in_sight
        
        mask_in_sight[focal_agent] = False
        mask_behind[focal_agent] = False
        
        return mask_in_sight, mask_behind
    
    def get_encounters(self, agent_index):
        """
        Considering the agent flies, it gets the targets the agent encountered when it lands.
        (It does not take into account the targets it crossed while making the step).
        
        Parameters
        ----------
        agent_index : (int)

        Returns
        -------
        Mask with the target it encountered (if any).
        If there are more than one, it takes the closest to its position.

        """
        y = self.coord_mod(self.target_positions[:,1], self.positions[agent_index,1], self.L)
        x = self.coord_mod(self.target_positions[:,0], self.positions[agent_index,0], self.L)
        
        distance = np.sqrt(x**2 + y**2)
        mask_encounters = distance < self.at
        
        #if there is more than one encounter, pick the one nearest the agent's position.
        if torch.sum(mask_encounters) > 1:
            one_encounter_mask = torch.zeros(self.Nt, dtype=torch.bool)
            one_encounter_mask[np.argmin(distance)] = True
            return one_encounter_mask
        else:
            return mask_encounters
        
    #----------helper methods--------------# 
    
    def check_bc(self, agent_index=0): 
        if self.boundary_condition == 'reflectant':
            while torch.max(self.positions[agent_index]) > self.L or torch.min(self.positions[agent_index])< 0: 
                self.positions[agent_index][self.positions[agent_index] > self.L] = self.positions[agent_index][self.positions[agent_index] > self.L] - 2*(self.positions[agent_index][self.positions[agent_index] > self.L] - self.L)
                self.positions[agent_index][self.positions[agent_index] < 0] = - self.positions[agent_index][self.positions[agent_index] < 0]


        elif self.boundary_condition == 'periodic':
            while torch.max(self.positions[agent_index]) > self.L or torch.min(self.positions[agent_index])< 0: 
                self.positions[agent_index][self.positions[agent_index] > self.L] = self.positions[agent_index][self.positions[agent_index] > self.L] - self.L
                self.positions[agent_index][self.positions[agent_index] < 0] = self.L + self.positions[agent_index][self.positions[agent_index] < 0]  

    
    def coord_mod(self, coord1, coord2, mod):
        """
        Computes the distance difference between two coordinates, in a world with size 'mod'
        and periodic boundary conditions.

        Parameters
        ----------
        coord1 : value, np.array, torch.tensor (can be shape=(n,1))
            First coordinate.
        coord2 : np.array, torch.tensor -- shape=(1,1)
            Second coordinate, substracted from coord1.
        mod : int
            World size.

        Returns
        -------
        diff_min : float
            Distance difference (with correct sign, not absolute value).

        """
        diff = np.remainder(coord1 - coord2, mod)
        diff_min = np.minimum(diff, mod-diff)
        
        diff_min[diff_min != diff] = -diff_min[diff_min != diff]
        
        return diff_min
        