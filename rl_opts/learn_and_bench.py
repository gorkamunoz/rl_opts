# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/lib_nbs/02_learning_and_benchmark.ipynb.

# %% auto 0
__all__ = ['learning', 'walk_from_policy', 'agent_efficiency', 'average_search_efficiency']

# %% ../nbs/lib_nbs/02_learning_and_benchmark.ipynb 3
import numpy as np
import pathlib

from .rl_framework import TargetEnv, Forager
from .utils import get_encounters

# %% ../nbs/lib_nbs/02_learning_and_benchmark.ipynb 5
def learning(config, results_path, run):
    """
    Training of the RL agent
    
    Parameters
    ----------
    config : dict
        Dictionary with all the parameters
    results_path : str
        Path to save the results
    run : int
        Agent identifier
    """
    
    #Simulation parameters
    TIME_EP = config['MAX_STEP_L'] #time steps per episode
    EPISODES = config['NUM_EPISODES'] #number of episodes
    
    #initialize environment
    env = TargetEnv(Nt=config['NUM_TARGETS'], L=config['WORLD_SIZE'], r=config['r'], lc=config['lc'])
    
    #initialize agent 
    STATE_SPACE = [np.linspace(0, config['MAX_STEP_L']-1, config['NUM_BINS']), np.arange(1), np.arange(1)]
    NUM_STATES = np.prod([len(i) for i in STATE_SPACE])
    
    #default initialization policy
    if config['PI_INIT'] == 0.5:
        INITIAL_DISTR = None
    #change initialization policy
    elif config['PI_INIT'] == 0.99:
        INITIAL_DISTR = []
        for percept in range(NUM_STATES):
            INITIAL_DISTR.append([0.99, 0.01])
            
    agent = Forager(num_actions=config['NUM_ACTIONS'],
                    state_space=STATE_SPACE,
                    gamma_damping=config['GAMMA'],
                    eta_glow_damping=config['ETA_GLOW'],
                    initial_prob_distr=INITIAL_DISTR)
    
    for e in range(EPISODES):
        
        #initialize environment and agent's counter and g matrix
        env.init_env()
        agent.agent_state = 0
        agent.reset_g()
    
        for t in range(TIME_EP):
            
            #step to set counter to its min value n=1
            if t == 0 or env.kicked[0]:
                #do one step with random direction (no learning in this step)
                env.update_pos(1)
                #check boundary conditions
                env.check_bc()
                #reset counter
                agent.agent_state = 0
                #set kicked value to false again
                env.kicked[0] = 0
                
            else:
                #get perception
                state = agent.get_state()
                #decide
                action = agent.deliberate(state)
                #act (update counter)
                agent.act(action)
                
                #update positions
                env.update_pos(action)
                #check if target was found + kick if it is
                reward = env.check_encounter()
                    
                #check boundary conditions
                env.check_bc()
                #learn
                agent.learn(reward)
                
                
        if (e+1)%500 == 0:
            #save h matrix of the agent at this stage of the learning process
            np.save(results_path+'memory_agent_'+str(run)+'_episode_'+str(e+1)+'.npy', agent.h_matrix)
                

# %% ../nbs/lib_nbs/02_learning_and_benchmark.ipynb 7
def walk_from_policy(policy, time_ep, n, L, Nt, r, lc, destructive=False, with_bound=False, bound=100):
    """
    Walk of foragers given a policy. Performance is evaluated as the number of targets found in a fixed time time_ep.
    
    Parameters
    ----------
    policy : list
        Starting from counter=1, prob of continuing for each counter value.
    time_ep : int
        Number of steps (decisions).
    n : int
        Number of agents that walk in parallel (all with the same policy, they do not interact). This is "number of walks" in the paper.
    L : int
        World size.
    Nt : int
        Number of targets.
    r : float
        Target radius.
    lc : float
        Cutoff length. Agent is displaced a distance lc from the target when it finds it.
    destructive : bool, optional
        True if targets are destructive. The default is False.
    with_bound : bool, optional
        True if policy is cut. The default is False.
    bound : int, optional
        Bound of the policy (maximum value for the counter). The default is 20.

    Returns
    -------
    reward : list, len(rewards)=n
        Number of targets found by each agent in time_ep steps of d=1.

    """
    
    #initialize agents clocks, positions and directions, as well as targets in the env.
    pos = np.zeros((time_ep, n, 2)) 
    pos[0] = np.random.rand(n,2)*L
    
    current_pos = np.random.rand(n,2)*L
    
    direction = np.random.rand(n)*2*np.pi 
    internal_counter = [0]*n
    target_positions = np.random.rand(Nt,2) * L
    reward = [0]*n
    
    #cut policy
    if with_bound:
        policy[bound:] = [0] * (len(policy)-bound)
        
    for t in range(1, time_ep):   
        
        #update position
        previous_pos = np.copy(current_pos)
        current_pos[:,0] = previous_pos[:, 0] + np.cos(direction)
        current_pos[:,1] = previous_pos[:, 1] + np.sin(direction)
        
        #check reward
        encounters = get_encounters(previous_pos, current_pos, target_positions, L, r)
        
        for ag, num_encounters in enumerate(np.sum(encounters,axis=0)):
            kick = False
            
            if num_encounters > 0: 
                
                first_encounter = np.arange(len(target_positions))[encounters[:,ag]]
                
                if destructive:
                    #target is destroyed, sample position for a new target.
                    target_positions[first_encounter] = np.random.rand(2) * L
                else:
                    #----KICK----
                    # If there was encounter, we reset direction and change position of particle to (pos target + lc)
                    kick_direction = np.random.rand()*2*np.pi  
                    
                    current_pos[ag, 0] = target_positions[first_encounter, 0] + lc*np.cos(kick_direction)
                    current_pos[ag, 1] = target_positions[first_encounter, 1] + lc*np.sin(kick_direction)
                    
                    #------------
                internal_counter[ag] = 0
                reward[ag] += 1
                kick = True
                
            
            current_pos[ag] %= L
            
            if np.random.rand() > policy[internal_counter[ag]] or kick:
                internal_counter[ag] = 0
                direction[ag] = np.random.rand()*2*np.pi  
                
            else:
                internal_counter[ag] += 1
                
    return reward

# %% ../nbs/lib_nbs/02_learning_and_benchmark.ipynb 9
from .utils import get_config, get_policy

def agent_efficiency(results_path, config, run, num_walks, episode_interval):
    """
    Computes the agent's average search efficiency over a number of walks where the agent follows a fixed policy. 
    This is repeated with the policies at different stages of the training to analyze the evolution of its performance.
    
    Parameters
    ----------
    results_path : str
        Path to the results folder, from which to extract the agent's policies
    config : dict
        Dictionary with all the parameters. It needs to be the same configuration file as the one used to train the agent.
    run : int
        Id of the agent
    num_walks : int
        Number of (independent) walks
    episode_interval : int
        Every 'episode_interval' training episodes, the policy of the agent is taken and its performance is analyzed.
        
    """

    print('Statistics postlearning of agent', run, '\nData obtained from folder: ', results_path)
    
    
    for training_episode in [i for i in range(0, config['NUM_EPISODES'] + 1, episode_interval)]:
        
        if training_episode == 0 and config['PI_INIT'] == 0.99:
            frozen_policy = [0.99 for percept in range(config['MAX_STEP_L'])] #initial policy
            
        elif training_episode == 0 and config['PI_INIT'] == 0.5:
            frozen_policy  = [0.5 for percept in range(config['MAX_STEP_L'])] #initial policy
            
        else:
            #get policy from the stored h matrix at the given training_episode
            frozen_policy = get_policy(results_path, run, training_episode)
            
        #run the 10^4 walks (in parallel) with the same policy
        rewards = walk_from_policy(policy=frozen_policy,
                                   time_ep=config['MAX_STEP_L'],
                                   n=num_walks,
                                   L=config['WORLD_SIZE'],
                                   Nt=config['NUM_TARGETS'],
                                   r=config['r'],
                                   lc=config['lc'])
        
        #save results
        np.save(results_path+'performance_post_training_agent_'+str(run)+'_episode_'+str(training_episode)+'.npy', rewards)
        
        

# %% ../nbs/lib_nbs/02_learning_and_benchmark.ipynb 12
import pathlib
import warnings


from .analytics import get_policy_from_dist, pdf_powerlaw, pdf_multimode

# %% ../nbs/lib_nbs/02_learning_and_benchmark.ipynb 13
def average_search_efficiency(config):
    """
    Get the average search efficiency, considering the benchmark model defined in config.

    Parameters
    ----------
    config : dict
        Dictionary with the configuration of the benchmark model.
    """
    
    try:
        from ray import tune
        from ray.tune.search.bayesopt import BayesOptSearch
        from ray.tune.search import ConcurrencyLimiter
    except:
        warnings.warn('ray[tune] is currently not installed. If you intend to use the'
                      '`average_search_efficiency` function, consider installing via pip install ray[tune].', stacklevel=2)
    
    #get parameters of the distributions depending on the chosen model
    if config['model'] == 'powerlaw':
        parameters = [config['beta']]
        #get policy from benchmark model
        policy = get_policy_from_dist(n_max = config['time_ep'], 
                                      func = pdf_powerlaw,
                                      beta = config['beta']
                                     )
    
    elif config['model'] == 'double_exp':
        parameters = [config['d_int'], config['d_ext'], config['p']]
        #get policy from benchmark model
        policy = get_policy_from_dist(n_max=config['time_ep'],
                                      func = pdf_multimode,
                                      lambdas = np.array(parameters[:2]),
                                      probs = np.array([parameters[2], 1-parameters[2]])
                                  )
    
    
    #run the walks in parallel
    efficiencies = walk_from_policy(policy=policy,
                                    time_ep=config['time_ep'],
                                    n=config['n'],
                                    L=config['L'],
                                    Nt=config['Nt'],
                                    r=config['r'],
                                    lc=config['lc'])
    
    #get the mean search efficiency over the walks
    mean_eff = np.mean(efficiencies) 
    tune.report(mean_eff = mean_eff)
    
    #save results
    if config['results_path']:
        np.save(config['results_path']+'efficiencies_'+ str([np.round(p, 10) for p in parameters])+'.npy', efficiencies)
     
