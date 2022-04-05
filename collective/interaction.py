import time
import numpy as np

from env import TargetEnv
from forager import Forager


# Parameters
# -----------------------------------------------------------------------------
EPISODES = 1
TIME_EP = 500


#Environment
N = 30
L = 20 
at = 1
ls = 2
TAU = 3
NUM_AGENTS = 20
                 
print('\nDensity of agents: ', np.round(NUM_AGENTS / L**2,4))
print('\nDensity of targets: ', np.round(N / L**2,2))


#Forager parameters
VISUAL_CONE = np.pi/2
VISUAL_RADIUS = 3
STATE_SPACE = [np.linspace(0, 100, 100), np.arange(3), np.arange(3)]
NUM_ACTIONS = 2

# Learning_parameters
GAMMA = 0.003
ETA_GLOW = 0.02



#initialize record of performance
rewards = np.zeros((NUM_AGENTS, EPISODES, TIME_EP))
pos_agents = np.zeros((NUM_AGENTS, EPISODES, TIME_EP, 2))

#initialize environment
env = TargetEnv(Nt=N, L=L, at=at, ls=ls, tau=TAU, num_agents=NUM_AGENTS)

#initialize agents
agents = []
for ag in range(NUM_AGENTS):
    agents.append(Forager(visual_cone=VISUAL_CONE,
                              visual_radius=VISUAL_RADIUS,
                              state_space=STATE_SPACE,
                              num_actions=NUM_ACTIONS,
                              gamma=GAMMA,
                              eta_glow_damping=ETA_GLOW,
                              ))            

for e in range(EPISODES):
    
    env.init_env()
    for i in range(NUM_AGENTS):
        agents[i].agent_state = 0
        agents[i].reset_g()
    
    
    t1 = time.perf_counter()
    for t in range(TIME_EP):
        
        for i in range(NUM_AGENTS):
            
            # print('\nAgent ',i, 'time ', t, '#steps ', agents[i].agent_state)
            pos_agents[i, e, t, :] = env.positions[i].clone()    
            
            #get perception
            visual_perception = env.get_state(i, agents[i].visual_cone, agents[i].visual_radius)
            state = agents[i].get_state(visual_perception)
            # print(state)
            action = agents[i].deliberate(state)
            
            #act
            agents[i].act(action)
            
            #update positions and internal state and check reward
            env.update_pos(action, agent_index=i)
            reward = env.check_encounter(agent_index=i)
            if reward == 1:
                agents[i].agent_state = 0
            
            agents[i].learn(reward)
                
            # Save info
            rewards[i, e, t] = reward
            
        
        
    t2 = time.perf_counter()
   
    print('Episode:', e,  '#rewarded agents:', np.mean([np.sum(rewards[:,e,t]) for t in range(TIME_EP)]))
    print('\n Time (in min) for 1 episode of '+str(TIME_EP)+' time steps:' , (t2-t1) / 60)    
    
# np.save('results/pos_agents.npy', pos_agents)

#%%
import matplotlib.pyplot as plt
plt.figure(figsize=(5,5))

#environment
plt.scatter(env.target_positions[:, 0], env.target_positions[:, 1], c = 'C2', s = 1)

plt.axhline(L, ls = '--', alpha = 0.3, c = 'k')
plt.axhline(0, ls = '--', alpha = 0.3, c = 'k')
plt.axvline(L, ls = '--', alpha = 0.3, c = 'k')
plt.axvline(0, ls = '--', alpha = 0.3, c = 'k')

#agents
for ag in range(NUM_AGENTS):
    plt.scatter(pos_agents[ag, 0, 6, 0], pos_agents[ag, 0, 6, 1],marker='^', c = 'k', s = 5)

