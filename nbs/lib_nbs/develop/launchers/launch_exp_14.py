from rl_opts.rl_framework.numba.agents import run_collective_directions
from rl_opts.rl_framework.numba.environments import CollectiveDirectionsEnv
import numba
import os
import numpy as np
import socket

num_agents = 50
max_counter = 50
parallel_runs = numba.get_num_threads()
num_parallel_runs = 1

Nt = 100
L = 50; r = 0.5
agent_step = 1
# Time delays
taus = np.arange(2, 12)
tau_reward = 2
# Visual cone
visual_activated = True
visual_range = 2.0
visual_angle = np.pi / 2  # from center, so ±45°

# Direction observation settings
max_agents_directions = 2
num_vals_directions = 4

# Training parameters
gamma_damping = 0.00001
eta_glow_damping = 0.1
# state_space is built inside run_collective_directions:
# [max_counter, 3, num_vals_directions+1, ..., num_vals_directions+1]

time_ep = 5000
episodes = 2000

# Saving data
if os.getlogin() == "gorka":
    if socket.gethostname() == "gpu-qic":
        out_dir = "/media/gorka/DATA/rl_opts_data/results_learning/collective/"
    elif socket.gethostname() == "gpu-ada-qic":
        out_dir = "/home/gorka/rl_opts/nbs/lib_nbs/develop/results/"
elif os.getlogin() == "c7051165":
    out_dir = "/scratch/c7051165/github/rl_opts/nbs/lib_nbs/develop/results/"

filename = os.path.basename(__file__)  
exp = filename.replace("launch_exp_", "").replace(".py", "")
out_dir = os.path.join(out_dir, f"exp_{exp}")
os.makedirs(out_dir, exist_ok=True)
# Test save
np.save(os.path.join(out_dir, 'test_save.npy'), np.arange(10))

# import time
# time.sleep(2 * 3600) # Wait three hours

for tau in taus:
    for shared_depletion in [True, False]:

        file_name = f"_shared_{shared_depletion}_tau_{tau}.npy"
        rewards_path = os.path.join(out_dir, 
                                    "rewards"+file_name)
        h_matrix_path = os.path.join(out_dir, 
                                     f"h_matrix"+file_name)


        if os.path.exists(rewards_path) and os.path.exists(h_matrix_path):
            print(f"Skipping existing case: {file_name}")
            continue
    
        
    
        rews, mats = run_collective_directions(
            episodes=episodes, time_ep=time_ep,
            parallel_runs=parallel_runs,
            num_parallel_runs=num_parallel_runs,
            # Environment props
            Nt=Nt,
            L=L,
            r=r,
            tau=tau,
            tau_reward=tau_reward,
            num_agents=num_agents,
            agent_step=agent_step,
            visual_range=visual_range,
            visual_angle=visual_angle,
            shared_depletion=shared_depletion,
            visual_activated=visual_activated,
            # Agent props
            num_actions=3,
            max_counter=max_counter,
            gamma_damping=gamma_damping,
            eta_glow_damping=eta_glow_damping,
            upd_pos_method='LR',
            turn_angle=np.pi / 4,
            max_agents_directions=max_agents_directions,
            num_vals_directions=num_vals_directions,
        )
        
        
    
        np.save(rewards_path, rews)
        np.save(h_matrix_path, mats)