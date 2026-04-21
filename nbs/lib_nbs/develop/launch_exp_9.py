from rl_opts.rl_framework.numba.agents import Foragers_efficient, run_collective
from rl_opts.rl_framework.numba.environments import CollectiveEnv
import numba
import os
import numpy as np

num_agents = 50
max_counter = 50
runs = 5

Nt = 100; 
L = 50; r = 0.5
agent_step = 1
# Time delays
taus = np.arange(2, 12)
tau_reward = 2
# Visual cone
visual_activated = True
visual_range = 2.0
visual_angle = np.pi / 2 # Note that this is from the center, so from +45 to -45

# Training parameters
gamma_damping = 0.00001
eta_glow_damping = 0.1
# State space: [counter, any_agent_in_cone (0/1), rewarded_agent_in_cone (0/1)]
state_space = np.array([max_counter, 2, 2])

time_ep = 5000
episodes = 2000

# Saving data
if os.getlogin() == "gorka":
    out_dir = "/media/gorka/DATA/rl_opts_data/results_learning/collective/"
elif os.getlogin() == "c7051165":
    out_dir = "/scratch/c7051165/github/rl_opts/nbs/lib_nbs/develop/results/"

filename = os.path.basename(__file__)  
exp = filename.replace("launch_exp_", "").replace(".py", "")
out_dir = os.path.join(out_dir, f"exp_{exp}")
os.makedirs(out_dir, exist_ok=True)

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
    
        
    
        rews, mats = run_collective(episodes = episodes, time_ep = time_ep, 
                                    runs=runs * numba.get_num_threads(),
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
                                    visual_activated = visual_activated,
                                    # Agent props
                                    num_actions=2,
                                    state_space=state_space,
                                    gamma_damping=gamma_damping,
                                    eta_glow_damping=eta_glow_damping,
                                    )
        
        
    
        np.save(rewards_path, rews)
        np.save(h_matrix_path, mats)