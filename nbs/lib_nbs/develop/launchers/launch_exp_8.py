from rl_opts.rl_framework.numba.agents import Foragers_efficient, run_collective
from rl_opts.rl_framework.numba.environments import CollectiveEnv
import numba
import os, socket
import numpy as np

num_agents = 50
max_counter = 50
runs = 5

Nt = 100; 
# We divide by 2 to allow better visibility between agents. The density stays the same
L = 50; r = 0.5
agent_step = 1
# Time delays
taus = np.arange(2, 12)

visual_activated = False

# Training parameters
gamma_damping = 0.00001
eta_glow_damping = 0.1
# State space: [counter, any_agent_in_cone (0/1), rewarded_agent_in_cone (0/1)]
state_space = np.array([max_counter, 2, 2])
# agents = Foragers_efficient(num_agents, 2, state_space)

time_ep = 5000
episodes = 10000

# Saving data
if os.getlogin() == "gorka":
    if socket.gethostname() == "gpu-qic":
        out_dir = "/media/gorka/DATA/rl_opts_data/results_learning/collective/"
    elif socket.gethostname() == "gpu-ada-qic":
        out_dir = "/sata1/gorka/collective/"
elif os.getlogin() == "c7051165":
    out_dir = "/scratch/c7051165/github/rl_opts/nbs/lib_nbs/develop/results/"

filename = os.path.basename(__file__)  
exp = filename.replace("launch_exp_", "").replace(".py", "")
out_dir = os.path.join(out_dir, f"exp_{exp}")
os.makedirs(out_dir, exist_ok=True)
print(out_dir)

test_save = os.path.join(out_dir, 'test_save.npy')
np.save(test_save, np.arange(10))

# Wait three hours
import time
# Create progress bar for waiting
for i in range(3 * 3600):
    if i % 60 == 0:
        print(f"Waiting... {i//60} minutes passed")
    time.sleep(1)
print("Done waiting, starting experiments")

shared_depletion = False
for tau in taus:
                
    file_name = f"_shared_{shared_depletion}_tau_{tau}.npy"
    rewards_path = os.path.join(out_dir, 
                                "rewards"+file_name)
    h_matrix_path = os.path.join(out_dir, 
                                 "h_matrix"+file_name)

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
                                num_agents=num_agents,
                                agent_step=agent_step,
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
    
    print(f"Finished case: tau={tau}, shared_depletion={shared_depletion}")
