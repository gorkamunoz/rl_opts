from rl_opts.rl_framework.numba.agents import Foragers_efficient, run_collective
from rl_opts.rl_framework.numba.environments import CollectiveEnv
import numba
import os
import numpy as np

# Change for this run:
nums_agents = np.arange(10, 80, 10)
tau = 3
tau_reward = 3
visual_range = 4

max_counter = 50
runs = 1

Nt = 100; 
L = 50; r = 0.5
agent_step = 1


# Visual cone
visual_activated = True
visual_angle=np.pi / 2 # Note that this is from the center, so from +45 to -45

# Training parameters
gamma_damping = 0.00001
eta_glow_damping = 0.1
# State space: [counter, any_agent_in_cone (0/1), rewarded_agent_in_cone (0/1)]
state_space = np.array([max_counter, 2, 2])


time_ep = 5000
episodes = 2000 # also available: 200, filename is then file_name = f"_shared_{shared_depletion}_num_agents_{num_agents}.npy


out_dir = "results/exp_7_2/"
os.makedirs(out_dir, exist_ok=True)

# test save
np.save(os.path.join(out_dir, "test.npy"), np.array([1,2,3]))

# Original was with shared_depletion = [True, False], but we rerun again only with False, and a name change, to test the fact that agents don't share targets
# when they are in the same position when shared_depletion = False.
shared_depletion = False
for num_agents in nums_agents:
    rews, mats = run_collective(episodes = episodes, time_ep = time_ep, 
                                runs= runs * numba.get_num_threads(),
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
    
    # file_name = f"_shared_{shared_depletion}_num_agents_{num_agents}_ep_{episodes}.npy" # before the change in commit ee39257
    file_name = f"_shared_{shared_depletion}_num_agents_{num_agents}_ep_{episodes}_check_share_depletion.npy"
    rewards_path = os.path.join(out_dir, 
                                "rewards"+file_name)
    h_matrix_path = os.path.join(out_dir, 
                                    f"h_matrix"+file_name)

    np.save(rewards_path, rews)
    np.save(h_matrix_path, mats)