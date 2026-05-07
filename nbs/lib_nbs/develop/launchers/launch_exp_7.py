from rl_opts.rl_framework.numba.agents import Foragers_efficient, run_collective
from rl_opts.rl_framework.numba.environments import CollectiveEnv
import numba
import os
import numpy as np

num_agents = 50
max_counter = 50
runs = 5

Nt = 100; 
# We divide by 2 to allow better visibility between agents. The density stays the same
L = 50; r = 0.5
agent_step = 1
# Time delays
taus = [2, 3, 4]; 
taus_reward = [3, 4, 5]
# Visual cone
visual_activated = True
visual_ranges = [2.0, 3.0, 4.0]
visual_angle=np.pi / 2 # Note that this is from the center, so from +45 to -45

# Training parameters
gamma_damping = 0.00001
eta_glow_damping = 0.1
# State space: [counter, any_agent_in_cone (0/1), rewarded_agent_in_cone (0/1)]
state_space = np.array([max_counter, 2, 2])
# agents = Foragers_efficient(num_agents, 2, state_space)

time_ep = 5000
episodes = 2000

skip_cases = {
            (2, 3, 2.0, True),
            (2, 3, 2.0, False),
            (2, 3, 3.0, True),
            (2, 3, 3.0, False),
            (2, 3, 4.0, True),
            (2, 3, 4.0, False),
            (2, 4, 2.0, True),
            (2, 4, 2.0, False),
            (2, 4, 3.0, True),
            (2, 4, 3.0, False),
            (2, 4, 4.0, True),
            (2, 4, 4.0, False),
        }


out_dir = "/media/gorka/DATA/rl_opts_data/results_learning/collective/exp_7"
os.makedirs(out_dir, exist_ok=True)
for tau in taus:
    for tau_reward in taus_reward:
        for visual_range in visual_ranges:
            for shared_depletion in [True, False]:

                if tau == taus[0] and tau_reward == taus_reward[0] and visual_range == visual_ranges[0]:
                    continue  # already ran this case in EXP 6

                case = (tau, tau_reward, visual_range, shared_depletion)
                if case not in skip_cases:
                    print(f"Skipped case: tau={tau}, tau_reward={tau_reward}, visual_range={visual_range}, shared_depletion={shared_depletion}")
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
                
                file_name = f"_shared_{shared_depletion}_tau_{tau}_tauR_{tau_reward}_vr_{visual_range}.npy"
                rewards_path = os.path.join(out_dir, 
                                            "rewards"+file_name)
                h_matrix_path = os.path.join(out_dir, 
                                             f"h_matrix"+file_name)

                np.save(rewards_path, rews)
                np.save(h_matrix_path, mats)