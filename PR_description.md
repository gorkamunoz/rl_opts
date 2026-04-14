# Pull Request: LabVIEW-wrapper extensions and bug fixes

## Overview

This fork extends the `labview_wrapper` branch with changes motivated by
coupling the RL-opts framework to real experimental data from a LabVIEW-controlled
active-matter setup.  Two files are modified:

- `rl_opts/rl_framework/numba/environments.py`
- `rl_opts/virtual_abm.py`

All changes are marked with `# [CHANGE]` or `# [NEW METHOD]` comments in the
source so they are easy to locate.

---

## `rl_opts/rl_framework/numba/environments.py`

### 1 — Bug fix: `TargetEnv.check_encounter()` — reward not returned for destructive targets

**Problem.**  In the original code the lines

```python
self.current_rewards[agent_index] = 1
return 1
```

were indented inside the `else` branch of `if self.destructive_targets`, so
a successful encounter in destructive mode silently returned `None` (fell
through to the `else: return 0` path at the end of the method).

**Fix.**  `self.current_rewards[agent_index] = 1` and `return 1` are moved one
indentation level out — they now execute for **both** destructive and
non-destructive encounters.  Additionally, `self.kicked[agent_index] = 1` is
moved out of the inner `for idx_first in first_encounter` loop so it is set
once per encounter event (no functional difference for a single target, but
correct semantics in general).

```python
# BEFORE (simplified)
if self.destructive_targets:
    self.target_positions[first_encounter] = np.random.rand(2)*self.L
else:
    ...kick logic...
    self.kicked[agent_index] = 1      # inside for-loop
    self.current_rewards[agent_index] = 1   # inside else — never reached for destructive!
    return 1                                # inside else

# AFTER
if self.destructive_targets:
    self.target_positions[first_encounter] = np.random.rand(2)*self.L
else:
    ...kick logic...
self.kicked[agent_index] = 1          # outside for-loop, inside else
# outside if/else — reached for both destructive and non-destructive:
self.current_rewards[agent_index] = 1
return 1
```

### 2 — New method: `TargetEnv.check_inside_target()`

`check_encounter()` detects encounters by checking whether the agent's
trajectory segment crosses the target boundary arc (`isBetween_c_Vec_numba`).
When the agent displacement per step is small relative to the target radius `r`
the agent can be physically inside a target without having crossed the arc during
the current step (e.g. after a kick that lands inside another target, or at
initialisation).  `check_inside_target()` provides a complementary
point-in-disc test.

```python
def check_inside_target(self, agent_index=0):
    diff = self.positions[agent_index] - self.target_positions  # (Nt, 2)
    if self.Nt > 1:
        dists = np.sqrt(diff[:, 0]**2 + diff[:, 1]**2)  # Numba-safe row norm
        return (dists <= self.r).any()
    else:
        return np.sqrt(diff[0, 0]**2 + diff[0, 1]**2) <= self.r
```

**Numba note.**  `np.linalg.norm(..., axis=1)` is not supported inside
`@jitclass` methods in Numba, so the Euclidean distance is computed manually
via component-wise squares.

---

## `rl_opts/virtual_abm.py`

### 3 — `virtual_ABM.__init__()`: early initialisation of `self.done`

`self.done = False` is now set in `__init__` before `init_training()` is
called.  This ensures the attribute exists immediately after construction even
if a subclass or external code inspects it before the first `step()`.

### 4 — `virtual_ABM.init_epoch()`: reset `done` flag on new epoch

```python
# [CHANGE] added
if self.done:
    self.done = False
```

When the caller detects `done=True` and calls `init_epoch()` to start a new
episode, the flag is now cleared automatically.  The original required the
caller to reset it manually.

### 5 — `virtual_ABM.step()`: dual target-detection (main extension)

The passive-phase reward signal now combines both crossing and inside-disc
detection:

```python
# BEFORE
if self.current_phase == 0:
    reward = self.env.check_encounter()

# AFTER
if self.current_phase == 0:
    crossed = self.env.check_encounter()
    inside  = self.env.check_inside_target()
    reward  = 1 if (crossed or inside) else 0
```

This is the primary motivation for adding `check_inside_target()` above.  In
LabVIEW-driven experiments the displacement per frame can be very small
(sub-pixel after calibration), making pure arc-crossing detection unreliable.

### 6 — `virtual_ABM.step()`: optional learning (`learn=True` parameter)

```python
# BEFORE
def step(self, disp, return_reward=False):

# AFTER
def step(self, disp, return_reward=False, learn=True):
```

Passing `learn=False` runs the environment forward without updating the PS
matrices.  Useful for evaluation / replay loops without touching the learned
policy.  Default `True` is fully backward-compatible.

### 7 — `virtual_ABM.step()`: episode-end condition

In the original, `done` was set to `True` both on time-limit and on any
non-zero reward:

```python
# ORIGINAL
if self.t_ep == self.time_ep:
    self.done = True
if reward != 0:
    self.done = True   # ← removed
```

The reward-based termination is removed.  `done` is now set only when the
episode time-limit is reached.  The caller can still detect individual target
hits from the returned `reward` value.  This allows training episodes that
contain multiple target encounters (relevant for multi-target or high-density
experiments).

### 8 — `get_ABM_motion()`: `bc_periodic` default changed to `True`

```python
# BEFORE
def get_ABM_motion(..., bc_periodic=None):
    if bc_periodic is not None: x = x % L; y = y % L

# AFTER
def get_ABM_motion(..., bc_periodic=True):
    if bc_periodic is True: x = x % L; y = y % L
```

The default is changed from `None` (periodic BCs off) to `True` (periodic BCs
on) to match the typical experimental setup (particles in a periodic square
arena).  Pass `bc_periodic=False` explicitly to disable.

---

## Additional files

`nbs/tutorials/` contains several new notebooks that use the above extensions
to interface with LabVIEW-acquired data and run the PS agent on experimental
trajectories:

| Notebook | Description |
|---|---|
| `RL_opts_LV_functions.ipynb` | Core LabVIEW data-loading and pre-processing utilities |
| `RL_opts_LV_functions_cell.ipynb` | Single-cell geometry experiments |
| `RL_opts_LV_functions_grid.ipynb` | Grid-geometry experiments |
| `RL_opts_LV_functions_cluster_grid.ipynb` | Cluster analysis on grid experiments |
| `RL_opts_LV_functions_cluster_nsp_analysis.ipynb` | Non-self-propelled (passive) fraction analysis |
| `test_target_gen_function.ipynb` | Unit tests for the target-generation helpers |
| `unpack_pkl.ipynb` | Utility to inspect and export saved `.pkl` result files |
