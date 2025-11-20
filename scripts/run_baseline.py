# scripts/run_baseline.py
import time

# from env_cityflow import CityFlowTLS
from cityflow_env import CityFlowEnv

EPOCH_STEPS = 2000  # total sim seconds
DECISION_EVERY = 5  # must match env.step_seconds

env = CityFlowEnv("config/config.json", step_seconds=DECISION_EVERY)
# obs = env.reset()
sizes = env.action_space_sizes()


# Naive baseline: for each intersection, keep current phase unless queue grows, then cycle
def pick_actions(env):
    actions = {}
    for iid in env.intersections:
        cur = env.eng.get_tl_phase(iid)
        lanes = env.eng.get_lane_ids_by_intersection(iid)
        q = sum(env.eng.get_lane_waiting_vehicle_count(l) for l in lanes)
        if q > 10:
            actions[iid] = (cur + 1) % env.action_sizes[iid]
        else:
            actions[iid] = cur
    return actions


total_reward = 0.0
for t in range(0, EPOCH_STEPS, DECISION_EVERY):
    a = pick_actions(env)
    obs, r, done, info = env.step(a)
    total_reward += r

print("Total reward (higher is better):", total_reward)
