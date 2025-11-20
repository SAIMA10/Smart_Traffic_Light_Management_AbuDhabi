import json, numpy as np, cityflow


class CityFlowTLS:
    def __init__(self, config_path, step_seconds=5, yellow_time=0):
        self.config_path = config_path
        self.step_seconds = step_seconds
        self.yellow_time = yellow_time
        self.eng = None
        self.intersections = []
        self.phase_lists = {}
        self.action_sizes = {}

    def reset(self):
        self.eng = cityflow.Engine(self.config_path, thread_num=1)
        self.intersections = [
            iid
            for iid in self.eng.get_intersection_ids()
            if self.eng.is_intersection_has_signal(iid)
        ]
        for iid in self.intersections:
            self.phase_lists[iid] = self.eng.get_tl_phase_list(iid)
            self.action_sizes[iid] = len(self.phase_lists[iid])
        return self._obs()

    def _obs(self):
        # Example state: lane vehicle counts concatenated per signalized intersection
        obs = []
        for iid in self.intersections:
            lanes = self.eng.get_lane_ids_by_intersection(iid)
            counts = [self.eng.get_lane_vehicle_count(l) for l in lanes]
            obs.extend(counts + [self.eng.get_tl_phase(iid)])
        return np.array(obs, dtype=np.float32)

    def _reward(self):
        # Example reward: negative total waiting time (lower waiting is better)
        return -float(self.eng.get_total_waiting_time())

    def step(self, action_dict):
        # action_dict: {intersection_id: phase_index}
        for iid, a in action_dict.items():
            a = int(a) % self.action_sizes[iid]
            self.eng.set_tl_phase(iid, a)
        # Advance the sim a few seconds per decision
        for _ in range(self.step_seconds):
            self.eng.next_step()
        obs = self._obs()
        rew = self._reward()
        done = False  # episodic termination can be time-based outside
        info = {}
        return obs, rew, done, info

    def action_space_sizes(self):
        return {iid: self.action_sizes[iid] for iid in self.intersections}
