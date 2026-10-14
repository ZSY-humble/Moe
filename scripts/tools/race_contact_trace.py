"""用 GPU 接触张量记录实际接触对与接触点，只用于独立诊断。"""

from collections import Counter, defaultdict, deque
import json

import numpy as np
from scipy.spatial.transform import Rotation


class RaceContactTrace:
    """按原三帧终止窗口保留实际接触；由接触点定位机身分段。"""

    def __init__(self, env, collision_json):
        from pxr import UsdPhysics
        import omni.usd

        self.env = env
        assert env.cfg.scene.robot.spawn.filter_base_hip_housings
        self.robot = env.scene['robot']
        self.body_names = self.robot.body_names
        self.base_id = self.body_names.index('base_link')
        self.history = env.scene.sensors['contact_forces'].cfg.history_length
        self.pending = [deque() for _ in range(env.num_envs)]
        self.episodes = []
        self.dominant = Counter()
        self.any_pair = Counter()
        self.unmatched = 0
        self.point_count = 0
        self.max_point_count_per_step = 0
        self.ambiguous_segment_points = 0
        spec = json.loads(collision_json.read_text())
        self.boxes = [g for g in spec['geoms'] if g['body'] == 'base_link']
        assert all(g['type'] == 'box' and g['quat'] == [1., 0., 0., 0.]
                   for g in self.boxes)
        self.centers = np.asarray([g['pos'] for g in self.boxes])
        self.half_sizes = np.asarray([g['size'] for g in self.boxes])
        self.other_shape_names = {}
        for body in self.body_names:
            names = [g['name'] for g in spec['geoms'] if g['body'] == body
                     and not g['name'].endswith(('_mjx_hip', '_hip_housing'))]
            self.other_shape_names[body] = names
        self.filters = [n for n in self.body_names if n != 'base_link']
        patterns = ['/World/envs/env_*/Robot/'+n for n in self.filters]
        stage = omni.usd.get_context().get_stage()
        for prim in stage.Traverse():
            path = str(prim.GetPath())
            if path.startswith('/World/ground/') and prim.HasAPI(
                    UsdPhysics.CollisionAPI):
                self.filters.append(path)
                patterns.append(path)
        sensor = env.scene.sensors['base_pair_contacts']
        self.capacity = 16384
        self.view = sensor._physics_sim_view.create_rigid_contact_view(
            '/World/envs/env_*/Robot/base_link', patterns,
            max_contact_data_count=self.capacity)
        assert self.view.sensor_count == env.num_envs
        assert self.view.filter_count == len(self.filters)
        self.original_update = env.scene.update

        def traced_update(dt):
            result = self.original_update(dt)
            self.record_contacts()
            return result

        env.scene.update = traced_update

    def record_contacts(self):
        """每个物理步在状态更新后读取接触，不改变原传感器历史。"""
        arrays = self.view.get_contact_data(dt=self.env.physics_dt)
        forces, points, normals, separations, counts, starts = [
            value.cpu().numpy() for value in arrays]
        counts = counts.reshape(self.env.num_envs, len(self.filters))
        starts = starts.reshape(counts.shape)
        total = int(counts.sum())
        assert total < self.capacity, '接触点缓冲已满，本次归因无效。'
        self.max_point_count_per_step = max(self.max_point_count_per_step, total)
        self.point_count += total
        if total == 0:
            return
        positions = self.robot.data.body_pos_w[:, self.base_id].cpu().numpy()
        quats = self.robot.data.body_quat_w[:, self.base_id].cpu().numpy()
        rotations = Rotation.from_quat(quats[:, [1, 2, 3, 0]]).as_matrix()
        step = int(self.env._sim_step_counter)
        for env_id, filter_id in np.argwhere(counts > 0):
            offset = int(starts[env_id, filter_id])
            count = int(counts[env_id, filter_id])
            groups = defaultdict(list)
            for i in range(offset, offset+count):
                local = (points[i]-positions[env_id]) @ rotations[env_id]
                delta = np.abs(local-self.centers)-self.half_sizes
                signed = (np.linalg.norm(np.maximum(delta, 0), axis=1)
                          + np.minimum(delta.max(axis=1), 0))
                surface_distances = np.abs(signed)
                box_id = int(surface_distances.argmin())
                candidates = np.flatnonzero(
                    surface_distances <= surface_distances[box_id]+.001)
                self.ambiguous_segment_points += len(candidates) > 1
                force = float(forces[i, 0])
                groups[box_id].append({
                    'position_w_m':points[i].tolist(),
                    'position_base_frame_m':local.tolist(),
                    'normal_w':normals[i].tolist(),
                    'separation_m':float(separations[i, 0]),
                    'normal_force_n':force,
                    'nearest_base_box_surface_distance_m':float(
                        surface_distances[box_id]),
                    'base_box_candidates_within_1mm':[
                        self.boxes[j]['name'] for j in candidates],
                })
            other_body = self.filters[filter_id]
            for box_id, contact_points in groups.items():
                vector = np.sum([p['normal_force_n']*np.asarray(p['normal_w'])
                                 for p in contact_points], axis=0)
                force = float(np.linalg.norm(vector))
                if force < 1e-6:
                    continue
                records = self.pending[env_id]
                names = self.other_shape_names.get(other_body, [])
                records.append({
                    'physics_step':step,
                    'base_box_from_contact_point':self.boxes[box_id]['name'],
                    'other_body':other_body,
                    'other_shape_candidates':names,
                    'normal_contact_force_n':force,
                    'contact_points':contact_points,
                })
                while records and records[0]['physics_step'] < step-self.history+1:
                    records.popleft()

    @staticmethod
    def pair_label(record):
        names = record['other_shape_candidates']
        other = names[0] if len(names) == 1 else record['other_body']
        return record['base_box_from_contact_point']+' <-> '+other

    def capture_reset(self, env_ids, heights, upright, base_forces):
        """保存复位前最近三物理步内的接触，保持原终止逻辑。"""
        earliest = int(self.env._sim_step_counter)-self.history+1
        for index, env_id in enumerate(env_ids.cpu().tolist()):
            contacts = [r for r in self.pending[env_id]
                        if r['physics_step'] >= earliest]
            significant = [r for r in contacts
                           if r['normal_contact_force_n'] > 1.]
            for pair in {self.pair_label(r) for r in significant}:
                self.any_pair[pair] += 1
            dominant = max(significant,
                           key=lambda r:r['normal_contact_force_n'], default=None)
            if dominant:
                self.dominant[self.pair_label(dominant)] += 1
            else:
                self.unmatched += 1
            self.episodes.append({
                'env_id':env_id,
                'episode_length_steps':int(self.env.episode_length_buf[env_id]),
                'height_m':float(heights[index]),
                'upright':float(upright[index]),
                'base_sensor_history_peak_n':float(base_forces[index]),
                'dominant_pair':self.pair_label(dominant) if dominant else None,
                'contacts_in_termination_window':contacts,
            })
            self.pending[env_id].clear()

    def report(self):
        return {
            'source':'GPU刚体接触张量；实际刚体对、接触点和法向力。',
            'point_force_sign':'normal_force_n为SDK有符号法向力；接触组力取向量和的模。',
            'segment_assignment':'用求解后基座姿态将实际接触点转入局部坐标，按最近盒面推定分段；不是原始形状ID。',
            'segment_assignment_tolerance_m':.001,
            'ambiguous_segment_points':int(self.ambiguous_segment_points),
            'history_physics_steps':self.history,
            'physics_dt_s':self.env.physics_dt,
            'total_contact_points':self.point_count,
            'max_contact_points_per_step':self.max_point_count_per_step,
            'dominant_pair_episode_counts':dict(self.dominant.most_common()),
            'any_pair_episode_counts':dict(self.any_pair.most_common()),
            'unmatched_episodes':self.unmatched,
            'note':'同回合可有多个接触对，any_pair不能直接相加。dominant取窗口中最大的报告，未必是唯一触发源。',
            'episodes':self.episodes,
        }

    def close(self):
        self.env.scene.update = self.original_update
