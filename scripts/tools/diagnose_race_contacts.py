"""隔离诊断 Race 的接触与 PD 稳定性，不修改训练默认值。"""

import argparse
from pathlib import Path
import json

import h5py  # noqa: F401
import tensordict  # noqa: F401
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--num_envs', type=int, default=4)
parser.add_argument('--steps', type=int, default=100)
parser.add_argument('--no_self_collisions', action='store_true')
parser.add_argument('--unfiltered_hip_housings', action='store_true')
parser.add_argument('--armature', type=float, nargs=3)
parser.add_argument('--training_resets', action='store_true')
parser.add_argument('--action_std', type=float, default=0.)
parser.add_argument('--joint_delta_bound', type=float)
parser.add_argument('--pair_forces', action='store_true')
parser.add_argument('--shape_contacts', action='store_true')
parser.add_argument('--contact_offset', type=float)
parser.add_argument('--collision_json', type=Path)
parser.add_argument('--verify_collision_json', type=Path)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--action_seed', type=int, default=20261014)
parser.add_argument('--contact_probe_poses', type=Path)
parser.add_argument('--output', type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.collision_json is not None and args.verify_collision_json is not None:
    parser.error('候选替换与默认模型核验请分开运行。')
if args.collision_json is not None and args.contact_offset is not None:
    parser.error('候选几何与接触偏移测试请分开运行，避免覆盖转换器。')
if args.shape_contacts and not (
        args.training_resets and args.pair_forces and args.collision_json
        and not args.unfiltered_hip_housings and not args.contact_probe_poses):
    parser.error('逐点归因需要训练复位、成对力、候选JSON和四对髋壳过滤。')
if args.joint_delta_bound is not None:
    if not args.training_resets or args.joint_delta_bound < 0:
        parser.error('均匀目标角扰动需要 --training_resets，且幅度不能为负。')
launcher = AppLauncher(args)


def controlled_contact_probes(env, cfg, cases_path):
    """用固定姿态分别验证髋壳过滤、大腿碰撞及机身触地。"""
    import torch

    cases = json.loads(cases_path.read_text())['cases']
    assert args.num_envs % len(cases) == 0
    robot = env.scene['robot']
    sensor = env.scene.sensors['contact_forces']
    base_ids, _ = sensor.find_bodies('base_link')
    roots = robot.data.default_root_state.clone()
    roots[:, :3] += env.scene.env_origins
    joints = robot.data.default_joint_pos.clone()
    velocities = torch.zeros_like(joints)
    labels = []
    for i in range(args.num_envs):
        case = cases[i % len(cases)]
        roots[i, 2] = env.scene.env_origins[i, 2] + case['height_m']
        roots[i, 3:7] = torch.tensor(case.get('quat_wxyz', [1., 0., 0., 0.]),
                                     device=env.device)
        roots[i, 7:] = 0.
        for name, value in case.get('joint_angles', {}).items():
            joints[i, robot.joint_names.index(name)] = value
        labels.append(case['name'])
    robot.write_root_state_to_sim(roots)
    robot.write_joint_state_to_sim(joints, velocities)
    robot.set_joint_position_target(joints)
    robot.set_joint_velocity_target(velocities)
    env.scene.reset()
    rows = []
    with torch.inference_mode():
        for step in range(args.steps):
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(dt=cfg.sim.dt)
            sensor.update(cfg.sim.dt, force_recompute=True)
            base_forces = sensor.data.net_forces_w_history[:, :, base_ids]
            base_forces = base_forces.norm(dim=-1).amax(dim=(1, 2))
            illegal = env.termination_manager.compute().clone()
            row = {'physics_step': step + 1,
                   'base_contact_n': base_forces.cpu().tolist(),
                   'illegal_contact': illegal.cpu().tolist(),
                   'root_pose_w': robot.data.root_state_w[:, :7].cpu().tolist()}
            all_forces = sensor.data.net_forces_w.norm(dim=-1)
            maxima, indices = all_forces.max(dim=1)
            row['max_contact_n'] = maxima.cpu().tolist()
            row['max_contact_body'] = [sensor.body_names[i]
                                       for i in indices.cpu().tolist()]
            if args.pair_forces:
                pairs = env.scene.sensors['base_pair_contacts']
                pairs.update(cfg.sim.dt, force_recompute=True)
                row['base_to_leg_contact_n'] = (
                    pairs.data.force_matrix_w[:, 0].norm(dim=-1).cpu().tolist()
                )
            rows.append(row)
    return {'test_only_overrides': True,
            'description': '固定姿态接触探针，不执行策略或自动复位。',
            'cases': cases, 'env_case_order': labels,
            'physics_dt_s': cfg.sim.dt,
            'env_origins_w': env.scene.env_origins.cpu().tolist(),
            'max_base_contact_n_per_env': [
                max(row['base_contact_n'][i] for row in rows)
                for i in range(args.num_envs)],
            'any_illegal_contact_per_env': [
                any(row['illegal_contact'][i] for row in rows)
                for i in range(args.num_envs)],
            'trace': rows}


def training_rollout(env, args):
    """保留训练随机化和自动复位，记录复位前真实状态。"""
    import torch

    contact_trace = None
    if args.shape_contacts:
        from race_contact_trace import RaceContactTrace
        contact_trace = RaceContactTrace(env, args.collision_json)

    robot = env.scene['robot']
    sensor = env.scene.sensors['contact_forces']
    body_ids, body_names = sensor.find_bodies('base_link')
    assert body_names == ['base_link'], body_names
    lengths = []
    trace = []
    samples = []
    termination_counts = {name: 0 for name in
                          env.termination_manager.active_terms}
    action_rng = torch.Generator(device=env.device)
    action_rng.manual_seed(args.action_seed)
    original_reset = env._reset_idx

    def capture_reset(env_ids):
        # 所有状态在真正复位前读取，避免把复位后的高度当成故障高度。
        for name in termination_counts:
            termination_counts[name] += int(
                env.termination_manager.get_term(name)[env_ids].sum()
            )
        lengths.extend(env.episode_length_buf[env_ids].cpu().tolist())
        force = sensor.data.net_forces_w_history[env_ids][:, :, body_ids]
        force = force.norm(dim=-1).amax(dim=(1, 2))
        height = robot.data.root_pos_w[env_ids, 2]
        height = height - env.scene.env_origins[env_ids, 2]
        upright = -robot.data.projected_gravity_b[env_ids, 2]
        if contact_trace:
            contact_trace.capture_reset(env_ids, height, upright, force)
        trace.append({
            'count': len(env_ids),
            'mean_steps': env.episode_length_buf[env_ids].float().mean().item(),
            'min_height_m': height.min().item(),
            'max_height_m': height.max().item(),
            'max_base_contact_n': force.max().item(),
            'min_upright': upright.min().item(),
        })
        if len(samples) < 10:
            i = int(env_ids[0])
            samples.append({
                'length_steps': int(env.episode_length_buf[i]),
                'height_m': float(height[0]),
                'base_contact_n': float(force[0]),
                'upright': float(upright[0]),
                'joint_pos_rad': robot.data.joint_pos[i].cpu().tolist(),
                'joint_vel_rad_s': robot.data.joint_vel[i].cpu().tolist(),
                'joint_names': robot.joint_names,
            })
            if args.pair_forces:
                pairs = env.scene.sensors['base_pair_contacts']
                pairs.update(env.physics_dt, force_recompute=True)
                values = pairs.data.force_matrix_w[i, 0].norm(dim=-1)
                samples[-1]['base_pair_contact_n'] = values.cpu().tolist()
        return original_reset(env_ids)

    env._reset_idx = capture_reset
    with torch.inference_mode():
        for step in range(args.steps):
            if args.joint_delta_bound is None:
                action = torch.randn((args.num_envs, 12), device=env.device,
                                     generator=action_rng)
                action *= args.action_std
            else:
                # 原配置动作缩放为 0.25；这里限制实际目标角的均匀扰动。
                action = torch.rand((args.num_envs, 12), device=env.device,
                                    generator=action_rng)
                action = (2 * action - 1) * args.joint_delta_bound / 0.25
            env.step(action)
            if step % 20 == 19:
                recent = lengths[-100:]
                mean = sum(recent) / len(recent) if recent else None
                print(json.dumps({'step': step + 1,
                                  'completed_episodes': len(lengths),
                                  'recent_mean_steps': mean},
                                 ensure_ascii=False), flush=True)
    shape_report = contact_trace.report() if contact_trace else None
    if contact_trace:
        contact_trace.close()
    return {
        'test_only_overrides': True,
        'self_collisions': not args.no_self_collisions,
        'num_envs': args.num_envs,
        'steps': args.steps,
        'contact_offset_override_m': args.contact_offset,
        'training_randomization_enabled': True,
        'action_std': (args.action_std
                       if args.joint_delta_bound is None else None),
        'joint_delta_bound_rad': args.joint_delta_bound,
        'action_distribution': ('normal' if args.joint_delta_bound is None
                                else 'uniform_target_delta'),
        'action_seed': args.action_seed,
        'action_sequence_independent_of_reset_rng': True,
        'total_environment_steps': args.num_envs * args.steps,
        'resets_per_1000_environment_steps': (
            1000 * len(lengths) / (args.num_envs * args.steps)
        ),
        'completed_episodes_under_3_steps': sum(v <= 2 for v in lengths),
        'unfinished_episode_steps': env.episode_length_buf.cpu().tolist(),
        'completed_episodes': len(lengths),
        'termination_counts': termination_counts,
        'shape_contact_trace': shape_report,
        'mean_completed_episode_steps': (sum(lengths) / len(lengths)
                                          if lengths else None),
        'recent_mean_episode_steps': (sum(lengths[-100:]) / len(lengths[-100:])
                                      if lengths else None),
        'first_ten_reset_samples': samples,
        'pair_body_order': [f'{kind}_leg_{leg}_link'
                            for leg in ('f_l', 'f_r', 'b_l', 'b_r')
                            for kind in ('pitch01', 'pitch02')]
                           if args.pair_forces else None,
        'reset_trace': trace,
    }


def main():
    import gymnasium as gym
    import torch
    import robot_lab.tasks  # noqa: F401
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    from robot_lab.tasks.race.env_cfg import joint_parameter

    cfg = load_cfg_from_registry('RobotLab-Race-Flat-v0', 'env_cfg_entry_point')
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.seed = args.seed
    # 所有覆盖仅作用于本次诊断：缩小资源、去掉纹理和随机扰动。
    cfg.sim.physx.gpu_max_rigid_patch_count = 32768
    cfg.sim.physx.gpu_max_rigid_contact_count = 262144
    cfg.sim.physx.gpu_found_lost_pairs_capacity = 262144
    cfg.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 262144
    cfg.sim.physx.gpu_total_aggregate_pairs_capacity = 262144
    cfg.scene.terrain.visual_material = None
    cfg.scene.sky_light = None
    generator = cfg.scene.terrain.terrain_generator
    generator.num_rows = 1
    generator.num_cols = 1
    cfg.scene.terrain.max_init_terrain_level = 0
    candidate_metadata = None
    if args.unfiltered_hip_housings:
        cfg.scene.robot.spawn.filter_base_hip_housings = False
        cfg.scene.robot.spawn.usd_dir = str(Path(
            cfg.scene.robot.spawn.usd_dir
        ).with_name('usd_go2_training_unfiltered_diagnosis'))
    if args.collision_json is not None:
        from race_collision_assets import configure_candidate
        cfg.scene.robot.spawn, candidate_metadata = configure_candidate(
            cfg.scene.robot.spawn, args.collision_json,
        )
    if args.contact_offset is not None:
        from copy import deepcopy
        from robot_lab.assets.race import RaceUrdfFileCfg
        original_spawn = cfg.scene.robot.spawn
        diagnostic_spawn = RaceUrdfFileCfg()
        for name, value in vars(original_spawn).items():
            if name != 'func':
                setattr(diagnostic_spawn, name, deepcopy(value))
        # 使用独立缓存，只改本次诊断的接触提前距离，不动原训练 USD。
        diagnostic_spawn.usd_dir = str(
            Path(original_spawn.usd_dir).with_name('usd_contact_diagnosis')
        )
        diagnostic_spawn.imported_contact_offset = args.contact_offset
        diagnostic_spawn.imported_rest_offset = 0.
        cfg.scene.robot.spawn = diagnostic_spawn
    if args.pair_forces:
        from isaaclab.sensors import ContactSensorCfg
        cfg.scene.base_pair_contacts = ContactSensorCfg(
            prim_path='{ENV_REGEX_NS}/Robot/base_link',
            update_period=0.,
            filter_prim_paths_expr=[
                '{ENV_REGEX_NS}/Robot/' + body
                for leg in ('f_l', 'f_r', 'b_l', 'b_r')
                for body in (f'pitch01_leg_{leg}_link',
                             f'pitch02_leg_{leg}_link')
            ],
        )
    if not args.training_resets:
        for name in list(vars(cfg.events)):
            if name.startswith('randomize'):
                setattr(cfg.events, name, None)
        cfg.events.reset_base.params['pose_range'] = {}
        cfg.events.reset_base.params['velocity_range'] = {}
        cfg.events.reset_robot_joints.params['position_range'] = (1., 1.)
        cfg.events.reset_robot_joints.params['velocity_range'] = (0., 0.)
    cfg.curriculum.terrain_levels = None
    cfg.observations.policy.enable_corruption = False
    cfg.observations.single_obs.enable_corruption = False
    cfg.commands.base_velocity.debug_vis = False
    if args.no_self_collisions:
        cfg.scene.robot.spawn.articulation_props.enabled_self_collisions = False
    if args.armature is not None:
        cfg.scene.robot.actuators['GO2HV'].armature = joint_parameter(args.armature)
    action_description = (f'动作标准差={args.action_std}'
                          if args.joint_delta_bound is None else
                          f'均匀目标角扰动=±{args.joint_delta_bound} rad')
    print(f'[诊断] 训练复位随机化={args.training_resets}；'
          f'{action_description}。', flush=True)
    env = gym.make('RobotLab-Race-Flat-v0', cfg=cfg).unwrapped
    try:
        env.reset()
        from pxr import PhysxSchema, Usd, UsdPhysics
        import omni.usd
        stage = omni.usd.get_context().get_stage()
        root = stage.GetPrimAtPath('/World/envs/env_0/Robot')
        filter_report = []
        for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
            if prim.HasAPI(UsdPhysics.FilteredPairsAPI):
                targets = UsdPhysics.FilteredPairsAPI(
                    prim
                ).GetFilteredPairsRel().GetTargets()
                filter_report.append({'source': str(prim.GetPath()),
                                      'targets': list(map(str, targets))})
        expected_filter = cfg.scene.robot.spawn.filter_base_hip_housings
        if expected_filter:
            assert len(filter_report) == 1, filter_report
            assert filter_report[0]['source'].endswith('/base_link')
            assert len(filter_report[0]['targets']) == 4
            for path in filter_report[0]['targets']:
                prim = stage.GetPrimAtPath(path)
                assert prim and prim.HasAPI(UsdPhysics.CollisionAPI), path
                assert path.startswith('/World/envs/env_0/Robot/'), path
        verification_path = args.collision_json or args.verify_collision_json
        if verification_path is not None:
            from race_collision_assets import verify_candidate
            if candidate_metadata is None:
                import hashlib
                candidate_metadata = {
                    'verification_only': True,
                    'active_asset_path': cfg.scene.robot.spawn.asset_path,
                    'source_json': str(verification_path.resolve()),
                    'source_sha256': hashlib.sha256(
                        verification_path.read_bytes()).hexdigest(),
                }
            candidate_metadata.update(verify_candidate(stage,
                                                        verification_path))
        collider_offsets = []
        for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                api = PhysxSchema.PhysxCollisionAPI(prim)
                collider_offsets.append({
                    'path': str(prim.GetPath()),
                    'contact_offset_m': api.GetContactOffsetAttr().Get(),
                    'rest_offset_m': api.GetRestOffsetAttr().Get(),
                })
        if args.training_resets or args.contact_probe_poses:
            report = (controlled_contact_probes(env, cfg,
                                                args.contact_probe_poses)
                      if args.contact_probe_poses else
                      training_rollout(env, args))
            report['seed'] = args.seed
            report['candidate_metadata'] = candidate_metadata
            report['collider_offsets'] = collider_offsets
            report['base_hip_housing_filter_enabled'] = expected_filter
            report['collision_filter_relationships'] = filter_report
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, ensure_ascii=False,
                                               indent=2) + '\n')
            print(f'[诊断完成] {args.output}', flush=True)
            return
        robot = env.scene['robot']
        sensor = env.scene.sensors['contact_forces']
        ids, names = sensor.find_bodies('base_link')
        assert names == ['base_link'], names
        base_body = robot.body_names.index('base_link')
        action = torch.zeros((args.num_envs, 12), device=env.device)
        actuator = robot.actuators['GO2HV']
        rows = []
        initial = {
            'root_height_m': (robot.data.root_pos_w[:, 2]
                              - env.scene.env_origins[:, 2]).cpu().tolist(),
            'joint_names': robot.joint_names,
            'joint_pos_rad': robot.data.joint_pos.cpu().tolist(),
            'armature': actuator.armature.cpu().tolist(),
            'base_sensor_bodies': names,
        }
        first_termination = None
        with torch.inference_mode():
            for step in range(args.steps):
                env.action_manager.process_action(action)
                for _ in range(cfg.decimation):
                    env.action_manager.apply_action()
                    env.scene.write_data_to_sim()
                    env.sim.step(render=False)
                    env.scene.update(dt=cfg.sim.dt)
                # 直接读求解后的状态，不让 env.step 自动复位覆盖故障现场。
                sensor.update(cfg.sim.dt, force_recompute=True)
                force = sensor.data.net_forces_w_history[:, :, ids].norm(dim=-1)
                force = force.amax(dim=(1, 2))
                height = robot.data.body_pos_w[:, base_body, 2]
                height = height - env.scene.env_origins[:, 2]
                bad = force > 1.
                row = {
                    'step': step + 1,
                    'time_s': (step + 1) * cfg.decimation * cfg.sim.dt,
                    'base_height_m': height.cpu().tolist(),
                    'base_contact_n': force.cpu().tolist(),
                    'illegal_contact_fraction': bad.float().mean().item(),
                    'max_abs_joint_vel_rad_s': robot.data.joint_vel.abs().amax().item(),
                    'max_abs_torque_nm': robot.data.applied_torque.abs().amax().item(),
                    'min_upright': (-robot.data.projected_gravity_b[:, 2]).amin().item(),
                }
                if first_termination is None and bad.any():
                    first_termination = row
                rows.append(row)
                if step < 5 or step % 20 == 19:
                    print(json.dumps(row, ensure_ascii=False), flush=True)
                if not torch.isfinite(robot.data.joint_vel).all():
                    raise RuntimeError('关节速度出现非有限值')
        report = {
            'test_only_overrides': True,
            'candidate_metadata': candidate_metadata,
            'base_hip_housing_filter_enabled': expected_filter,
            'collision_filter_relationships': filter_report,
            'self_collisions': not args.no_self_collisions,
            'initial': initial,
            'collider_offsets': collider_offsets,
            'first_base_contact_over_1n': first_termination,
            'last_step': rows[-1],
            'trace': rows,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)
                               + '\n')
        print(f'[诊断完成] {args.output}', flush=True)
    finally:
        env.close()


try:
    main()
except Exception:
    # Kit 关闭过程可能提前结束解释器，先输出异常，防止被退出过程吞掉。
    import traceback
    traceback.print_exc()
    raise
finally:
    launcher.app.close()
