"""检查历史 Race 固定 PD 接口；原版训练对照请用 check_race_original.py。"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import traceback
import xml.etree.ElementTree as ET

# 在 Kit 导入前加载原生扩展，保持与训练入口相同的初始化次序。
import h5py  # noqa: F401
import tensordict  # noqa: F401

from isaaclab.app import AppLauncher


ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="RobotLab-Race-Legacy-Flat-v0")
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "logs/race-training/structure.json",
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.num_envs < 1 or args.steps < 1:
        parser.error("并行环境数与测试步数必须为正数。")
    return args


def deterministic_config(cfg, steps):
    """仅在验证实例中关闭扰动，正式训练配置不受影响。"""
    for name in vars(cfg.events):
        if name.startswith("randomize"):
            setattr(cfg.events, name, None)
    cfg.events.reset_robot_joints.params["position_range"] = (0.0, 0.0)
    cfg.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
    cfg.events.reset_base.params["pose_range"] = {}
    cfg.events.reset_base.params["velocity_range"] = {}
    cfg.observations.policy.enable_corruption = False
    cfg.observations.single_obs.enable_corruption = False
    cfg.curriculum.terrain_levels = None
    if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
    ranges = cfg.commands.base_velocity.ranges
    for name in ("lin_vel_x", "lin_vel_y", "ang_vel_z"):
        setattr(ranges, name, (0.0, 0.0))
    cfg.commands.base_velocity.rel_standing_envs = 1.0
    cfg.commands.base_velocity.debug_vis = False
    # 增加测试步数时也不允许时间上限触发自动复位、掩盖支撑情况。
    cfg.episode_length_s = max(
        cfg.episode_length_s, (steps + 5) * cfg.decimation * cfg.sim.dt
    )
    cfg.seed = 0
    return cfg


def assert_close(actual, expected, label, tolerance=1.0e-5):
    import torch

    expected_tensor = torch.as_tensor(
        expected, device=actual.device, dtype=actual.dtype
    )
    if not torch.allclose(actual, expected_tensor, atol=tolerance, rtol=0):
        raise AssertionError(
            f"{label} 不符：实际 {actual.detach().cpu().tolist()}，"
            f"预期 {expected_tensor.detach().cpu().tolist()}"
        )


def check_articulation(robot):
    """核对按关节名排列的姿态、执行器参数和 PhysX 被动参数。"""
    import torch

    from isaaclab.utils.types import ArticulationActions
    from robot_lab.assets.race import (
        FOOT_LINK_NAMES, JOINT_NAMES, RACE_CONTROL,
    )

    assert len(robot.joint_names) == 12, robot.joint_names
    assert set(robot.joint_names) == set(JOINT_NAMES), robot.joint_names
    assert set(FOOT_LINK_NAMES).issubset(robot.body_names), robot.body_names
    ordered_ids = [robot.joint_names.index(name) for name in JOINT_NAMES]
    assert_close(
        robot.data.default_joint_pos[:, ordered_ids],
        [RACE_CONTROL["default_angles"]] * robot.num_instances,
        "默认关节角",
    )
    assert_close(
        robot.data.joint_armature[:, ordered_ids],
        [RACE_CONTROL["armature"]] * robot.num_instances,
        "仿真关节 armature",
    )
    zero_properties = (
        "joint_stiffness", "joint_damping", "joint_friction_coeff",
        "joint_dynamic_friction_coeff", "joint_viscous_friction_coeff",
    )
    for name in zero_properties:
        assert_close(getattr(robot.data, name), 0.0, f"PhysX {name}")
    assert_close(
        robot.root_physx_view.get_dof_stiffnesses(), 0.0,
        "PhysX 实际关节刚度",
    )
    assert_close(
        robot.root_physx_view.get_dof_dampings(), 0.0,
        "PhysX 实际关节阻尼",
    )
    assert_close(
        robot.root_physx_view.get_dof_friction_properties(), 0.0,
        "PhysX 实际三类关节摩擦",
    )
    actuator = robot.actuators["legs"]
    actuator_ids = [actuator.joint_names.index(name) for name in JOINT_NAMES]
    parameters = {
        "stiffness": "kps", "damping": "kds",
        "effort_limit": "torque_limits", "armature": "armature",
        "passive_damping": "passive_damping",
        "passive_friction": "passive_friction",
    }
    for attribute, source_key in parameters.items():
        assert_close(
            getattr(actuator, attribute)[:, actuator_ids],
            [RACE_CONTROL[source_key]] * robot.num_instances,
            f"执行器 {attribute}",
        )
    # 实际调用执行器验证饱和次序及速度死区；随后 env.step 会重算缓存。
    q = torch.zeros_like(actuator.stiffness)
    dq = torch.tensor(
        [0.0, 1.0e-7, 1.0, -2.0] * 3,
        device=q.device, dtype=q.dtype,
    ).expand_as(q)
    target = torch.full_like(q, 10.0)
    motor = actuator.stiffness * target - actuator.damping * dq
    clipped = torch.clamp(
        motor, min=-actuator.effort_limit, max=actuator.effort_limit
    )
    friction = torch.where(dq.abs() > 1.0e-6, dq.sign(), 0.0)
    passive = -actuator.passive_damping * dq
    passive -= actuator.passive_friction * friction
    action = ArticulationActions(
        joint_positions=target,
        joint_velocities=torch.zeros_like(q),
        joint_efforts=torch.zeros_like(q),
    )
    actual = actuator.compute(action, q, dq)
    assert_close(actual.joint_efforts, clipped + passive, "实际 PD 合成力矩")
    assert_close(
        actuator.computed_effort - actuator.applied_effort,
        motor - clipped, "电机饱和残差", tolerance=5.0e-5,
    )
    masses = robot.root_physx_view.get_masses()
    totals = masses.sum(dim=1)
    assert_close(totals, 15.266135420222458, "整机原始质量", 3.0e-5)
    return {
        "joint_names_runtime": robot.joint_names,
        "joint_names_control": JOINT_NAMES,
        "body_names": robot.body_names,
        "foot_names": FOOT_LINK_NAMES,
        "mass_kg": totals.cpu().tolist(),
        "default_angles_rad": RACE_CONTROL["default_angles"],
        "actuator_parameters": {
            source_key: RACE_CONTROL[source_key]
            for source_key in parameters.values()
        },
        "physx_passive_parameters_zero": True,
        "pd_saturation_and_passive_terms_verified": True,
    }


def check_collision_geometry(stage):
    """遍历实例代理，核对碰撞几何数量、类型及固定足端球半径。"""
    from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics

    from robot_lab.assets.race import FOOT_LINK_NAMES

    robot_prim = stage.GetPrimAtPath("/World/envs/env_0/Robot")
    assert robot_prim.IsValid(), "env_0 中缺少 Robot 根节点。"
    colliders = []
    for prim in Usd.PrimRange(robot_prim, Usd.TraverseInstanceProxies()):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        if not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get():
            continue
        path = str(prim.GetPath())
        shape = prim.GetTypeName()
        item = {"path": path, "type": shape}
        properties = PhysxSchema.PhysxCollisionAPI(prim)
        item["contact_offset_m"] = properties.GetContactOffsetAttr().Get()
        item["rest_offset_m"] = properties.GetRestOffsetAttr().Get()
        assert abs(item["contact_offset_m"] - 0.001) < 1e-8, item
        assert abs(item["rest_offset_m"]) < 1e-8, item
        if prim.IsA(UsdGeom.Sphere):
            item["radius_m"] = UsdGeom.Sphere(prim).GetRadiusAttr().Get()
        elif prim.IsA(UsdGeom.Cylinder):
            geometry = UsdGeom.Cylinder(prim)
            item["radius_m"] = geometry.GetRadiusAttr().Get()
            item["length_m"] = geometry.GetHeightAttr().Get()
        elif prim.IsA(UsdGeom.Cube):
            item["size"] = UsdGeom.Cube(prim).GetSizeAttr().Get()
        colliders.append(item)
    counts = dict(Counter(item["type"] for item in colliders))
    assert len(colliders) == 26, (len(colliders), colliders)
    assert counts == {"Cube": 5, "Cylinder": 16, "Sphere": 5}, counts
    for foot in FOOT_LINK_NAMES:
        matches = [
            item for item in colliders
            if f"/{foot}/" in item["path"]
        ]
        assert len(matches) == 1, (foot, matches)
        assert matches[0]["type"] == "Sphere", (foot, matches)
        assert abs(matches[0]["radius_m"] - 0.0235) < 1.0e-7
    urdf = ROOT / "resources/race/urdf/race.urdf"
    digest = hashlib.sha256(urdf.read_bytes()).hexdigest()
    manifest = json.loads(
        (ROOT / "resources/race/import_manifest.json").read_text()
    )
    assert digest == manifest["files"]["urdf/race.urdf"], "URDF 已改变。"
    source_counts = Counter(
        collision.find("geometry")[0].tag
        for collision in ET.parse(urdf).findall(".//collision")
    )
    assert dict(source_counts) == {"box": 5, "cylinder": 16, "sphere": 5}
    return {
        "count": len(colliders), "types": counts,
        "foot_radius_m": 0.0235, "urdf_sha256": digest,
        "original_urdf_unchanged": True, "colliders": colliders,
    }


def check_mass_properties(stage, robot):
    """重建 USD 主惯量矩阵，与原 URDF 的完整张量和质心逐连杆核对。"""
    import numpy as np
    from pxr import UsdPhysics
    from scipy.spatial.transform import Rotation

    urdf = ET.parse(ROOT / "resources/race/urdf/race.urdf")
    maximum_inertia_error = 0.0
    maximum_com_error = 0.0
    for link in urdf.findall("link"):
        name = link.get("name")
        assert name in robot.body_names, name
        prim = stage.GetPrimAtPath(f"/World/envs/env_0/Robot/{name}")
        api = UsdPhysics.MassAPI(prim)
        inertial = link.find("inertial")
        expected_mass = float(inertial.find("mass").get("value"))
        assert abs(api.GetMassAttr().Get() - expected_mass) < 1e-6, name
        origin = inertial.find("origin")
        expected_com = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
        rpy = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
        rotation = Rotation.from_euler("xyz", rpy).as_matrix()
        v = {key: float(value) for key, value in inertial.find("inertia").attrib.items()}
        expected = np.array([[v["ixx"], v["ixy"], v["ixz"]],
                             [v["ixy"], v["iyy"], v["iyz"]],
                             [v["ixz"], v["iyz"], v["izz"]]])
        expected = rotation @ expected @ rotation.T
        quaternion = api.GetPrincipalAxesAttr().Get()
        q = [*quaternion.GetImaginary(), quaternion.GetReal()]
        axes = Rotation.from_quat(q).as_matrix()
        actual = axes @ np.diag(api.GetDiagonalInertiaAttr().Get()) @ axes.T
        error = float(np.max(np.abs(actual - expected)))
        com_error = float(np.max(np.abs(
            np.array(api.GetCenterOfMassAttr().Get()) - expected_com
        )))
        assert np.allclose(actual, expected, rtol=1e-4, atol=1e-7), (name, actual, expected)
        assert com_error < 1e-7, (name, com_error)
        maximum_inertia_error = max(maximum_inertia_error, error)
        maximum_com_error = max(maximum_com_error, com_error)
    return {"verified_link_count": len(urdf.findall("link")),
            "maximum_inertia_error_kg_m2": maximum_inertia_error,
            "maximum_com_error_m": maximum_com_error,
            "full_urdf_inertia_and_com_preserved": True}


def check_observations(observations, num_envs):
    import torch

    dimensions = {"policy": 450, "critic": 275, "single_obs": 45}
    for name, dimension in dimensions.items():
        values = observations[name]
        assert values.shape == (num_envs, dimension), (name, values.shape)
        assert torch.isfinite(values).all().item(), f"{name} 观测出现非有限值。"
    return dimensions


def run_checks(args):
    import gymnasium as gym
    import numpy as np
    import torch
    import omni.usd

    from isaaclab_tasks.utils import parse_env_cfg
    from robot_lab.assets.race import FOOT_LINK_NAMES
    import robot_lab.tasks  # noqa: F401

    cfg = parse_env_cfg(
        args.task, device=args.device or "cuda:0", num_envs=args.num_envs,
    )
    deterministic_config(cfg, args.steps)
    env = gym.make(args.task, cfg=cfg)
    try:
        raw = env.unwrapped
        observations, _ = env.reset(seed=0)
        raw.episode_length_buf.zero_()
        robot = raw.scene["robot"]
        sensor = raw.scene["contact_forces"]
        foot_ids, _ = sensor.find_bodies(FOOT_LINK_NAMES, preserve_order=True)
        structure = check_articulation(robot)
        stage = omni.usd.get_context().get_stage()
        geometry = check_collision_geometry(stage)
        mass_properties = check_mass_properties(stage, robot)
        dimensions = check_observations(observations, args.num_envs)
        action = torch.zeros(
            args.num_envs, raw.action_manager.total_action_dim,
            device=raw.device,
        )
        heights, tilts, forces, rewards = [], [], [], []
        terminations, timeouts = [], []
        with torch.inference_mode():
            for step in range(args.steps):
                observations, reward, terminated, truncated, _ = env.step(action)
                check_observations(observations, args.num_envs)
                assert torch.isfinite(reward).all().item(), "奖励出现非有限值。"
                heights.append(
                    robot.data.root_pos_w[:, 2].cpu().numpy().copy()
                )
                gravity_z = robot.data.projected_gravity_b[:, 2]
                tilt = torch.rad2deg(torch.acos((-gravity_z).clamp(-1, 1)))
                tilts.append(tilt.cpu().numpy().copy())
                force = sensor.data.net_forces_w[:, foot_ids, 2]
                assert torch.isfinite(force).all().item(), "足端力出现非有限值。"
                forces.append(force.cpu().numpy().copy())
                rewards.append(reward.cpu().numpy().copy())
                if terminated.any().item():
                    terminations.append({
                        "step": step + 1,
                        "env_ids": terminated.nonzero().flatten().cpu().tolist(),
                        "terms": {
                            name: raw.termination_manager.get_term(name)
                            .nonzero().flatten().cpu().tolist()
                            for name in raw.termination_manager.active_terms
                        },
                    })
                if truncated.any().item():
                    timeouts.append(step + 1)
        height = np.asarray(heights)
        tilt = np.asarray(tilts)
        force = np.asarray(forces)
        tail_count = min(args.steps, max(1, round(1.0 / raw.step_dt)))
        tail_force = force[-tail_count:]
        support = {
            "duration_s": args.steps * raw.step_dt,
            "steps": args.steps,
            "initialization_only": True,
            "height_last_m": height[-1].tolist(),
            "height_range_m": [float(height.min()), float(height.max())],
            "tilt_last_deg": tilt[-1].tolist(),
            "tilt_max_deg": float(tilt.max()),
            "foot_force_last_second_mean_n": tail_force.mean(axis=0).tolist(),
            "all_feet_contact_last_second_fraction": (
                (tail_force > 1.0).all(axis=2).mean(axis=0).tolist()
            ),
            "reward_mean": float(np.asarray(rewards).mean()),
            "unexpected_terminations": terminations,
            "timeouts": timeouts,
        }
        result = {
            "passed": not terminations and not timeouts,
            "task": args.task, "num_envs": args.num_envs,
            "physics_dt": raw.physics_dt, "policy_dt": raw.step_dt,
            "observations": dimensions,
            "structure": structure, "collision_geometry": geometry,
            "mass_properties": mass_properties,
            "short_pd": support,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({
            "passed": result["passed"], "output": str(args.output),
            "mass_kg": structure["mass_kg"],
            "collisions": geometry["types"],
            "observations": dimensions, "short_pd": support,
        }, ensure_ascii=False))
        assert not terminations, "短时零动作测试出现非预期终止，详见 JSON。"
        assert not timeouts, "短时测试发生非预期超时。"
    finally:
        env.close()


def main():
    args = parse_args()
    app = AppLauncher(args).app
    try:
        run_checks(args)
    except Exception as error:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result = {"passed": False}
        # 保留本轮已写出的支撑诊断，并避免旧的成功报告掩盖本轮失败。
        if args.output.exists():
            try:
                existing = json.loads(args.output.read_text(encoding="utf-8"))
                if existing.get("passed") is False:
                    result.update(existing)
            except (OSError, ValueError):
                pass
        result.update({
            "passed": False, "error": str(error),
            "traceback": traceback.format_exc(),
        })
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        raise
    finally:
        app.close()


if __name__ == "__main__":
    main()
