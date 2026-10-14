"""核对原 Go2 训练规则及用户指定的 Race 模型、电机和保存间隔适配。"""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

import h5py  # noqa: F401
import tensordict  # noqa: F401

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path,
                    default=ROOT / "logs/original_go2_audit/config.json")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)


def differences(left, right, path=""):
    """递归比较所有配置字段，函数统一用模块与名称表示。"""
    if callable(left):
        left = f"{left.__module__}:{left.__qualname__}"
    if callable(right):
        right = f"{right.__module__}:{right.__qualname__}"
    if isinstance(left, dict) and isinstance(right, dict):
        result = []
        for name in sorted(set(left) | set(right)):
            current = f"{path}.{name}" if path else name
            if name not in left or name not in right:
                result.append(current)
            else:
                result += differences(left[name], right[name], current)
        return result
    if isinstance(left, (tuple, list)) and isinstance(right, (tuple, list)):
        if len(left) != len(right):
            return [path]
        result = []
        for index, (a, b) in enumerate(zip(left, right)):
            result += differences(a, b, f"{path}.{index}")
        return result
    return [] if left == right else [path]


def check_actuator(cfg, joint_names):
    """逆序构造实际执行器，核验关节映射、PD 和双向力矩转速限幅。"""
    import torch
    from isaaclab.utils.types import ArticulationActions

    names = list(reversed(joint_names))
    actuator = cfg.class_type(
        deepcopy(cfg), joint_names=names, joint_ids=slice(None),
        num_envs=1, device=args.device,
    )
    expected_by_kind = {
        "roll": (30.0, 1.15, 22.0, 31.4),
        "pitch01": (48.0, 1.20, 22.0, 31.4),
        "pitch02": (51.0, 1.00, 33.0, 21.0),
    }
    expected = torch.tensor(
        [expected_by_kind[name.split("_leg_", 1)[0]] for name in names],
        device=args.device,
    )
    assert torch.allclose(actuator.stiffness[0], expected[:, 0])
    assert torch.allclose(actuator.damping[0], expected[:, 1])
    for limit in (actuator.effort_limit, actuator.effort_limit_sim,
                  actuator._effort_y1, actuator._effort_y2):
        assert torch.allclose(limit[0], expected[:, 2])
    for limit in (actuator.velocity_limit, actuator.velocity_limit_sim,
                  actuator._velocity_x2):
        assert torch.allclose(limit[0], expected[:, 3])
    assert torch.allclose(actuator._velocity_x1[0], expected[:, 3] * 0.45)
    positions = torch.zeros_like(actuator.stiffness)
    actuator.reset(None)
    checked = 0
    # 每个关节都覆盖满力矩区、拐点、中点、速度端点及超速区域。
    for speed_ratio in (0.0, 0.1, 0.45, 0.725, 1.0, 1.2):
        fraction = min(1.0, max(0.0, (1.0 - speed_ratio) / 0.55))
        for velocity_sign in (-1.0, 1.0):
            velocities = expected[:, 3].unsqueeze(0) * (
                speed_ratio * velocity_sign
            )
            for torque_sign in (-1.0, 1.0):
                # 重复提交目标，覆盖原版最多四步延迟。
                for _ in range(cfg.max_delay + 1):
                    result = actuator.compute(
                        ArticulationActions(
                            joint_positions=positions + 100.0 * torque_sign,
                            joint_velocities=torch.zeros_like(positions),
                            joint_efforts=torch.zeros_like(positions),
                        ),
                        positions, velocities,
                    )
                expected_effort = expected[:, 2] * fraction * torque_sign
                assert torch.allclose(result.joint_efforts[0], expected_effort,
                                      atol=1e-5, rtol=1e-6)
                checked += 1
    return {"passed": True, "speed_direction_cases": checked,
            "reversed_joint_order_verified": True}


def main():
    import gymnasium as gym
    import numpy as np
    from isaaclab.managers import SceneEntityCfg
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    import robot_lab.tasks  # noqa: F401
    from robot_lab.tasks.go2.env_cfg import Go2EnvCfg
    from robot_lab.tasks.go2.rsl_rl_cfg import MoECTSRunnerCfg
    from robot_lab.tasks.race.env_cfg import NAME_MAP, MODEL_SPAWN_FIELDS

    original = Go2EnvCfg()
    before = deepcopy(original.to_dict())
    race = load_cfg_from_registry("RobotLab-Race-v0", "env_cfg_entry_point")
    flat = load_cfg_from_registry("RobotLab-Race-Flat-v0", "env_cfg_entry_point")
    assert not differences(before, original.to_dict()), "污染了原 Go2 配置"
    assert not differences(before, Go2EnvCfg().to_dict()), "污染了原 Go2 类默认值"
    expected = deepcopy(original)
    # 仅把 Go2 实体名映射到 Race，其余递归比较，包含所有数值参数。
    def remap(value):
        if isinstance(value, SceneEntityCfg):
            for field in ("joint_names", "body_names"):
                names = getattr(value, field)
                if isinstance(names, str):
                    setattr(value, field, NAME_MAP.get(names, names))
                elif isinstance(names, (list, tuple)):
                    setattr(value, field, [NAME_MAP.get(n, n) for n in names])
        elif isinstance(value, dict):
            for child in value.values():
                remap(child)
        elif hasattr(value, "__dict__") and not callable(value):
            for child in vars(value).values():
                remap(child)

    for group in (expected.observations, expected.events, expected.rewards,
                  expected.terminations, expected.curriculum):
        remap(group)
    expected.actions.joint_pos.joint_names = race.actions.joint_pos.joint_names
    expected.actions.joint_pos.scale = {
        NAME_MAP.get(k, k): v for k, v in expected.actions.joint_pos.scale.items()
    }
    for scanner in (expected.scene.height_scanner,
                    expected.scene.height_scanner_small):
        scanner.prim_path = scanner.prim_path.replace("/base", "/base_link")
    # 用户指定的机器人高度适配：释放 0.35～0.37 m，奖励目标 0.335 m。
    expected.scene.robot.init_state.pos = (0.0, 0.0, 0.36)
    expected.events.reset_base.params["pose_range"]["z"] = (-0.01, 0.01)
    expected.rewards.base_height_l2.params["target_height"] = 0.335
    expected.rewards.feet_regulation.params["base_height_target"] = 0.335
    expected_actuator = expected.scene.robot.actuators["GO2HV"]
    settings = {
        "stiffness": (30.0, 48.0, 51.0),
        "damping": (1.15, 1.20, 1.00),
        "Y1": (22.0, 22.0, 33.0),
        "Y2": (22.0, 22.0, 33.0),
        "effort_limit": (22.0, 22.0, 33.0),
        "effort_limit_sim": (22.0, 22.0, 33.0),
        "X1": tuple(v * 0.45 for v in (31.4, 31.4, 21.0)),
        "X2": (31.4, 31.4, 21.0),
        "velocity_limit": (31.4, 31.4, 21.0),
        "velocity_limit_sim": (31.4, 31.4, 21.0),
    }
    from robot_lab.assets.race import JOINT_NAMES
    for field, values in settings.items():
        by_kind = dict(zip(("roll", "pitch01", "pitch02"), values))
        setattr(expected_actuator, field, {
            name: by_kind[name.split("_leg_", 1)[0]] for name in JOINT_NAMES
        })
    stand_by_kind = {"roll": 0.0, "pitch01": -0.12, "pitch02": 0.30}
    assert race.scene.robot.init_state.joint_pos == {
        name: stand_by_kind[name.split("_leg_", 1)[0]] for name in JOINT_NAMES
    }
    env_changes = differences(expected.to_dict(), race.to_dict())
    allowed = {
        "scene.robot.init_state.joint_pos",
        "scene.robot.joint_sdk_names",
        *(f"scene.robot.spawn.{name}" for name in MODEL_SPAWN_FIELDS),
        "scene.robot.spawn.import_inertia_tensor",
    }
    unexpected = [p for p in env_changes if not any(
        p == prefix or p.startswith(prefix + ".") for prefix in allowed
    )]
    assert not unexpected, f"出现模型适配之外的训练差异：{unexpected}"
    assert race.scene.robot.spawn.filter_base_hip_housings is True

    runner = MoECTSRunnerCfg().to_dict()
    race_runner = load_cfg_from_registry(
        "RobotLab-Race-v0", "rsl_rl_cfg_entry_point",
    ).to_dict()
    runner["experiment_name"] = race_runner["experiment_name"]
    runner_changes = differences(runner, race_runner)
    assert runner_changes == ["save_interval"], runner_changes
    runner["save_interval"] = 1000
    assert not differences(runner, race_runner), "训练器存在用户指定外的差异"
    flat_runner = load_cfg_from_registry(
        "RobotLab-Race-Flat-v0", "rsl_rl_cfg_entry_point",
    ).to_dict()
    flat_runner["experiment_name"] = race_runner["experiment_name"]
    assert not differences(race_runner, flat_runner)
    for task in ("RobotLab-Race-v0", "RobotLab-Race-Flat-v0",
                 "RobotLab-Race-Go2Reference-v0"):
        assert gym.spec(task).entry_point == gym.spec("RobotLab-Go2-v0").entry_point
    flat_expected = deepcopy(race.to_dict())
    flat_expected["scene"]["terrain"]["terrain_generator"]["sub_terrains"] = (
        flat.to_dict()["scene"]["terrain"]["terrain_generator"]["sub_terrains"]
    )
    assert not differences(flat_expected, flat.to_dict())

    generator = race.scene.terrain.terrain_generator
    names = list(generator.sub_terrains)
    proportions = np.array([c.proportion for c in generator.sub_terrains.values()])
    cumulative = np.cumsum(proportions / proportions.sum())
    counts = {name: 0 for name in names}
    for index in range(generator.num_cols):
        counts[names[int(np.where(index / generator.num_cols + .001
                                  < cumulative)[0].min())]] += 1
    assert counts["stairs_up"] == 5 and counts["stairs_down"] == 2
    actuator = race.scene.robot.actuators["GO2HV"]
    assert actuator.to_dict() == expected_actuator.to_dict()
    actuator_result = check_actuator(actuator, JOINT_NAMES)
    assert race.scene.num_envs == flat.scene.num_envs == 16384
    release_height = race.scene.robot.init_state.pos[2]
    reset_range = race.events.reset_base.params["pose_range"]["z"]
    assert release_height == 0.36
    assert np.allclose(np.array(reset_range) + release_height, [0.35, 0.37])
    assert race.rewards.base_height_l2.params["target_height"] == 0.335
    assert race.rewards.feet_regulation.params["base_height_target"] == 0.335
    reward_names = {name for name, value in vars(race.rewards).items()
                    if value is not None and not name.startswith("_")}
    assert len(reward_names) == 15
    assert not reward_names.intersection({"feet_air_time", "feet_height",
                                          "feet_height_body", "stairs_success"})
    # 核对旧权重来源验证器兼容两种执行器，并拒绝交叉加载。
    sys.path.insert(0, str(ROOT / "scripts/rsl_rl"))
    from race_checkpoint import verify_race_checkpoint
    from isaaclab.utils.io import dump_yaml
    fixture = args.output.parent / "checkpoint_interface"
    fixture.mkdir(parents=True, exist_ok=True)
    dump_yaml(str(fixture / "params/env.yaml"), race)
    checkpoint = fixture / "model_0.pt"
    checkpoint.touch()
    verify_race_checkpoint(checkpoint, race)
    legacy = load_cfg_from_registry(
        "RobotLab-Race-Legacy-v0", "env_cfg_entry_point",
    )
    legacy_fixture = args.output.parent / "legacy_checkpoint_interface"
    legacy_fixture.mkdir(parents=True, exist_ok=True)
    dump_yaml(str(legacy_fixture / "params/env.yaml"), legacy)
    legacy_checkpoint = legacy_fixture / "model_0.pt"
    legacy_checkpoint.touch()
    verify_race_checkpoint(legacy_checkpoint, legacy)
    try:
        verify_race_checkpoint(legacy_checkpoint, race)
    except ValueError:
        pass
    else:
        raise AssertionError("未拒绝把历史 PD 权重加载到原版 GO2HV")
    bad = deepcopy(race)
    bad.scene.robot.actuators["GO2HV"].stiffness[JOINT_NAMES[0]] += 1.0
    try:
        verify_race_checkpoint(checkpoint, bad)
    except ValueError:
        pass
    else:
        raise AssertionError("未拒绝不同的执行器参数")

    report = {
        "passed": True,
        "upstream_commit": "28b4516d22617b11aeaf8ead63cc00b0c0bcd1bd",
        "unexpected_training_differences": unexpected,
        "runner_differences_except_log_name": runner_changes,
        "model_import_differences": env_changes,
        "model_height_adaptation": {
            "original_go2_release_height_m": original.scene.robot.init_state.pos[2],
            "original_go2_reward_target_m": .38,
            "race_release_height_m": release_height,
            "race_release_height_range_m": [release_height + v
                                             for v in reset_range],
            "race_reward_target_m": 0.335,
            "reward_form_weights_curriculum_unchanged": True,
        },
        "terrain_columns": counts,
        "flat_changes_only_sub_terrains": True,
        "original_go2_unmodified_by_race": True,
        "default_num_envs": race.scene.num_envs,
        "episode_seconds": race.episode_length_s,
        "max_iterations": race_runner["max_iterations"],
        "save_interval": race_runner["save_interval"],
        "exploration_noise": race_runner["policy"]["init_noise_std"],
        "actuator": {k: getattr(actuator, k) for k in
                     ("stiffness", "damping", "min_delay", "max_delay",
                      "X1", "X2", "Y1", "Y2", "effort_limit",
                      "effort_limit_sim", "velocity_limit",
                      "velocity_limit_sim")},
        "torque_speed_model_assumption": {
            "velocity_source": "Race URDF limit.velocity",
            "full_torque_knee_ratio": 0.45,
            "knee_ratio_is_measured": False,
        },
        "default_joint_pos": race.scene.robot.init_state.joint_pos,
        "actuator_runtime_test": actuator_result,
        "checkpoint_interface_verified": True,
        "legacy_checkpoint_cross_loading_rejected": True,
        "active_reward_count": len(reward_names),
        "extra_stairs_clearance_airtime_rewards_enabled": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print("[通过] 原 Go2 规则不变；Race 模型、电机及用户保存间隔核验通过。")
    print(json.dumps(report, ensure_ascii=False, indent=2))


try:
    main()
finally:
    launcher.app.close()
