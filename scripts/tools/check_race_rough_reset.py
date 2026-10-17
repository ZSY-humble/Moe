"""核验复杂地形重置隔离、关节重排及独立均匀采样分布。"""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET


def run_checks(rough, baseline, flat):
    """比较完整配置，并用非原生关节顺序验证实际事件实现。"""
    import torch
    from isaaclab.managers import SceneEntityCfg
    from robot_lab.assets.race import JOINT_NAMES
    from robot_lab.tasks.race.events import RaceAdditiveJointReset

    expected_angles = dict(zip(JOINT_NAMES, (
        .10, .00, .10, -.10, .00, .10,
        .10, .20, .10, -.10, .20, .10,
    ), strict=True))
    expected_widths = dict(zip(JOINT_NAMES, (
        .0375, .3000, .5625, .0375, .3000, .5625,
        .0375, .3750, .5625, .0375, .3750, .5625,
    ), strict=True))
    assert rough.scene.robot.init_state.joint_pos == expected_angles
    assert rough.scene.robot.init_state.pos == (0., 0., .39)
    assert rough.events.reset_base.params["pose_range"]["z"] == (0., .16)
    event = rough.events.reset_robot_joints
    assert event.func is RaceAdditiveJointReset and event.mode == "reset"
    assert event.params["offset_half_widths"] == expected_widths
    assert event.params["asset_cfg"].joint_names == list(JOINT_NAMES)
    assert event.params["asset_cfg"].preserve_order

    def plain(value):
        """配置函数比较使用导入路径，避免配置深拷贝影响比较。"""
        if callable(value):
            return f"{value.__module__}:{value.__qualname__}"
        if isinstance(value, dict):
            return {key: plain(child) for key, child in value.items()}
        if isinstance(value, (list, tuple)):
            return [plain(child) for child in value]
        return value

    restored = deepcopy(rough.to_dict())
    original = baseline.to_dict()
    for field in ("pos", "joint_pos"):
        restored["scene"]["robot"]["init_state"][field] = (
            original["scene"]["robot"]["init_state"][field]
        )
    restored["events"]["reset_base"]["params"]["pose_range"]["z"] = (
        original["events"]["reset_base"]["params"]["pose_range"]["z"]
    )
    restored["events"]["reset_robot_joints"] = (
        original["events"]["reset_robot_joints"]
    )
    assert plain(restored) == plain(original), "复杂地形出现初始化之外的修改"
    assert flat.scene.robot.init_state.pos == (0., 0., .36)
    assert flat.events.reset_base.params["pose_range"]["z"] == (-.01, .01)
    assert flat.events.reset_robot_joints.func is baseline.events.reset_robot_joints.func
    assert flat.events.reset_robot_joints.params == baseline.events.reset_robot_joints.params

    # 原生排列与事件选择排列都刻意打乱，覆盖 PhysX 重排及按名匹配。
    native_names = [JOINT_NAMES[i] for i in (10, 0, 11, 3, 6, 8, 1, 5, 9, 4, 2, 7)]
    selected_names = list(reversed(JOINT_NAMES))
    joint_ids = [native_names.index(name) for name in selected_names]
    num_envs = 65536
    default = torch.tensor([expected_angles[name] for name in native_names])
    urdf = ET.parse(rough.scene.robot.spawn.asset_path).getroot()
    limits = []
    for name in native_names:
        limit = urdf.find(f"joint[@name='{name}']/limit")
        lower, upper = float(limit.get("lower")), float(limit.get("upper"))
        center = (lower + upper) / 2
        half = (upper - lower) / 2 * rough.scene.robot.soft_joint_pos_limit_factor
        limits.append((center - half, center + half))
    data = SimpleNamespace(
        default_joint_pos=default.repeat(num_envs, 1),
        soft_joint_pos_limits=torch.tensor(limits).repeat(num_envs, 1, 1),
        joint_pos=torch.full((num_envs, 12), 9.),
        joint_vel=torch.ones(num_envs, 12),
    )
    writes = []

    def write(positions, velocities, *, joint_ids, env_ids):
        rows = torch.as_tensor(env_ids)[:, None]
        data.joint_pos[rows, joint_ids] = positions
        data.joint_vel[rows, joint_ids] = velocities
        writes.append(len(env_ids))

    asset = SimpleNamespace(joint_names=native_names, data=data,
                            write_joint_state_to_sim=write)
    env = SimpleNamespace(scene={"robot": asset}, device="cpu", num_envs=num_envs)
    cfg = deepcopy(event)
    cfg.params["asset_cfg"] = SceneEntityCfg(
        "robot", joint_names=selected_names, joint_ids=joint_ids,
        preserve_order=True,
    )
    reset = RaceAdditiveJointReset(cfg, env)
    # 独立生成器状态隔离，统计核验不改变其他审计所用的随机序列。
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        reset(env, None, **cfg.params)
        widths = torch.tensor([expected_widths[name] for name in native_names])
        normalized = (data.joint_pos - data.default_joint_pos) / widths
        assert normalized.abs().max() <= 1.00001
        assert normalized.mean(dim=0).abs().max() < .012
        assert (normalized.var(dim=0) - 1 / 3).abs().max() < .006
        histogram_error = max(
            float((torch.histc(normalized[:, i], 10, -1., 1.) / num_envs - .1)
                  .abs().max()) for i in range(12)
        )
        assert histogram_error < .006
        correlation = torch.corrcoef(normalized.T) - torch.eye(12)
        assert correlation.abs().max() < .018
        assert torch.count_nonzero(data.joint_vel) == 0
        before = data.joint_pos.clone()
        subset = torch.tensor([3, 7, 100])
        reset(env, subset, **cfg.params)
        changed = (data.joint_pos != before).any(dim=1).nonzero().flatten()
        assert torch.equal(changed, subset)
        reset(env, slice(2, 8, 2), **cfg.params)
        assert writes[-1] == 3
        count = len(writes)
        reset(env, torch.empty(0, dtype=torch.long), **cfg.params)
        assert len(writes) == count
    assert torch.equal(data.default_joint_pos, default.repeat(num_envs, 1))
    invalid = deepcopy(cfg)
    invalid.params["offset_half_widths"][JOINT_NAMES[0]] = 10.
    try:
        RaceAdditiveJointReset(invalid, env)
    except ValueError:
        pass
    else:
        raise AssertionError("越限采样没有被拒绝")
    return {
        "passed": True,
        "default_joint_pos": expected_angles,
        "offset_half_widths_rad": expected_widths,
        "release_height_range_m": [.39, .55],
        "height_reward_target_m": rough.rewards.base_height_l2.params["target_height"],
        "sample_count": num_envs,
        "maximum_normalized_mean_abs": normalized.mean(dim=0).abs().max().item(),
        "maximum_histogram_bin_error": histogram_error,
        "maximum_joint_correlation_abs": correlation.abs().max().item(),
        "joint_reordering_and_subset_reset_verified": True,
        "sampling_within_soft_limits": True,
        "other_rough_configuration_unchanged": True,
        "flat_reset_unchanged": True,
    }


if __name__ == "__main__":
    import argparse
    import h5py  # noqa: F401
    import tensordict  # noqa: F401
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("logs/race_rough_reset/sampling_check.json"))
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    launcher = AppLauncher(args)
    try:
        from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
        import robot_lab.tasks  # noqa: F401

        configs = [load_cfg_from_registry(task, "env_cfg_entry_point") for task in (
            "RobotLab-Race-v0", "RobotLab-Race-Go2Reference-v0", "RobotLab-Race-Flat-v0",
        )]
        report = run_checks(*configs)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(json.dumps(report, ensure_ascii=False, indent=2))
    finally:
        launcher.app.close()
