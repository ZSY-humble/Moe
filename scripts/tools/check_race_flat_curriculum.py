"""确定性核验全向采样、分方向达标、失败计分和课程恢复。"""

from copy import deepcopy
from types import SimpleNamespace

import torch


def run_checks(env_cfg):
    """用实际命令类配合受控机身速度，逐一触发升级与拒绝路径。"""
    from robot_lab.tasks.race.commands import (
        GROUP_NAMES, RANGE_NAMES, RaceFlatOmniCommand,
    )

    class CommandProbe(RaceFlatOmniCommand):
        def _init_terrain_infos(self):
            # 唯一替身是地形查表，避免为了命令单元验证启动物理仿真。
            self.terrain_types = ["flat"]
            self.terrain_type2idx = {"flat": 0}
            self.terrain_idxs = torch.zeros(self.num_envs, dtype=torch.long)
            self.terrain_length = 8.

    count = 8192
    data = SimpleNamespace(root_lin_vel_b=torch.zeros(count, 3),
                           root_ang_vel_b=torch.zeros(count, 3))
    fake = SimpleNamespace(
        num_envs=count, device="cpu", step_dt=.02,
        common_step_counter=0, cfg=env_cfg,
        scene={"robot": SimpleNamespace(data=data)},
        termination_manager=SimpleNamespace(
            terminated=torch.zeros(count, dtype=torch.bool)),
    )
    command = CommandProbe(deepcopy(env_cfg.commands.base_velocity), fake)
    ids = torch.arange(count)
    torch.manual_seed(20261016)
    command._resample(ids)
    buckets, amounts = command.groups.unique(return_counts=True)
    assert set(buckets.tolist()) == {-1, *range(len(GROUP_NAMES))}
    assert (command.commands.abs() <= .500001).all()
    assert torch.all(command.commands[command.groups == -1] == 0.)
    for i in range(6):
        values = command.commands[command.groups == i]
        axis = i // 2
        other = [index for index in range(3) if index != axis]
        assert (values[:, other] == 0.).all()
        assert (values[:, axis].abs() >= .1).all()
        assert ((values[:, axis] < 0) == bool(i % 2)).all()
    assert (command.commands[command.groups >= 6].abs() >= .1).all()
    probe_values = command.commands[command.probes].abs()
    assert ((probe_values == 0) | (probe_values >= .4 - 1e-6)).all()

    def window(score=.9):
        command.window_counts.fill_(32)
        command.window_sums.fill_(32 * score)
        fake.common_step_counter += command.check_steps

    # 缺一组、任一方向差、组合转向差，都不能被其他高分掩盖。
    window()
    command.window_counts[-1] = 0
    command._maybe_advance()
    assert command.level == 0
    window()
    command.window_sums[3, 0] = 32 * .84
    command._maybe_advance()
    assert command.level == 0
    window()
    command.window_sums[-1, 1] = 32 * .84
    command._maybe_advance()
    assert command.level == 0
    window(.85)
    command._maybe_advance()
    assert command.level == 1
    assert all(command.command_ranges[key] == [-.6, .6] for key in RANGE_NAMES)

    # 旧级别尚未完成的命令不能帮新级别再次升级。
    command._finish_segments(ids)
    assert command.window_counts.sum() == 0
    for _ in range(15):
        window()
        command._maybe_advance()
    assert command.level == 10
    assert all(command.command_ranges[key] == [-1.5, 1.5]
               for key in RANGE_NAMES)
    assert all(command.env_command_ranges[key][:, 1].eq(1.5).all()
               for key in RANGE_NAMES)

    # 级别及范围须能精确恢复，非法范围必须拒绝。
    saved = command.curriculum_state()
    restored = CommandProbe(deepcopy(env_cfg.commands.base_velocity), fake)
    restored.load_curriculum_state(saved)
    assert restored.curriculum_state() == saved
    bad = deepcopy(saved)
    bad["ranges"]["lin_vel_y"] = [-1.6, 1.6]
    try:
        restored.load_curriculum_state(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("没有拒绝超过上限的课程状态")

    # 静止在0.4的近上限命令下得分不足85%，不能靠不动骗升级。
    command = CommandProbe(deepcopy(env_cfg.commands.base_velocity), fake)
    command.commands[:] = torch.tensor([.4, 0., 0.])
    command.groups[:] = 0
    command.probes[:] = True
    command.just_reset[:] = False
    command.segment_steps[:] = command.settle_steps
    command._update_metrics()
    stationary_score = command.segment_sums[0, 0].item()
    assert stationary_score < .85
    # 完美跟踪样本得分为1；提前失败仅存活约1秒不能得到高分。
    data.root_lin_vel_b[:, 0] = .4
    command.segment_sums.zero_()
    command.score_steps.zero_()
    command._update_metrics()
    assert torch.allclose(command.segment_sums, torch.ones(count, 2))
    command.segment_sums[:] = 25.
    command.score_steps[:] = 25
    command._finish_segments(ids, failed=True)
    failure_score = float((command.window_sums[0]
                           / command.window_counts[0]).min())
    assert failure_score < .85
    # 浮点计时可能让命令多保持一帧，失败得分也不能超过1。
    command.window_sums.zero_()
    command.window_counts.zero_()
    command.score_steps[:] = command.planned_score_steps + 1
    command.segment_sums[:] = command.planned_score_steps + 1
    command._finish_segments(ids, failed=True)
    assert (command.window_sums[0] / command.window_counts[0]).max() <= 1.

    # 环境重置必须只收集一次，且不得把新出生状态当作一次成功跟踪。
    command.window_sums.zero_()
    command.window_counts.zero_()
    command.groups[:] = 0
    command.sample_levels[:] = command.level
    command.probes[:] = True
    command.score_steps[:] = 25
    command.segment_sums[:] = 25.
    fake.termination_manager.terminated[:] = True
    command.reset(ids)
    assert command.window_counts[0] == count
    command._update_metrics()
    assert not command.score_steps.any()
    return {
        "passed": True, "sampled_commands": count,
        "graded_direction_groups": list(GROUP_NAMES),
        "sampling_group_counts": dict(zip(buckets.tolist(), amounts.tolist())),
        "missing_direction_prevents_upgrade": True,
        "one_bad_direction_prevents_upgrade": True,
        "mixed_yaw_separately_required": True,
        "threshold_inclusive": True,
        "synchronous_increment": .1, "maximum_level": 10,
        "lateral_limit_reaches_1_5": True,
        "old_level_samples_excluded": True,
        "zero_command_excluded": True,
        "stationary_probe_score": stationary_score,
        "early_failure_score": failure_score,
        "failure_score_bounded_by_one": True,
        "reset_collection_exactly_once": True,
        "curriculum_state_roundtrip": True,
        "invalid_state_rejected": True,
    }
