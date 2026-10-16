"""Race 平地全向命令：各方向都达标后同步扩大速度范围。"""

from math import isfinite

import torch

from isaaclab.utils import configclass

from robot_lab.tasks.go2.mdp.commands import (
    Go2RLGymCommand, Go2RLGymCommandCfg,
)


GROUP_NAMES = (
    "forward", "backward", "left", "right", "yaw_left", "yaw_right",
    *(f"mixed_{i:03b}" for i in range(8)),
)
RANGE_NAMES = ("lin_vel_x", "lin_vel_y", "ang_vel_yaw")


class RaceFlatOmniCommand(Go2RLGymCommand):
    """沿用原命令接口及地形记录，只替换平地采样和速度课程。"""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if self.terrain_types != ["flat"]:
            raise ValueError("达标全向速度课程只用于平地任务")
        if cfg.command_range_curriculum:
            raise ValueError("达标课程不能同时启用固定轮次扩速")
        values = (cfg.curriculum_threshold, cfg.curriculum_step,
                  cfg.maximum_velocity, cfg.minimum_motion_command,
                  cfg.probe_fraction, cfg.probe_probability,
                  cfg.settling_time, cfg.resampling_time)
        if not all(isfinite(v) for v in values):
            raise ValueError("全向速度课程参数必须为有限数")
        if (not 0 < cfg.curriculum_threshold <= 1
                or cfg.curriculum_step <= 0
                or not 0 < cfg.minimum_motion_command < cfg.maximum_velocity
                or not 0 < cfg.probe_fraction <= 1
                or not 0 < cfg.probe_probability <= 1
                or not 0 <= cfg.settling_time < cfg.resampling_time
                or cfg.minimum_segments_per_group < 1):
            raise ValueError("全向速度课程参数范围无效")
        probabilities = torch.tensor(cfg.sampling_probabilities,
                                     device=self.device)
        if (probabilities.shape != (5,) or not (probabilities > 0).all()
                or not torch.isclose(probabilities.sum(),
                                     probabilities.new_tensor(1.))):
            raise ValueError("站立、前后、横移、转向、组合的概率必须为正且和为1")
        self.probabilities = probabilities
        self.initial_limits = [self.command_ranges[name][1]
                               for name in RANGE_NAMES]
        for name, limit in zip(RANGE_NAMES, self.initial_limits):
            if (self.command_ranges[name] != [-limit, limit]
                    or not cfg.minimum_motion_command < limit
                    <= cfg.maximum_velocity):
                raise ValueError("全向课程初始范围必须对称且不超过最大范围")
        self.level = 0
        self.groups = torch.full((self.num_envs,), -1, dtype=torch.long,
                                 device=self.device)
        self.sample_levels = torch.zeros_like(self.groups)
        self.probes = torch.zeros(self.num_envs, dtype=torch.bool,
                                  device=self.device)
        self.just_reset = torch.ones_like(self.probes)
        self.segment_steps = torch.zeros_like(self.groups)
        self.score_steps = torch.zeros_like(self.groups)
        self.segment_sums = torch.zeros(self.num_envs, 2, device=self.device)
        self.window_sums = torch.zeros(len(GROUP_NAMES), 2, device=self.device)
        self.window_counts = torch.zeros(len(GROUP_NAMES), device=self.device)
        self.last_scores = torch.zeros_like(self.window_sums)
        self.last_check_step = 0
        self.check_steps = max(1, round(cfg.resampling_time / env.step_dt))
        self.settle_steps = round(cfg.settling_time / env.step_dt)
        self.planned_score_steps = max(1, self.check_steps - self.settle_steps)
        self.linear_std = env.cfg.rewards.track_lin_vel_xy_exp.params["std"]
        self.yaw_std = env.cfg.rewards.track_ang_vel_z_exp.params["std"]
        for name in ("max_command_x", "max_command_y", "max_command_yaw",
                     "command_level",
                     *("tracking_" + name for name in GROUP_NAMES)):
            self.metrics[name] = torch.zeros(self.num_envs, device=self.device)

    def _finish_segments(self, env_ids, failed=False):
        """失败回合按剩余命令时间记零分，不能只统计幸存样本。"""
        current = (self.sample_levels[env_ids] == self.level)
        active = (self.groups[env_ids] >= 0) & self.probes[env_ids] & current
        ids = env_ids[active]
        if isinstance(failed, bool):
            failures = torch.full_like(ids, failed, dtype=torch.bool)
        else:
            failures = failed[active]
        observed = self.score_steps[ids]
        countable = (observed > 0) | failures
        ids, failures = ids[countable], failures[countable]
        denominator = torch.where(
            failures,
            self.score_steps[ids].clamp(min=self.planned_score_steps),
            self.score_steps[ids].clamp(min=1),
        )
        scores = self.segment_sums[ids] / denominator[:, None]
        groups = self.groups[ids]
        self.window_sums.index_add_(0, groups, scores)
        self.window_counts.index_add_(0, groups,
                                      torch.ones_like(groups, dtype=torch.float))

    def _maybe_advance(self):
        """统计窗口只规定观测长度；升级不依赖训练轮数。"""
        step = self._env.common_step_counter
        if step - self.last_check_step < self.check_steps:
            return
        if not (self.window_counts >=
                self.cfg.minimum_segments_per_group).all().item():
            return
        self.last_check_step = step
        self.last_scores[:] = self.window_sums / self.window_counts[:, None]
        passed = (self.last_scores >= self.cfg.curriculum_threshold).all().item()
        can_expand = any(self.command_ranges[name][1]
                         < self.cfg.maximum_velocity for name in RANGE_NAMES)
        if passed and can_expand:
            self.level += 1
            for name, initial in zip(RANGE_NAMES, self.initial_limits):
                limit = min(self.cfg.maximum_velocity,
                            round(initial + self.level * self.cfg.curriculum_step, 6))
                self.command_ranges[name] = [-limit, limit]
            self._update_env_command_ranges()
            self.max_lin_vel = max(self.command_ranges[name][1]
                                   for name in RANGE_NAMES[:2])
            print(f"[全向课程] 各方向跟踪均达标，升级到{self.level}级："
                  f"{self.command_ranges}", flush=True)
        # 升级后丢弃旧级别统计，未达标则收集下一批独立窗口。
        self.window_sums.zero_()
        self.window_counts.zero_()

    def _update_metrics(self):
        # IsaacLab 在物理步之后、复位之后更新命令；新回合首帧不能归给旧命令。
        active = (self.groups >= 0) & ~self.just_reset
        self.segment_steps += active
        settled = active & (self.segment_steps > self.settle_steps)
        linear_error = (self.commands[:, :2]
                        - self.robot.data.root_lin_vel_b[:, :2]).square().sum(1)
        yaw_error = (self.commands[:, 2]
                     - self.robot.data.root_ang_vel_b[:, 2]).square()
        scores = torch.stack((torch.exp(-linear_error / self.linear_std**2),
                              torch.exp(-yaw_error / self.yaw_std**2)), dim=1)
        self.segment_sums += scores * settled[:, None]
        self.score_steps += settled
        self.just_reset.zero_()
        self._maybe_advance()
        self._fill_metrics()

    def _fill_metrics(self):
        self.metrics["max_command_x"].fill_(self.command_ranges["lin_vel_x"][1])
        self.metrics["max_command_y"].fill_(self.command_ranges["lin_vel_y"][1])
        self.metrics["max_command_yaw"].fill_(
            self.command_ranges["ang_vel_yaw"][1])
        self.metrics["command_level"].fill_(self.level)
        for i, name in enumerate(GROUP_NAMES):
            self.metrics["tracking_" + name][:] = self.last_scores[i].min()

    def reset(self, env_ids=None):
        if env_ids is None or isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)[
                slice(None) if env_ids is None else env_ids]
        else:
            env_ids = torch.as_tensor(env_ids, device=self.device,
                                      dtype=torch.long)
        failed = self._env.termination_manager.terminated[env_ids]
        self._finish_segments(env_ids, failed)
        # 防止下方父类 reset 调用 _resample 时重复收集同一段。
        self.groups[env_ids] = -1
        self._maybe_advance()
        self._fill_metrics()
        extras = super().reset(env_ids)
        self.just_reset[env_ids] = True
        return extras

    def _resample(self, env_ids):
        if not len(env_ids):
            return
        self._finish_segments(env_ids)
        self._maybe_advance()
        count = len(env_ids)
        modes = torch.multinomial(self.probabilities, count, replacement=True)
        negative = torch.rand(count, 3, device=self.device) < .5
        probe = torch.rand(count, device=self.device) < self.cfg.probe_probability
        group = torch.full_like(modes, -1)
        mixed_sign = (negative.long() * negative.new_tensor(
            [4, 2, 1], dtype=torch.long)).sum(1)
        group = torch.where(modes == 4, 6 + mixed_sign, group)
        self.commands[env_ids] = 0.
        for column, name in enumerate(RANGE_NAMES):
            # 每种纯运动保留独立比例，组合运动同时激活前后、横移、转向。
            active = (modes == column + 1) | (modes == 4)
            limit = self.env_command_ranges[name][env_ids, 1]
            minimum = torch.where(probe, limit * self.cfg.probe_fraction,
                                  limit.new_full((count,),
                                                 self.cfg.minimum_motion_command))
            magnitude = minimum + torch.rand(count, device=self.device) * (
                limit - minimum)
            value = torch.where(negative[:, column], -magnitude, magnitude)
            self.commands[env_ids, column] = torch.where(active, value, 0.)
            group = torch.where(modes == column + 1,
                                2 * column + negative[:, column].long(), group)
        self.groups[env_ids] = group
        self.probes[env_ids] = probe & (modes != 0)
        self.sample_levels[env_ids] = self.level
        self.segment_steps[env_ids] = 0
        self.score_steps[env_ids] = 0
        self.segment_sums[env_ids] = 0.
        self.time_left[env_ids] = self.cfg.resampling_time
        self.command_counter[env_ids] += 1
        self.commands_xy_accumulation[env_ids] += self.commands[env_ids, :2]

    def curriculum_state(self):
        """只保存已达到的课程级别；未完成统计窗口不跨进程恢复。"""
        return {"version": 1, "level": self.level,
                "ranges": {name: list(self.command_ranges[name])
                           for name in RANGE_NAMES}}

    def load_curriculum_state(self, state):
        """恢复时核对级别与范围，防止续训默默退回初始速度。"""
        level = state.get("level")
        maximum = round((self.cfg.maximum_velocity - min(self.initial_limits))
                        / self.cfg.curriculum_step)
        if (state.get("version") != 1 or type(level) is not int
                or not 0 <= level <= maximum):
            raise ValueError("平地全向课程检查点格式或级别无效")
        expected = {}
        for name, initial in zip(RANGE_NAMES, self.initial_limits):
            limit = min(self.cfg.maximum_velocity,
                        round(initial + level * self.cfg.curriculum_step, 6))
            expected[name] = [-limit, limit]
        if state.get("ranges") != expected:
            raise ValueError("平地全向课程检查点范围与级别不一致")
        self.level = level
        self.command_ranges.update(expected)
        self._update_env_command_ranges()
        self.max_lin_vel = max(expected[name][1] for name in RANGE_NAMES[:2])
        self.groups.fill_(-1)
        self.window_sums.zero_()
        self.window_counts.zero_()
        self.last_scores.zero_()
        self.last_check_step = self._env.common_step_counter
        self._resample(torch.arange(self.num_envs, device=self.device))
        self.just_reset.fill_(True)
        self._fill_metrics()


@configclass
class RaceFlatOmniCommandCfg(Go2RLGymCommandCfg):
    """线速度以 m/s、转向以 rad/s 计，全部达标后同步增加0.1。"""

    class_type: type = RaceFlatOmniCommand
    command_range_curriculum: list = []
    dynamic_resample_commands: bool = False
    zero_command_curriculum: dict | None = None
    limit_vel_prob: float = 0.
    sampling_probabilities: tuple = (.10, .20, .20, .20, .30)
    minimum_motion_command: float = .1
    probe_probability: float = .5
    probe_fraction: float = .8
    curriculum_threshold: float = .85
    curriculum_step: float = .1
    maximum_velocity: float = 1.5
    minimum_segments_per_group: int = 32
    settling_time: float = .5
