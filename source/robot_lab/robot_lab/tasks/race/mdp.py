"""复用原 Go2 奖励，并提供用户指定的 SLR 默认关节姿态奖励。"""

from math import isfinite

import torch

from isaaclab.managers import ManagerTermBase

from isaaclab.envs.mdp import *  # noqa: F401, F403
from isaaclab_tasks.manager_based.locomotion.velocity.mdp import *  # noqa: F401, F403

# 原任务中的这些函数仅操作解析后的实体索引，不含 Go2 关节名称。
from robot_lab.tasks.go2.mdp.events import randomize_action_joint_pos_offset
from robot_lab.tasks.go2.mdp.observations import foot_contact_force_norm, joint_acc
from robot_lab.tasks.go2.mdp.rewards import (
    action_smoothness_l2,
    base_height_l2,
    feet_regulation,
    hip_pos_penalty_l1,
    joint_pos_penalty_l1,
    joint_power,
    track_ang_vel_z_exp,
    track_lin_vel_xy_exp,
)


class SLRPoseReward(ManagerTermBase):
    """按关节名缓存 SLR 姿态容许偏差，兼容仿真中的关节重排。"""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        asset_cfg = cfg.params["asset_cfg"]
        asset = env.scene[asset_cfg.name]
        joint_ids = asset_cfg.joint_ids
        names = (asset.joint_names[joint_ids] if isinstance(joint_ids, slice)
                 else [asset.joint_names[i] for i in joint_ids])
        for mode in ("standing", "moving"):
            tolerances = cfg.params[mode + "_std"]
            values = []
            for name in names:
                matches = [value for prefix, value in tolerances.items()
                           if name.startswith(prefix)]
                if len(matches) != 1:
                    raise ValueError(f"姿态容许偏差匹配不唯一：{name}")
                value = float(matches[0])
                if not isfinite(value) or value <= 0:
                    raise ValueError("姿态容许偏差必须是有限正数")
                values.append(value)
            setattr(self, mode + "_std", torch.as_tensor(
                values, device=env.device, dtype=asset.data.joint_pos.dtype,
            ))
        threshold = cfg.params["command_threshold"]
        if not isfinite(threshold) or threshold < 0:
            raise ValueError("姿态命令阈值必须是有限非负数")

    def __call__(self, env, command_name, standing_std, moving_std,
                 command_threshold, asset_cfg):
        """完全沿用 SLR 核：指数中的误差对关节取均值，不求和。"""
        asset = env.scene[asset_cfg.name]
        command = env.command_manager.get_command(command_name)
        size = command[:, :2].norm(dim=1) + command[:, 2].abs()
        standing = size < command_threshold
        std = torch.where(standing[:, None], self.standing_std,
                          self.moving_std)
        error = (asset.data.joint_pos[:, asset_cfg.joint_ids]
                 - asset.data.default_joint_pos[:, asset_cfg.joint_ids]) / std
        return torch.exp(-error.square().mean(dim=1))
