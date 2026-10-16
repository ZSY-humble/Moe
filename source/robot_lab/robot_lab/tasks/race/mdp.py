"""复用原 Go2 奖励，提供 SLR 姿态与可选 Race 脚底净空奖励。"""

from math import isfinite
from pathlib import Path
import xml.etree.ElementTree as ET

import torch

from isaaclab.managers import ManagerTermBase
from isaaclab.utils.math import quat_apply

from isaaclab.envs.mdp import *  # noqa: F401, F403
from isaaclab_tasks.manager_based.locomotion.velocity.mdp import *  # noqa: F401, F403

# 原任务中的这些函数仅操作解析后的实体索引，不含 Go2 关节名称。
from robot_lab.tasks.go2.mdp.events import randomize_action_joint_pos_offset
from robot_lab.tasks.go2.mdp.observations import foot_contact_force_norm, joint_acc
from robot_lab.tasks.go2.mdp.rewards import (
    _get_base_height,
    action_smoothness_l2,
    base_height_l2,
    feet_regulation,
    hip_pos_penalty_l1,
    joint_pos_penalty_l1,
    joint_power,
    track_ang_vel_z_exp,
    track_lin_vel_xy_exp,
)


class RaceSoleFeetRegulation(ManagerTermBase):
    """可选的 Race 足部项：用 URDF 碰撞球最低点计算脚底净空。"""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        asset_cfg = cfg.params["asset_cfg"]
        asset = env.scene[asset_cfg.name]
        ids = asset_cfg.body_ids
        names = (asset.body_names[ids] if isinstance(ids, slice)
                 else [asset.body_names[i] for i in ids])
        expected = {f"pitch03_leg_{leg}_link"
                    for leg in ("f_l", "f_r", "b_l", "b_r")}
        if len(names) != 4 or set(names) != expected:
            raise ValueError("Race 脚底奖励必须选择四个足端连杆")
        target = cfg.params["base_height_target"]
        if not isfinite(target) or target <= 0:
            raise ValueError("脚底奖励的机身高度参考必须为有限正数")
        urdf = ET.parse(Path(asset.cfg.spawn.asset_path)).getroot()
        offsets, radii = [], []
        for name in names:
            link = urdf.find(f"link[@name='{name}']")
            collisions = [] if link is None else link.findall("collision")
            sphere = (collisions[0].find("geometry/sphere")
                      if len(collisions) == 1 else None)
            if sphere is None:
                raise ValueError(f"{name} 必须具有唯一球形足底碰撞体")
            origin = collisions[0].find("origin")
            xyz = ("0 0 0" if origin is None
                   else origin.get("xyz", "0 0 0"))
            offset = [float(value) for value in xyz.split()]
            radius = float(sphere.get("radius"))
            if (len(offset) != 3 or not all(map(isfinite, offset))
                    or not isfinite(radius) or radius <= 0):
                raise ValueError(f"{name} 的足底碰撞几何参数无效")
            offsets.append(offset)
            radii.append(radius)
        dtype = asset.data.body_pos_w.dtype
        self._offsets = torch.tensor(offsets, device=env.device,
                                     dtype=dtype).unsqueeze(0)
        self._radii = torch.tensor(radii, device=env.device,
                                   dtype=dtype).view(1, 4, 1)
        gravity = torch.as_tensor(env.sim.cfg.gravity, device=env.device,
                                  dtype=dtype)
        if not torch.isfinite(gravity).all() or gravity.norm() <= 0:
            raise ValueError("脚底奖励要求有限且非零的重力")
        self._down = (gravity / gravity.norm()).view(1, 1, 3)

    def _sole_heights(self, env, base_height_target, asset_cfg, sensor_cfg):
        """包含碰撞球偏移、连杆转动和球半径；接触穿透按零净空处理。"""
        asset = env.scene[asset_cfg.name]
        positions = asset.data.body_pos_w[:, asset_cfg.body_ids]
        quaternions = asset.data.body_quat_w[:, asset_cfg.body_ids]
        centers = positions + quat_apply(
            quaternions, self._offsets.expand_as(positions),
        )
        soles = centers + self._down * self._radii
        base_height = _get_base_height(
            env, base_height_target, asset_cfg, sensor_cfg,
        )
        delta = soles - asset.data.root_pos_w.unsqueeze(1)
        base_to_sole = (delta * self._down).sum(dim=-1)
        return (base_height.unsqueeze(1) - base_to_sole).clamp(min=0.)

    def __call__(self, env, base_height_target, asset_cfg, sensor_cfg=None):
        """沿用原水平足速和指数形式，仅将高度替换为真实脚底净空。"""
        asset = env.scene[asset_cfg.name]
        heights = self._sole_heights(
            env, base_height_target, asset_cfg, sensor_cfg,
        )
        feet_xy_vel_w = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
        return (feet_xy_vel_w.square().sum(dim=-1)
                * torch.exp(-heights / (.025 * base_height_target))).sum(dim=-1)


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
