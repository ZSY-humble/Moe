"""Race 复杂地形的逐关节加法重置，不改变作者的其他随机化。"""

from math import isfinite

import torch

from isaaclab.managers import ManagerTermBase


class RaceAdditiveJointReset(ManagerTermBase):
    """按真实关节名缓存幅度，各环境、各关节独立均匀采样。"""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        asset_cfg = cfg.params["asset_cfg"]
        asset = env.scene[asset_cfg.name]
        ids = asset_cfg.joint_ids
        names = (asset.joint_names[ids] if isinstance(ids, slice)
                 else [asset.joint_names[i] for i in ids])
        widths = cfg.params["offset_half_widths"]
        if len(names) != 12 or len(set(names)) != 12 or set(names) != set(widths):
            raise ValueError("Race 加法重置须完整指定十二个关节的独立幅度")
        values = [float(widths[name]) for name in names]
        if not all(isfinite(value) and value >= 0 for value in values):
            raise ValueError("关节重置幅度必须是有限非负数")
        self._half_widths = torch.tensor(
            values, device=env.device,
            dtype=asset.data.default_joint_pos.dtype,
        )
        default = asset.data.default_joint_pos[:, ids]
        limits = asset.data.soft_joint_pos_limits[:, ids]
        # 越限时显式报错，避免裁剪后破坏用户指定的均匀分布。
        if ((default - self._half_widths < limits[..., 0]).any()
                or (default + self._half_widths > limits[..., 1]).any()):
            raise ValueError("Race 重置采样范围超出软限位，请缩小随机幅度")

    def __call__(self, env, env_ids, offset_half_widths, asset_cfg):
        """仅写入本次重置的环境；关节初始速度沿用原设置零速。"""
        asset = env.scene[asset_cfg.name]
        if env_ids is None or isinstance(env_ids, slice):
            selection = slice(None) if env_ids is None else env_ids
            env_ids = torch.arange(env.num_envs, device=env.device)[selection]
        if len(env_ids) == 0:
            return
        positions = asset.data.default_joint_pos[env_ids][:, asset_cfg.joint_ids]
        offsets = (2.0 * torch.rand_like(positions) - 1.0) * self._half_widths
        asset.write_joint_state_to_sim(
            positions + offsets, torch.zeros_like(positions),
            joint_ids=asset_cfg.joint_ids, env_ids=env_ids,
        )
