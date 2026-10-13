"""Race 的显式 PD 执行器，与已有 MuJoCo 初始姿态控制保持相同次序。"""

from __future__ import annotations

import torch

from isaaclab.actuators import IdealPDActuator, IdealPDActuatorCfg
from isaaclab.utils import configclass
from isaaclab.utils.types import ArticulationActions


class RaceActuator(IdealPDActuator):
    """先限制电机 PD 力矩，再加入被动阻尼及库仑摩擦。"""

    cfg: RaceActuatorCfg

    def __init__(self, cfg: RaceActuatorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self.passive_damping = self._parse_joint_parameter(
            cfg.passive_damping, 0.0
        )
        self.passive_friction = self._parse_joint_parameter(
            cfg.passive_friction, 0.0
        )

    def compute(
        self,
        control_action: ArticulationActions,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> ArticulationActions:
        # 父类只对电机 PD 及前馈力矩限幅，不把被动项算入电机限幅。
        control_action = super().compute(control_action, joint_pos, joint_vel)
        friction_direction = torch.where(
            joint_vel.abs() > self.cfg.friction_velocity_threshold,
            torch.sign(joint_vel),
            torch.zeros_like(joint_vel),
        )
        passive_effort = (
            -self.passive_damping * joint_vel
            - self.passive_friction * friction_direction
        )
        # 两个缓冲都加入被动项，限幅残差仍只反映电机力矩饱和。
        self.computed_effort = self.computed_effort + passive_effort
        self.applied_effort = self.applied_effort + passive_effort
        control_action.joint_efforts = self.applied_effort
        return control_action


@configclass
class RaceActuatorCfg(IdealPDActuatorCfg):
    """Race 原始电机参数及独立被动阻尼、摩擦参数。"""

    class_type: type = RaceActuator
    passive_damping: dict[str, float] | float = 0.0
    passive_friction: dict[str, float] | float = 0.0
    friction_velocity_threshold: float = 1.0e-6

    # 所有被动项由上面的执行器显式计算，PhysX 不再重复施加。
    friction: dict[str, float] | float = 0.0
    dynamic_friction: dict[str, float] | float = 0.0
    viscous_friction: dict[str, float] | float = 0.0
