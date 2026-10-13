"""复用原 Go2 的奖励函数，由 Race 配置映射对应的关节和足端。"""

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
