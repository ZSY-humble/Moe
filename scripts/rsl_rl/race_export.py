"""导出 Race 当前训练的部署合同，避免误用 Go2 或旧 Race 参数。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OBSERVATION_ORDER = [
    "base_ang_vel", "projected_gravity", "velocity_commands",
    "joint_pos", "joint_vel", "actions",
]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_deployment_manifest(env, policy, output, *, task, checkpoint):
    """从本次运行解析后的关节顺序与配置提取实际推理接口。"""
    from robot_lab.assets.race import RACE_CONTROL, JOINT_NAMES

    output = Path(output)
    action = env.action_manager.get_term("joint_pos")
    if action._joint_names != JOINT_NAMES:
        raise ValueError("Race 动作关节顺序与部署标准顺序不一致")
    for group in ("policy", "single_obs"):
        if env.observation_manager.active_terms[group] != OBSERVATION_ORDER:
            raise ValueError(f"Race {group} 观测项顺序不符合部署接口")
    if int(policy.num_single_obs) != 45 or int(policy.num_actor_obs) != 450:
        raise ValueError("当前 Race 部署接口要求 45 维单帧与 10 帧历史")
    observation_scales = {}
    observation_clips = []
    for name in OBSERVATION_ORDER:
        term = getattr(env.cfg.observations.single_obs, name)
        observation_scales[name] = (
            1.0 if term.scale is None else float(term.scale)
        )
        observation_clips.append(list(term.clip))
    if any(clip != [-100., 100.] for clip in observation_clips):
        raise ValueError("Race 单帧观测裁剪范围与部署接口不一致")
    if not isinstance(action.cfg.scale, (float, int)):
        raise ValueError("Race 部署接口需要统一的标量动作缩放")
    if not np.isclose(env.physics_dt, RACE_CONTROL["pd_dt"]):
        raise ValueError("Race 训练物理周期必须等于 5 ms PD 更新周期")
    robot = env.scene["robot"]
    defaults = robot.data.default_joint_pos[0, action._joint_ids]
    defaults = defaults.detach().cpu().numpy()
    if not np.allclose(defaults, RACE_CONTROL["default_angles"]):
        raise ValueError("训练初始目标角与 Race 名义控制参数不一致")
    asset = env.cfg.scene.robot
    actuator = asset.actuators["legs"]
    manifest = {
        "schema_version": 1, "robot": "race", "task": task,
        "checkpoint": str(Path(checkpoint).resolve()),
        "joint_names": JOINT_NAMES, "default_angles": defaults.tolist(),
        "base_init_pos": list(asset.init_state.pos),
        "base_init_quat": list(asset.init_state.rot),
        "action_scale": float(action.cfg.scale),
        "action_clip": [-100., 100.],
        "target_joint_limits": action._clip[0].detach().cpu().tolist(),
        "observation_order": OBSERVATION_ORDER,
        "observation_scales": observation_scales,
        "observation_clip": observation_clips[0],
        "num_single_obs": int(policy.num_single_obs),
        "history_length": int(policy.num_actor_obs // policy.num_single_obs),
        "history_initialization": "repeat_first_frame",
        "policy_dt": float(env.step_dt),
        "pd_dt": float(env.physics_dt),
        "physics_dt": float(RACE_CONTROL["physics_dt"]),
        "urdf_sha256": sha256(ROOT / "resources/race/urdf/race.urdf"),
        "mujoco_sha256": sha256(ROOT / "resources/race/mjcf/race_pd.xml"),
        "policy_sha256": sha256(output / "policy.pt"),
    }
    for key, values in (
        ("kps", actuator.stiffness), ("kds", actuator.damping),
        ("torque_limits", actuator.effort_limit),
        ("passive_damping", actuator.passive_damping),
        ("passive_friction", actuator.passive_friction),
        ("armature", actuator.armature),
    ):
        manifest[key] = [float(values[name]) for name in JOINT_NAMES]
        if not np.allclose(manifest[key], RACE_CONTROL[key]):
            raise ValueError(f"导出时 {key} 与 Race 控制配置不一致")
    path = output / "deployment.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    print(f"[INFO] 已导出 Race 当前训练部署合同：{path}")
    return path
