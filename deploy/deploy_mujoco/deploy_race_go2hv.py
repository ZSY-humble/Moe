"""把原版 GO2HV 接口训练的 Race 权重部署到 MuJoCo。"""

import argparse
from collections import deque
from contextlib import nullcontext, redirect_stdout
import io
import json
import os
from pathlib import Path
from queue import SimpleQueue
import sys
import time
import xml.etree.ElementTree as ET

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/rsl_rl"))
sys.path.insert(0, str(ROOT / "scripts/tools"))

import mujoco
import mujoco.viewer
import numpy as np
import torch
import yaml
from tensordict import TensorDict
from rsl_rl.modules import ActorCriticMoECTS

from build_race_pd_model import build_model
from deploy_race import (
    KeyboardCommands, OBSERVATION_ORDER, OBSERVATION_SCALES,
    build_observation, ground_contacts, sha256, show_keyboard_status,
)
from race_checkpoint import RaceConfigLoader, _mapping_nodes, _saved_node
from utils import export_cts_policy_as_jit


def read_config(checkpoint):
    """只解析来源配置的必要节点，拒绝执行 YAML 对象标签。"""
    path = checkpoint.parent / "params/env.yaml"
    loader = RaceConfigLoader(path.read_text(encoding="utf-8"))
    root = loader.get_single_node()

    def get(*keys):
        return loader.construct_object(_saved_node(root, keys), deep=True)

    try:
        names = get("actions", "joint_pos", "joint_names")
        expected = [f"{kind}_leg_{leg}_joint"
                    for leg in ("f_l", "f_r", "b_l", "b_r")
                    for kind in ("roll", "pitch01", "pitch02")]
        if names != expected:
            raise ValueError("检查点不是标准 Race 12 关节动作顺序")

        def vector(value, default=None):
            if value is None:
                if default is None:
                    raise ValueError("必要的电机参数不能为 null")
                value = default
            values = ([value[name] for name in names]
                      if isinstance(value, dict) else [value] * 12)
            result = np.asarray(values, dtype=float)
            if result.shape != (12,) or not np.isfinite(result).all():
                raise ValueError("电机参数必须为 12 个有限数值")
            return result.tolist()

        actuator = get("scene", "robot", "actuators")
        if set(actuator) != {"GO2HV"}:
            raise ValueError("此入口用于 GO2HV；历史固定 PD 权重请用旧入口")
        actuator = actuator["GO2HV"]
        if actuator["class_type"] != (
                "robot_lab.assets.unitree_actuator:UnitreeActuator"):
            raise ValueError("不支持的训练执行器类型")
        cfg = {
            "joint_names": names,
            "default_angles": vector(get(
                "scene", "robot", "init_state", "joint_pos")),
            "base_init_pos": list(get(
                "scene", "robot", "init_state", "pos")),
            "base_init_quat": list(get(
                "scene", "robot", "init_state", "rot")),
            "gravity": list(get("sim", "gravity")),
            "pd_dt": float(get("sim", "dt")),
            "decimation": int(get("decimation")),
            "armature": vector(actuator["armature"], default=0.0),
            "kps": vector(actuator["stiffness"]),
            "kds": vector(actuator["damping"]),
            "torque_limits": vector(actuator["effort_limit_sim"]),
            "physx_joint_friction_coefficient": actuator["friction"],
            "delay_range": [actuator["min_delay"], actuator["max_delay"]],
        }
        for key in ("X1", "X2", "Y1", "Y2", "Fs", "Fd", "Va"):
            cfg[key] = vector(actuator[key])
        if np.any(np.asarray(cfg["X2"]) <= cfg["X1"]):
            raise ValueError("电机空载速度必须大于满力矩拐点")
        action = ("actions", "joint_pos")
        for key in ("preserve_order", "use_default_offset"):
            if get(*action, key) is not True:
                raise ValueError(f"动作配置 {key} 必须为 true")
        import re
        scale = get(*action, "scale")
        cfg["action_scale"] = []
        for name in names:
            matches = [v for pattern, v in scale.items()
                       if re.fullmatch(pattern, name)]
            if len(matches) != 1:
                raise ValueError(f"动作缩放匹配不唯一：{name}")
            cfg["action_scale"].append(float(matches[0]))
        if get(*action, "offset") != 0.0:
            raise ValueError("不支持额外的非零动作偏置")
        clip = get(*action, "clip")
        if set(clip) != {".*"}:
            raise ValueError("不支持当前目标角裁剪格式")
        cfg["target_clip"] = list(clip[".*"])
        for group, history in (("policy", 10), ("single_obs", 1)):
            entries = _mapping_nodes(
                _saved_node(root, ("observations", group)), group)
            terms = [name for name, node in entries.items()
                     if hasattr(node, "value") and isinstance(node, yaml.MappingNode)]
            if terms != OBSERVATION_ORDER:
                raise ValueError(f"{group} 观测项目顺序不一致")
            if get("observations", group, "history_length") != history:
                raise ValueError(f"{group} 观测历史长度不一致")
            for name in OBSERVATION_ORDER:
                prefix = ("observations", group, name)
                if float(get(*prefix, "scale")) != OBSERVATION_SCALES[name]:
                    raise ValueError(f"观测缩放不一致：{group}.{name}")
                if tuple(get(*prefix, "clip")) != (-100.0, 100.0):
                    raise ValueError(f"观测裁剪不一致：{group}.{name}")
        if get("scene", "robot", "spawn", "filter_base_hip_housings") is not True:
            raise ValueError("当前部署模型要求四对机身—髋壳过滤")
        if get("scene", "robot", "spawn", "articulation_props",
               "enabled_self_collisions") is not True:
            raise ValueError("当前部署模型要求训练自碰撞开启")
        filename = Path(get("scene", "robot", "spawn", "asset_path")).name
        if filename != "race_go2_single_box.urdf":
            raise ValueError("此入口仅支持当前单机身盒训练模型")
        cfg["urdf_path"] = str(ROOT / "resources/race/urdf" / filename)
        cfg["physics_dt"] = 0.001
        cfg["policy_dt"] = cfg["pd_dt"] * cfg["decimation"]
        if not np.isclose(cfg["policy_dt"], 0.02):
            raise ValueError("当前策略接口要求 20 ms 策略周期")
        cfg["command_ranges"] = get("commands", "base_velocity", "ranges")
        cfg["env_yaml_sha256"] = sha256(path)
        cfg["urdf_sha256"] = sha256(cfg["urdf_path"])
        return cfg
    finally:
        loader.dispose()


def export_policy(checkpoint, output):
    """离线严格加载权重，并逐帧比较学生策略与 TorchScript。"""
    config = yaml.load(
        (checkpoint.parent / "params/agent.yaml").read_text(),
        Loader=RaceConfigLoader,
    )
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    weights = saved["model_state_dict"]
    if any(not torch.isfinite(v).all().item() for v in weights.values()
           if torch.is_floating_point(v)):
        raise ValueError("检查点含非有限权重")
    critic_dim = weights["teacher_encoder.0.network.0.weight"].shape[1]
    obs = TensorDict({
        "policy": torch.zeros(1, 450),
        "critic": torch.zeros(1, critic_dim),
        "single_obs": torch.zeros(1, 45),
    }, batch_size=[1])
    policy_cfg = dict(config["policy"])
    if policy_cfg.pop("class_name") != "ActorCriticMoECTS":
        raise ValueError("检查点必须使用 CTS 策略网络")
    with redirect_stdout(io.StringIO()):
        native = ActorCriticMoECTS(
            obs, {"policy": ["policy"], "critic": ["critic"]},
            12, **policy_cfg,
        ).eval()
    native.load_state_dict(weights, strict=True)
    output.mkdir(parents=True, exist_ok=True)
    export_cts_policy_as_jit(
        native, native.actor_obs_normalizer, native.single_obs_normalizer,
        str(output),
    )
    exported = torch.jit.load(str(output / "policy.pt")).eval()
    rng = torch.Generator().manual_seed(42)
    maximum = 0.0
    with torch.inference_mode():
        for _ in range(3):
            exported.reset()
            frames = []
            for _ in range(40):
                frame = torch.randn(1, 45, generator=rng) * 0.3
                if not frames:
                    frames = [frame.clone() for _ in range(10)]
                    for _ in range(9):
                        exported(frame)
                else:
                    frames = frames[1:] + [frame]
                obs["single_obs"] = frame
                obs["policy"] = torch.cat([
                    torch.cat([x[:, lo:hi] for x in frames], dim=-1)
                    for lo, hi in ((0, 3), (3, 6), (6, 9), (9, 21),
                                   (21, 33), (33, 45))
                ], dim=-1)
                error = (native.act_inference(obs) - exported(frame)).abs().max()
                maximum = max(maximum, float(error))
    if maximum > 1e-5:
        raise RuntimeError(f"导出策略不一致：最大误差 {maximum}")
    exported.reset()
    print(f"[通过] 第{saved['iter']}轮学生策略，120帧导出误差 {maximum:.3g}")
    info = {"iteration": saved["iter"], "export_max_abs_error": maximum,
            "export_comparison_frames": 120}
    state = (saved.get("infos") or {}).get("race_command_curriculum")
    if state is not None:
        info["race_command_curriculum"] = state
    return exported, info


def prepare_model(cfg):
    """从训练 URDF 生成独立模型，精确过滤四个髋壳形状。"""
    path = ROOT / "resources/race/mjcf" / (
        "race_go2hv_" + cfg["env_yaml_sha256"][:12] + ".xml")
    _, metadata = build_model(
        output=path, timestep=cfg["physics_dt"],
        urdf_path=cfg["urdf_path"], control_cfg=cfg,
    )
    tree = ET.parse(path)
    root = tree.getroot()
    elements = [g for g in root.iter("geom")
                if g.get("group") == "1" or g.get("name") == "ground"]
    if len(elements) != 27:
        raise ValueError("必须包含26个机器人碰撞体和一个地面")
    all_bits = (1 << len(elements)) - 1
    indices = {g.get("name"): i for i, g in enumerate(elements)}
    base = indices["collision_base_link_0"]
    hips = [indices[f"collision_pitch01_leg_{leg}_link_1"]
            for leg in ("f_l", "f_r", "b_l", "b_r")]
    for index, geom in enumerate(elements):
        affinity = all_bits
        if index == base:
            for hip in hips:
                affinity &= ~(1 << hip)
        elif index in hips:
            affinity &= ~(1 << base)
        geom.set("contype", str(1 << index))
        geom.set("conaffinity", str(affinity))
    ET.indent(root, space="  ")
    tree.write(path, encoding="utf-8", xml_declaration=True)
    model = mujoco.MjModel.from_xml_path(str(path))
    # 位掩码只额外剔除这四对，不扩大到整条大腿。
    masked = []
    for i, first in enumerate(elements):
        for second in elements[i + 1:]:
            a, b = model.geom(first.get("name")).id, model.geom(second.get("name")).id
            allowed = ((model.geom_contype[a] & model.geom_conaffinity[b])
                       or (model.geom_contype[b] & model.geom_conaffinity[a]))
            if not allowed:
                masked.append([first.get("name"), second.get("name")])
    if len(masked) != 4:
        raise ValueError("几何接触过滤范围错误")
    joints = [model.joint(name).id for name in cfg["joint_names"]]
    qpos, qvel = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
    model.body_pos[model.body("base_link").id] = cfg["base_init_pos"]
    model.qpos0[:3] = cfg["base_init_pos"]
    model.qpos0[3:7] = cfg["base_init_quat"]
    for index, joint in enumerate(joints):
        model.dof_armature[qvel[index]] = cfg["armature"][index]
        model.dof_damping[qvel[index]] = 0.0
        model.dof_frictionloss[qvel[index]] = 0.0
    # PhysX 的关节摩擦系数不是 MuJoCo 的固定干摩擦力矩，不偷换单位。
    # 此处保留差异记录；不叠加旧固定 PD 部署的被动摩擦或转子惯量。
    cfg["mujoco_joint_frictionloss_nm"] = 0.0
    cfg["model_path"] = str(path)
    cfg["collision_filter_pairs"] = masked
    cfg["total_mass_kg"] = metadata["total_mass"]
    cfg["simtosim_limitations"] = [
        "MuJoCo接触、限位和摩擦求解与PhysX不同，不能宣称动力学数值等价。",
        "PhysX关节摩擦系数0.01未当作MuJoCo的0.01N·m；当前MuJoCo关节干摩擦为0。",
        "MuJoCo积分周期1ms，PD更新5ms；训练PhysX积分周期5ms。",
        "速度相关力矩曲线已实现；PhysX额外的关节速度硬限制未直接映射。",
    ]
    actuators = [model.actuator("motor_" + name).id
                 for name in cfg["joint_names"]]
    return model, qpos, qvel, actuators


class Go2HVMotor:
    """按训练原算法执行5ms延迟目标、PD和速度相关力矩限幅。"""

    def __init__(self, cfg, delay):
        self.cfg = cfg
        self.delay = delay
        self.history = deque(maxlen=delay + 1)

    def reset(self):
        self.history.clear()

    def compute(self, target, position, velocity):
        self.history.append(np.asarray(target).copy())
        delayed = self.history[max(0, len(self.history) - 1 - self.delay)]
        torque = (np.asarray(self.cfg["kps"]) * (delayed - position)
                  - np.asarray(self.cfg["kds"]) * velocity)
        limit = np.where(velocity * torque > 0, self.cfg["Y1"], self.cfg["Y2"])
        speed = np.abs(velocity)
        x1, x2 = np.asarray(self.cfg["X1"]), np.asarray(self.cfg["X2"])
        limit = np.where(speed < x1, limit,
                         np.maximum(0.0, limit * (x2 - speed) / (x2 - x1)))
        applied = np.clip(torque, -limit, limit)
        applied -= (np.asarray(self.cfg["Fs"]) * np.tanh(
            velocity / np.asarray(self.cfg["Va"]))
            + np.asarray(self.cfg["Fd"]) * velocity)
        return np.clip(applied, -np.asarray(self.cfg["torque_limits"]),
                       np.asarray(self.cfg["torque_limits"]))


def run(args):
    """执行学生策略闭环；窗口从零命令开始，R键重新释放。"""
    torch.set_num_threads(1)
    checkpoint = args.checkpoint.expanduser().resolve()
    cfg = read_config(checkpoint)
    output = checkpoint.parent / "exported/mujoco_go2hv"
    policy, export = export_policy(checkpoint, output)
    if "race_command_curriculum" in export:
        state = export["race_command_curriculum"]
        names = ("lin_vel_x", "lin_vel_y", "ang_vel_yaw")
        level = state.get("level")
        if (state.get("version") != 1 or type(level) is not int
                or not 0 <= level <= 10
                or state.get("ranges") != {
                    name: [-round(.5 + .1 * level, 6),
                           round(.5 + .1 * level, 6)] for name in names}):
            raise ValueError("检查点中的平地全向速度课程状态无效")
        cfg["command_ranges"] = state["ranges"]
    model, qpos, qvel, actuators = prepare_model(cfg)
    data = mujoco.MjData(model)
    base_id = model.body("base_link").id
    foot_ids = [model.body(f"pitch03_leg_{leg}_link").id
                for leg in ("f_l", "f_r", "b_l", "b_r")]
    nominal = np.asarray(cfg["default_angles"])
    rng = np.random.default_rng(args.seed)
    delay = (int(rng.integers(*[cfg["delay_range"][0],
                                cfg["delay_range"][1] + 1]))
             if args.delay_steps is None else args.delay_steps)
    if not cfg["delay_range"][0] <= delay <= cfg["delay_range"][1]:
        raise ValueError("延迟步数超出训练范围")
    motor = Go2HVMotor(cfg, delay)
    command = np.asarray(args.command, dtype=np.float32)
    limits = np.asarray([cfg["command_ranges"][name]
                         for name in ("lin_vel_x", "lin_vel_y", "ang_vel_yaw")])
    if np.any(command < limits[:, 0]) or np.any(command > limits[:, 1]):
        raise ValueError("测试命令超出当前训练初始阶段范围")
    control = KeyboardCommands([0., 0., 0.]) if args.keyboard else None
    if control:
        control.speed = 0.5
        command[:] = 0.0
    events = SimpleQueue()

    def key_callback(key):
        if key == ord("R"):
            events.put("reset")
        elif control:
            control.key_callback(key)

    action = np.zeros(12, dtype=np.float32)
    forces = deque(maxlen=3)

    def reset():
        mujoco.mj_resetData(model, data)
        data.qpos[:3] = cfg["base_init_pos"]
        data.qpos[3:7] = cfg["base_init_quat"]
        data.qpos[qpos] = nominal
        mujoco.mj_forward(model, data)
        motor.reset()
        forces.clear()
        action[:] = 0.0
        policy.reset()
        first = build_observation(model, data, base_id, qpos, qvel,
                                  nominal, action, command)
        for _ in range(9):
            policy(torch.from_numpy(first).unsqueeze(0))

    manifest = {"schema_version": 2, "robot": "race", "actuator": "GO2HV",
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256(checkpoint), **export,
                "control": cfg, "delay_steps": delay,
                "policy_sha256": sha256(output / "policy.pt"),
                "model_sha256": sha256(cfg["model_path"])}
    (output / "deployment.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[部署] 策略20ms，PD5ms，MuJoCo1ms，动作延迟{delay}个PD步")
    print("[部署] 训练单机身盒、四对髋壳过滤、训练站姿与电机曲线")
    print("[说明] MuJoCo与PhysX的接触、关节摩擦及限位求解存在差异。")
    if control:
        print("[键盘] 点击MuJoCo画面；W/S前后，A/D横移，Q/E转向；"
              "松键停止，空格停止，R复位。Z/X调整速度，限制在训练范围。")
    rows = []
    resets = 0
    elapsed = 0.0
    failure = None
    context = (mujoco.viewer.launch_passive(model, data, key_callback=key_callback)
               if args.keyboard else nullcontext(None))
    with torch.inference_mode(), context as viewer:
        reset()
        if viewer:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = base_id
            viewer.cam.distance = 1.8
            viewer.cam.azimuth = 135.0
            viewer.cam.elevation = -20.0
            viewer.opt.geomgroup[1] = 0
            viewer.opt.geomgroup[2] = 1
        deadline = time.monotonic()
        while (viewer is None or viewer.is_running()) and (
                args.duration == 0 or elapsed < args.duration):
            if not events.empty():
                while not events.empty():
                    events.get()
                reset()
                resets += 1
            if control:
                command = np.clip(control.read(), limits[:, 0], limits[:, 1])
            frame = build_observation(model, data, base_id, qpos, qvel,
                                      nominal, action, command)
            action[:] = policy(torch.from_numpy(frame).unsqueeze(0))[0].numpy()
            target = np.clip(nominal + action * cfg["action_scale"],
                             *cfg["target_clip"])
            for _ in range(cfg["decimation"]):
                data.ctrl[actuators] = motor.compute(
                    target, data.qpos[qpos], data.qvel[qvel])
                for _ in range(round(cfg["pd_dt"] / cfg["physics_dt"])):
                    mujoco.mj_step(model, data)
                # mj_step后的运动学数据需刷新到新状态，再构建下一帧观测。
                mujoco.mj_forward(model, data)
                force = np.zeros(3)
                for contact_id in range(data.ncon):
                    contact = data.contact[contact_id]
                    a, b = (int(model.geom_bodyid[g]) for g in contact.geom)
                    if base_id not in (a, b):
                        continue
                    wrench = np.zeros(6)
                    mujoco.mj_contactForce(model, data, contact_id, wrench)
                    world_force = contact.frame.reshape(3, 3).T @ wrench[:3]
                    force += world_force if b == base_id else -world_force
                forces.append(float(np.linalg.norm(force)))
            elapsed += cfg["policy_dt"]
            if not np.isfinite(data.qpos).all() or not np.isfinite(action).all():
                raise RuntimeError("部署状态或动作出现非有限值")
            rotation = data.xmat[base_id].reshape(3, 3)
            velocity = np.zeros(6)
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY,
                                    base_id, velocity, 0)
            local = rotation.T @ velocity[3:]
            angular = rotation.T @ velocity[:3]
            loads, nonfeet = ground_contacts(model, data, foot_ids)
            clearance = []
            for foot in foot_ids:
                geom = next(g for g in range(model.ngeom)
                            if model.geom_bodyid[g] == foot
                            and model.geom_group[g] == 1)
                clearance.append(float(data.geom_xpos[geom, 2]
                                       - model.geom_size[geom, 0]))
            row = {"time_s": elapsed, "episode_time_s": float(data.time),
                   "command": command.tolist(),
                   "velocity": [float(local[0]), float(local[1]), float(angular[2])],
                   "height_m": float(data.xpos[base_id, 2]),
                   "tilt_deg": float(np.degrees(np.arccos(np.clip(rotation[2, 2], -1, 1)))),
                   "base_contact_n": max(forces, default=0.),
                   "foot_loads_n": loads.tolist(),
                   "foot_clearance_m": clearance,
                   "nonfoot_ground_contacts": nonfeet,
                   "max_abs_action": float(np.abs(action).max())}
            rows.append(row)
            if viewer:
                if len(rows) % 10 == 0:
                    show_keyboard_status(viewer, model, data, base_id, command)
                    status = {"checkpoint": str(checkpoint), **row,
                              "resets": resets}
                    (ROOT / "logs/keyboard").mkdir(parents=True, exist_ok=True)
                    (ROOT / "logs/keyboard/mujoco_go2hv_state.json").write_text(
                        json.dumps(status, ensure_ascii=False, indent=2))
                viewer.sync()
                deadline += cfg["policy_dt"]
                wait = deadline - time.monotonic()
                if wait > 0:
                    time.sleep(wait)
                elif wait < -0.5:
                    deadline = time.monotonic()
            if row["base_contact_n"] > 1.0:
                failure = "机身净接触力超过1N"
            elif row["tilt_deg"] > 90:
                failure = "翻倒超过90度（部署诊断记录）"
            if failure:
                if viewer:
                    print(f"\n[复位] {failure}，回合{data.time:.2f}秒", flush=True)
                    reset()
                    resets += 1
                    failure = None
                else:
                    break
            if len(rows) > 10000 and viewer:
                rows = rows[-5000:]
    steady = [row for row in rows if row["time_s"] >= min(2., elapsed / 2)]
    report = {"checkpoint": str(checkpoint), **export,
              "duration_s": elapsed, "failure": failure, "resets": resets,
              "delay_steps": delay, "command": list(args.command),
              "mean_velocity": np.mean([r["velocity"] for r in steady], axis=0).tolist(),
              "mean_height_m": float(np.mean([r["height_m"] for r in steady])),
              "max_tilt_deg": max(r["tilt_deg"] for r in rows),
              "maximum_foot_clearance_m": np.max([r["foot_clearance_m"] for r in rows], axis=0).tolist(),
              "trace": rows}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print("\n" + json.dumps({k: v for k, v in report.items() if k != "trace"},
                            ensure_ascii=False, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--keyboard", action="store_true")
    parser.add_argument("--duration", type=float, default=0.)
    parser.add_argument("--command", type=float, nargs=3, default=[0., 0., 0.])
    parser.add_argument("--delay_steps", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.duration < 0 or not np.isfinite(args.duration):
        parser.error("时长必须为非负有限数")
    if not args.keyboard and args.duration == 0:
        parser.error("无窗口测试需要 --duration")
    if not np.isfinite(args.command).all():
        parser.error("速度命令必须为有限值")
    run(args)


if __name__ == "__main__":
    main()
