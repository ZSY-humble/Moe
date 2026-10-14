"""离线导出 Race 检查点，执行 MuJoCo 对照评估或键盘回放。"""

import argparse
from contextlib import nullcontext, redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
import math
import os
from pathlib import Path
from queue import SimpleQueue
import sys
import time
import xml.etree.ElementTree as ET

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "deploy/deploy_mujoco"))
sys.path.insert(0, str(ROOT / "scripts/rsl_rl"))

import mujoco
import mujoco.viewer
import numpy as np
import torch
import yaml
from tensordict import TensorDict
from rsl_rl.modules import ActorCriticMoECTS

from deploy_race import (
    OBSERVATION_ORDER, OBSERVATION_SCALES, KeyboardCommands,
    build_observation, show_keyboard_status, verify_manifest,
)
from stand_race import load_robot, step_pd
from utils import export_cts_policy_as_jit

CONFIG = ROOT / "deploy/deploy_mujoco/configs/race.yaml"
TERRAINS = ["flat", "slope_1", "slope_3", "stairs_1", "stairs_3",
            "wave_1", "wave_3", "obstacle_1", "obstacle_3"]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export_checkpoint(checkpoint):
    """严格加载完整网络，并与原生学生策略逐步核对导出结果。"""
    checkpoint = checkpoint.resolve()
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    weights = saved["model_state_dict"]
    if any(not torch.isfinite(value).all() for value in weights.values()):
        raise ValueError("检查点含非有限权重")
    observations = TensorDict({
        "policy": torch.zeros(1, 450),
        "critic": torch.zeros(1, 275),
        "single_obs": torch.zeros(1, 45),
    }, batch_size=[1])
    with redirect_stdout(io.StringIO()):
        policy = ActorCriticMoECTS(
            observations, {"policy": ["policy"], "critic": ["critic"]},
            12, actor_hidden_dims=[512, 256, 128],
            critic_hidden_dims=[512, 256, 128],
            student_encoder_hidden_dims=[512, 256, 256],
            teacher_encoder_hidden_dims=[512, 256], expert_num=8,
            latent_dim=32, activation="elu", norm_type="l2norm",
        )
    policy.load_state_dict(weights, strict=True)
    policy.eval()
    output = ROOT / "exported" / checkpoint.stem
    export_cts_policy_as_jit(policy, None, None, str(output))
    exported = torch.jit.load(str(output / "policy.pt")).eval()
    history = []
    generator = torch.Generator().manual_seed(42)
    maximum_error = 0.0
    with torch.inference_mode():
        for _ in range(100):
            frame = torch.randn(1, 45, generator=generator)
            if not history:
                history = [frame.clone() for _ in range(10)]
                for _ in range(9):
                    exported(frame)
            else:
                history = history[1:] + [frame]
            # IsaacLab 按观测项分别存储历史，再拼接各项。
            stacked = torch.cat([
                torch.cat([x[:, start:end] for x in history], dim=-1)
                for start, end in ((0, 3), (3, 6), (6, 9), (9, 21),
                                   (21, 33), (33, 45))
            ], dim=-1)
            observations["policy"] = stacked
            observations["single_obs"] = frame
            reference = policy.act_inference(observations)
            actual = exported(frame)
            maximum_error = max(maximum_error,
                                float((reference - actual).abs().max()))
    if maximum_error > 1e-5:
        raise RuntimeError(f"导出与原生学生策略不一致：{maximum_error}")
    exported.reset()
    exported.save(str(output / "policy.pt"))
    model, _, cfg, *_ = load_robot(CONFIG)
    joint_ids = [model.joint(name).id for name in cfg["joint_names"]]
    contract = {
        "schema_version": 1, "robot": "race",
        "task": ("RobotLab-Race-Legacy-Flat-v0" if saved["iter"] == 3400
                 else "RobotLab-Race-Legacy-v0"),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": digest(checkpoint),
        "num_single_obs": 45, "history_length": 10,
        "action_scale": 0.25, "policy_dt": 0.02,
        "history_initialization": "repeat_first_frame",
        "action_clip": [-100.0, 100.0],
        "observation_clip": [-100.0, 100.0],
        "observation_order": OBSERVATION_ORDER,
        "observation_scales": OBSERVATION_SCALES,
        "target_joint_limits": model.jnt_range[joint_ids].tolist(),
        "urdf_sha256": digest(ROOT / "resources/race/urdf/race.urdf"),
        "mujoco_sha256": digest(ROOT / "resources/race/mjcf/race_pd.xml"),
        "policy_sha256": digest(output / "policy.pt"),
        "training_config_status": (
            "未提供原始 env.yaml；用户确认全零关节和 Kp=24/40/32；"
            "其他接口按当前仓库 Race 配置评估"
        ),
        "export_comparison_frames": 100,
        "export_max_abs_error": maximum_error,
    }
    for name in ("joint_names", "default_angles", "base_init_pos",
                 "base_init_quat", "pd_dt", "physics_dt", "kps", "kds",
                 "torque_limits", "passive_damping", "passive_friction",
                 "armature"):
        contract[name] = cfg[name]
    (output / "deployment.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    verify_manifest(output / "policy.pt", cfg)
    print(f"导出 {checkpoint.name}，100 帧最大误差={maximum_error:.3g}",
          flush=True)
    return output / "policy.pt", contract


def make_scene(terrain, destination):
    """保留机器人参数，直接复用附带 RoboGauge 地形几何。"""
    original = ROOT / "resources/race/mjcf/race_pd.xml"
    if terrain == "flat":
        return original
    kind = terrain.rsplit("_", 1)[0]
    terrain_path = (ROOT / "third_party/robogauge/resources/terrains"
                    / kind / f"{terrain}.xml")
    root = ET.parse(original).getroot()
    for mesh in root.findall("./asset/mesh"):
        mesh.set("file", str((original.parent / mesh.get("file")).resolve()))
    terrain_root = ET.parse(terrain_path).getroot()
    for item in terrain_root.findall("./asset/hfield"):
        item = deepcopy(item)
        item.set("file", str((terrain_path.parent / item.get("file")).resolve()))
        root.find("asset").append(item)
    ground = root.find("./worldbody/geom[@name='ground']")
    for index, item in enumerate(terrain_root.findall("./worldbody/geom")):
        if item.get("type") == "plane":
            continue
        item = deepcopy(item)
        item.set("name", f"terrain_{index}")
        for field in ("contype", "conaffinity", "condim", "friction",
                      "solref", "solimp", "margin"):
            item.set(field, ground.get(field))
        root.find("worldbody").append(item)
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / f"{terrain}.xml"
    ET.ElementTree(root).write(path)
    return path


def run_episode(policy_path, scene, command, seconds, seed, *,
                keyboard=False, friction=1.0, trace_path=None):
    """按 20/5/1 ms 运行；速度在机身坐标中测量，倾斜超 60 度判跌倒。"""
    robot = load_robot(CONFIG, {"model_path": str(scene)})
    model, data, cfg, qpos, qvel, actuators = robot
    baseline_cfg = yaml.safe_load(CONFIG.read_text())
    verify_manifest(policy_path, baseline_cfg)
    nominal = np.array(cfg["default_angles"], dtype=float)
    joint_ids = [model.joint(name).id for name in cfg["joint_names"]]
    limits = model.jnt_range[joint_ids]
    base_id = model.body("base_link").id
    feet = {model.body(f"pitch03_leg_{leg}_link").id
            for leg in ("f_l", "f_r", "b_l", "b_r")}
    rng = np.random.default_rng(seed)
    # 种子 0 为完全名义参数；其余种子轻微扰动释放位置和关节角。
    if seed:
        data.qpos[:2] += rng.uniform(-0.015, 0.015, 2)
        data.qpos[qpos] += rng.uniform(-0.015, 0.015, 12)
    world_geoms = model.geom_bodyid == 0
    model.geom_friction[world_geoms, 0] = friction
    # 楼梯和波浪从 x=0 开始；先从平地接近，避免释放时跨在地形边缘。
    if scene.stem.startswith(("stairs_", "wave_")):
        data.qpos[0] -= 1.0
    initial_z = data.qpos[2]
    mujoco.mj_forward(model, data)
    # 地形释放高度按四脚下最高表面调整，减少初始穿透。
    heights = []
    for foot in feet:
        point = data.xpos[foot].copy()
        point[2] = 10.0
        distance = mujoco.mj_ray(
            model, data, point, np.array([0., 0., -1.]),
            np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8),
            1, base_id, np.zeros(1, dtype=np.int32),
        )
        if distance >= 0:
            # 射线只统计静态地形，避免打到机器人自身。
            heights.append(10.0 - distance)
    if heights:
        data.qpos[2] = initial_z + max(heights)
    mujoco.mj_forward(model, data)
    policy = torch.jit.load(str(policy_path)).eval()
    policy.reset()
    action = np.zeros(12, dtype=np.float32)
    start = data.qpos[:3].copy()
    control = KeyboardCommands([0., 0., 0.]) if keyboard else None
    keyboard_events = SimpleQueue()
    requested_change = None

    def key_callback(keycode):
        """窗口线程只投递事件，切换和复位由主线程执行。"""
        if keycode in (ord("1"), ord("2"), ord("R"), ord("T")):
            keyboard_events.put(chr(keycode))
        else:
            control.key_callback(keycode)
    if control:
        control.speed = 0.5
    first = build_observation(model, data, base_id, qpos, qvel, nominal,
                              action, np.zeros(3, dtype=np.float32))
    with torch.inference_mode():
        for _ in range(9):
            policy(torch.from_numpy(first).unsqueeze(0))
    rows = []
    failure = None
    clipped_steps = 0
    max_action = 0.0
    max_tilt = 0.0
    context = (mujoco.viewer.launch_passive(
        model, data, key_callback=key_callback,
    ) if keyboard else nullcontext(None))
    deadline = time.monotonic()
    with context as viewer, torch.inference_mode():
        if viewer:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = base_id
            viewer.cam.distance = 2.5
            viewer.cam.azimuth = 135.0
            viewer.cam.elevation = -22.0
            viewer.opt.geomgroup[1] = 0
            viewer.opt.geomgroup[2] = 1
            print("点击 MuJoCo 窗口后按住 WASD 平移，QE 转向；"
                  "Z/X 调速，空格停止；1/2 切模型，T 切地形，R 复位；"
                  "关闭窗口结束。", flush=True)
        while (keyboard or data.time < seconds - 1e-9):
            if viewer and not viewer.is_running():
                break
            if not keyboard_events.empty():
                requested_change = keyboard_events.get()
                break
            # 前 2 秒零命令站稳；测试误差只统计之后的运动阶段。
            target_command = (control.read() if control else np.array(
                [0., 0., 0.] if data.time < 2.0 else command,
                dtype=np.float32,
            ))
            obs = build_observation(model, data, base_id, qpos, qvel,
                                    nominal, action, target_command)
            output = policy(torch.from_numpy(obs).unsqueeze(0)).numpy()
            if output.shape != (1, 12) or not np.isfinite(output).all():
                raise RuntimeError("策略输出异常")
            action = np.clip(output[0], -100., 100.)
            raw_target = nominal + 0.25 * action
            target = np.clip(raw_target, limits[:, 0], limits[:, 1])
            clipped_steps += int(np.any(raw_target != target))
            max_action = max(max_action, float(abs(action).max()))
            cfg["default_angles"] = target
            for _ in range(4):
                step_pd(model, data, cfg, qpos, qvel, actuators)
            if not all(np.isfinite(x).all() for x in
                       (data.qpos, data.qvel, data.ctrl)):
                raise RuntimeError("闭环状态出现非有限数值")
            mujoco.mj_forward(model, data)
            rotation = data.xmat[base_id].reshape(3, 3)
            velocity = np.zeros(6)
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY,
                                    base_id, velocity, 0)
            linear = rotation.T @ velocity[3:]
            angular = rotation.T @ velocity[:3]
            tilt = math.degrees(math.acos(np.clip(rotation[2, 2], -1., 1.)))
            max_tilt = max(max_tilt, tilt)
            nonfoot = False
            for contact in data.contact:
                b1, b2 = model.geom_bodyid[contact.geom]
                if (b1 == 0 and b2 not in feet and b2 != 0) or (
                        b2 == 0 and b1 not in feet and b1 != 0):
                    nonfoot = True
            rows.append([float(data.time), *data.qpos[:3], *linear[:2],
                         angular[2], tilt, int(nonfoot), *target_command])
            if failure is None and tilt > 60.0:
                failure = float(data.time)
            if viewer:
                if len(rows) % 25 == 0:
                    show_keyboard_status(viewer, model, data,
                                         base_id, target_command)
                    if hasattr(viewer, "set_texts"):
                        viewer.set_texts((
                            mujoco.mjtFontScale.mjFONTSCALE_150,
                            mujoco.mjtGridPos.mjGRID_TOPLEFT,
                            f"Race {policy_path.parent.name} | {scene.stem}\n"
                            "1/2: model  T: terrain  R: reset\n"
                            "WASD: move  QE: yaw  Z/X: speed  SPACE: stop\n"
                            "Command vx / vy / yaw\nMeasured vx / vy / yaw",
                            "\nClick viewport to focus\n"
                            + f"Speed: {control.speed:.1f} m/s\n"
                            + " / ".join(f"{v:+.2f}" for v in target_command)
                            + "\n"
                            + " / ".join(f"{v:+.2f}" for v in
                                         (linear[0], linear[1], angular[2])),
                        ))
                viewer.sync()
                deadline += 0.02
                time.sleep(max(0., deadline - time.monotonic()))
                if deadline < time.monotonic() - 0.5:
                    deadline = time.monotonic()
            elif failure is not None:
                break
    if keyboard:
        print(flush=True)
        return {"keyboard_event": requested_change}
    values = np.asarray(rows)
    active = values[values[:, 0] >= 3.0]
    result = {
        "seed": seed, "friction": friction, "command": list(command),
        "duration_s": float(data.time), "requested_duration_s": seconds,
        "fell": failure is not None, "fall_time_s": failure,
        "max_tilt_deg": max_tilt,
        "displacement_m": (data.qpos[:3] - start).tolist(),
        "initial_position_m": start.tolist(),
        "final_height_m": float(data.qpos[2]),
        "min_height_m": float(values[:, 3].min()),
        "nonfoot_contact_fraction": float(values[:, 8].mean()),
        "target_clip_fraction": clipped_steps / len(rows),
        "max_abs_action": max_action,
        "mean_body_velocity": (active[:, 4:7].mean(axis=0).tolist()
                               if len(active) else None),
        "velocity_rmse": (
            np.sqrt(((active[:, 4:7] - active[:, 9:12]) ** 2).mean(axis=0))
            .tolist() if len(active) else None
        ),
    }
    if trace_path:
        np.savetxt(trace_path, values, delimiter=",", comments="",
                   header="t,x,y,z,vx,vy,wz,tilt,nonfoot,cmd_vx,cmd_vy,cmd_wz")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--keyboard", action="store_true")
    parser.add_argument("--terrain", choices=TERRAINS, default="flat")
    parser.add_argument("--suite", choices=["flat", "terrain", "all"],
                        default="all")
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--output", type=Path,
                        default=ROOT / "logs/model_evaluation/2026-10-14")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if args.duration < 4.0 or not math.isfinite(args.duration):
        parser.error("评估时长必须为至少 4 秒的有限值")
    args.output.mkdir(parents=True, exist_ok=True)
    policy_path, contract = export_checkpoint(args.checkpoint)
    if args.keyboard:
        policies = {args.checkpoint.stem: policy_path}
        selected = args.checkpoint.stem
        terrain_index = TERRAINS.index(args.terrain)
        while True:
            terrain = TERRAINS[terrain_index]
            scene = make_scene(terrain, args.output / "scenes")
            print(f"当前模型：{selected}；当前地形：{terrain}", flush=True)
            result = run_episode(policies[selected], scene, [0., 0., 0.],
                                 args.duration, 0, keyboard=True)
            event = result["keyboard_event"]
            if event is None:
                break
            if event in ("1", "2"):
                selected = "model_3400" if event == "1" else "model_9400"
                if selected not in policies:
                    policies[selected], _ = export_checkpoint(
                        ROOT / f"{selected}.pt",
                    )
            elif event == "T":
                terrain_index = (terrain_index + 1) % len(TERRAINS)
        return
    episodes = []
    if args.suite in ("flat", "all"):
        episodes += [(name, "flat", command) for name, command in (
            ("stand", [0., 0., 0.]),
            ("forward_02", [0.2, 0., 0.]),
            ("forward_05", [0.5, 0., 0.]),
            ("forward_08", [0.8, 0., 0.]),
            ("backward", [-0.3, 0., 0.]),
            ("left", [0., 0.2, 0.]),
            ("right", [0., -0.2, 0.]),
            ("yaw_left", [0., 0., 0.5]),
            ("yaw_right", [0., 0., -0.5]),
            ("combined", [0.4, 0.15, 0.3]),
        )]
    if args.suite in ("terrain", "all"):
        episodes += [(name, name, [0.5, 0., 0.]) for name in TERRAINS[1:]]
    report = {"checkpoint": str(args.checkpoint.resolve()),
              "contract": contract, "simulation": "MuJoCo",
              "note": "自定义闭环评估，非官方 RoboGauge 综合评分；"
                      "前 2 秒站立，3 秒后统计速度误差，倾斜>60度停止；"
                      "楼梯和波浪从 x=-1m 的平地接近。",
              "episodes": []}
    report_path = args.output / f"{args.checkpoint.stem}.json"
    for name, terrain, command in episodes:
        scene = make_scene(terrain, args.output / "scenes")
        for seed in args.seeds:
            trace = args.output / f"{args.checkpoint.stem}_{name}_s{seed}.csv"
            result = run_episode(policy_path, scene, command,
                                 args.duration, seed, trace_path=trace)
            result.update({"name": name, "terrain": terrain,
                           "trace": str(trace)})
            report["episodes"].append(result)
            report_path.write_text(json.dumps(report, ensure_ascii=False,
                                              indent=2), encoding="utf-8")
            print(f"{args.checkpoint.stem} {name} seed={seed} "
                  f"跌倒={result['fell']} "
                  f"位移={result['displacement_m'][0]:.2f}m "
                  f"实际速度={result['mean_body_velocity']} "
                  f"最大倾斜={result['max_tilt_deg']:.1f}度", flush=True)


if __name__ == "__main__":
    main()
