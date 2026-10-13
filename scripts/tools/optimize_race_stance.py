"""保留碰撞体与 PD 参数，实测搜索 Race 的足端贴地和承重姿态。"""

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import time

import mujoco
import numpy as np
from scipy.optimize import differential_evolution
from scipy.spatial import ConvexHull
import trimesh
import yaml


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "deploy/deploy_mujoco"))
from stand_race import load_robot


LEGS = ("f_l", "f_r", "b_l", "b_r")


def clipped_patch_polygon(triangles, height, band=0.001):
    """截取 CAD 底部带，返回其水平投影凸包，用于几何对比。"""
    threshold = height + band
    vertices = triangles.reshape(-1, 3)
    low = vertices[vertices[:, 2] <= threshold]
    edges_a = triangles.reshape(-1, 3)
    edges_b = np.roll(triangles, -1, axis=1).reshape(-1, 3)
    crossing = (edges_a[:, 2] < threshold) != (edges_b[:, 2] < threshold)
    a, b = edges_a[crossing], edges_b[crossing]
    if len(a):
        fraction = (threshold - a[:, 2]) / (b[:, 2] - a[:, 2])
        low = np.vstack([low, a + (b - a) * fraction[:, None]])
    points = np.unique(low[:, :2], axis=0)
    if len(points) < 3:
        return np.empty((0, 2))
    return points[ConvexHull(points).vertices]


def clipped_patch_area(triangles, height, band=0.001):
    """计算原 CAD 近底部带的投影面积，不代表物理接触面积。"""
    points = clipped_patch_polygon(triangles, height, band)
    if len(points) < 3:
        return 0.0
    return float(ConvexHull(points).volume * 1e6)


class Experiment:
    def __init__(self, output, resume=False, config=None):
        robot = load_robot(config)
        self.model, _, self.cfg, self.qpos, self.qvel, self.actuators = robot
        self.kp = np.asarray(self.cfg["kps"])
        self.kd = np.asarray(self.cfg["kds"])
        self.limits = np.asarray(self.cfg["torque_limits"])
        self.damping = np.asarray(self.cfg["passive_damping"])
        self.friction = np.asarray(self.cfg["passive_friction"])
        self.steps = round(self.cfg["pd_dt"] / self.model.opt.timestep)
        self.base = self.model.body("base_link").id
        self.floor = self.model.geom("ground").id
        self.bodies = [self.model.body(f"pitch03_leg_{leg}_link").id
                       for leg in LEGS]
        self.geoms = [self.model.geom(f"collision_pitch03_leg_{leg}_link_0").id
                      for leg in LEGS]
        self.triangles = [np.asarray(trimesh.load_mesh(
            ROOT / f"resources/race/meshes/pitch03_leg_{leg}_link.stl",
            process=False,
        ).triangles) for leg in LEGS]
        self.force = np.zeros(6)
        self.output = output
        self.records = []
        self.cache = {}
        self.started = time.monotonic()
        records_path = self.output / "candidates.jsonl"
        if resume and records_path.exists():
            self.records = [json.loads(line)
                            for line in records_path.read_text().splitlines()]
            self.cache = {
                tuple(np.round(row["parameters"], 6)): row
                for row in self.records
            }

    @staticmethod
    def angles(parameters):
        front_hip, front_knee, rear_hip, rear_knee = parameters
        return np.asarray([
            0., front_hip, front_knee, 0., front_hip, front_knee,
            0., rear_hip, rear_knee, 0., rear_hip, rear_knee,
        ])

    def control_step(self, data, target):
        velocity = data.qvel[self.qvel]
        motor = self.kp * (target - data.qpos[self.qpos]) - self.kd * velocity
        torque = np.clip(motor, -self.limits, self.limits)
        torque -= self.damping * velocity
        torque -= self.friction * np.where(
            np.abs(velocity) > 1e-6, np.sign(velocity), 0.
        )
        data.ctrl[self.actuators] = torque
        mujoco.mj_step(self.model, data, nstep=self.steps)
        return motor, torque

    def measure(self, data, torque):
        model = self.model
        rotation = data.xmat[self.base].reshape(3, 3)
        tilt = np.degrees(np.arccos(np.clip(rotation[2, 2], -1., 1.)))
        loads = np.zeros(4)
        contact_points = np.zeros((4, 3))
        nonfoot = 0.0
        for i, contact in enumerate(data.contact):
            if self.floor not in (contact.geom1, contact.geom2):
                continue
            other = contact.geom2 if contact.geom1 == self.floor else contact.geom1
            mujoco.mj_contactForce(model, data, i, self.force)
            normal = max(0., self.force[0])
            if other in self.geoms:
                foot = self.geoms.index(other)
                loads[foot] += normal
                contact_points[foot] += normal * contact.pos
            else:
                nonfoot += normal
        for i in range(4):
            if loads[i] > 0:
                contact_points[i] /= loads[i]
        return {
            "height": float(data.qpos[2]), "tilt": float(tilt),
            "loads": loads, "nonfoot": nonfoot,
            "points": contact_points,
            "speed": float(np.linalg.norm(data.qvel[:3])),
            "torque_rms": float(np.sqrt(np.mean(torque**2))),
        }

    def evaluate(self, parameters, seconds=20., label="search", full=False,
                 friction=None, mass_scale=None, perturb=None):
        parameters = np.asarray(parameters, dtype=float)
        key = tuple(np.round(parameters, 6))
        if not full and key in self.cache:
            return self.cache[key]
        # 稳健性验证只改变本次运行的数值，不修改模型文件。
        previous_friction = self.model.geom_friction.copy()
        previous_mass = self.model.body_mass.copy()
        previous_inertia = self.model.body_inertia.copy()
        if friction is not None:
            self.model.geom_friction[:, 0] = friction
        if mass_scale is not None:
            self.model.body_mass[:] *= mass_scale
            self.model.body_inertia[:] *= mass_scale
        data = mujoco.MjData(self.model)
        target = self.angles(parameters)
        data.qpos[:3] = self.cfg["base_init_pos"]
        data.qpos[3:7] = self.cfg["base_init_quat"]
        data.qpos[self.qpos] = target
        if perturb is not None:
            data.qpos[self.qpos] += perturb
        mujoco.mj_forward(self.model, data)
        frames = []
        failed = None
        motor_peak = 0.0
        sample_every = round(.05 / self.cfg["pd_dt"])
        total = round(seconds / self.cfg["pd_dt"])
        for step in range(total):
            motor, torque = self.control_step(data, target)
            motor_peak = max(motor_peak, float(np.max(np.abs(motor) / self.limits)))
            if not np.isfinite(data.qpos).all():
                failed = "非有限状态"
                break
            if data.time > 1.0 and (
                data.qpos[2] < .20
                or data.xmat[self.base].reshape(3, 3)[2, 2] < .85
            ):
                failed = "倾倒或高度过低"
                break
            if (step + 1) % sample_every == 0:
                frames.append((float(data.time), self.measure(data, torque)))
        if not frames:
            failed = failed or "缺少观测"
        late = [frame for t, frame in frames if t >= seconds - min(2., seconds/3)]
        row = {"parameters": parameters.tolist(), "label": label,
               "seconds": seconds, "failure": failed,
               "maximum_motor_limit_ratio": motor_peak}
        if late and not failed:
            load_samples = np.asarray([frame["loads"] for frame in late])
            loads = load_samples.mean(0)
            fractions = (load_samples > 5.).mean(0)
            load_cv = float(loads.std() / loads.mean())
            mujoco.mj_forward(self.model, data)
            foot_pitch, patch, reference_patch, min_z = [], [], [], []
            for i, body in enumerate(self.bodies):
                rotation = data.xmat[body].reshape(3, 3)
                triangles = self.triangles[i] @ rotation.T + data.xpos[body]
                lowest = float(triangles[:, :, 2].min())
                foot_pitch.append(float(np.degrees(np.arcsin(
                    np.clip(-rotation[2, 0], -1., 1.)
                ))))
                patch.append(clipped_patch_area(triangles, 0.))
                reference_patch.append(clipped_patch_area(triangles, lowest))
                min_z.append(lowest * 1000)
            row.update(
                height_mean_m=float(np.mean([f["height"] for f in late])),
                maximum_tilt_deg=float(max(f["tilt"] for f in late)),
                normal_load_mean_N=loads.tolist(),
                normal_load_std_N=load_samples.std(0).tolist(),
                load_cv=load_cv, contact_fraction=fractions.tolist(),
                foot_pitch_deg=foot_pitch,
                near_ground_patch_mm2=patch,
                reference_patch_mm2=reference_patch,
                visual_min_z_mm=min_z,
                nonfoot_ground_load_N=float(max(f["nonfoot"] for f in late)),
                maximum_base_speed_mps=float(max(f["speed"] for f in late)),
                torque_rms_Nm=float(np.mean([f["torque_rms"] for f in late])),
                final_qpos=data.qpos.tolist(),
                final_qvel=data.qvel.tolist(),
            )
            # 先筛选四脚持续承重、机身稳定、无其他部位撑地的姿态。
            row["stable"] = bool(
                fractions.min() >= .98 and loads.min() >= 10.
                and row["maximum_tilt_deg"] < 5.
                and row["maximum_base_speed_mps"] < .025
                and row["nonfoot_ground_load_N"] < 1.
                and min(min_z) > -.7 and max(min_z) < .7
            )
            # 用姿态固有的底部带面积评分，避免通过网格穿地刷近地面积。
            patch_quality = min(reference_patch) / 180.
            row["score"] = float(
                (load_cv / .10)**2
                + 6. * max(0., 1.-patch_quality)**2
                + .25 * (row["maximum_tilt_deg"] / 3.)**2
                + .10 * (row["torque_rms_Nm"] / 3.)**2
                + .25 * (max(abs(x) for x in min_z) / .5)**2
                + (0 if row["stable"] else 100.)
            )
        else:
            row.update(stable=False, score=1000.)
        if full:
            row["trace"] = [dict(time_s=t, height_m=f["height"],
                tilt_deg=f["tilt"], loads_N=f["loads"].tolist(),
                speed_mps=f["speed"], torque_rms_Nm=f["torque_rms"])
                for t, f in frames]
        self.model.geom_friction[:] = previous_friction
        self.model.body_mass[:] = previous_mass
        self.model.body_inertia[:] = previous_inertia
        if not full:
            self.cache[key] = row
            self.records.append(row)
            with (self.output / "candidates.jsonl").open("a") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            if len(self.records) % 25 == 0:
                best = min(self.records, key=lambda result: result["score"])
                print(f"已测 {len(self.records)} 组，当前评分 {best['score']:.4f}，"
                      f"目标角 {best['parameters']}，"
                      f"耗时 {time.monotonic()-self.started:.1f} 秒", flush=True)
        return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "logs/race-stance-search")
    parser.add_argument("--iterations", type=int, default=14)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--resume", action="store_true", help="复用已测候选")
    parser.add_argument("--config", type=Path, help="指定控制参数基线配置")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "candidates.jsonl").exists() and not args.resume:
        parser.error("输出目录已有候选记录，请使用新的 --output 目录")
    baseline_path = ROOT / "deploy/deploy_mujoco/configs/race_baseline.yaml"
    config = args.config or (baseline_path if baseline_path.exists() else None)
    experiment = Experiment(args.output, resume=args.resume, config=config)
    baseline_angles = experiment.cfg["default_angles"]
    baseline_params = [baseline_angles[i] for i in (1, 2, 7, 8)]
    baseline = experiment.evaluate(baseline_params, seconds=20.,
                                   label="baseline", full=True)
    (args.output / "baseline.json").write_text(
        json.dumps(baseline, ensure_ascii=False, indent=2)
    )
    print("原始基线：", {key: baseline[key] for key in (
        "normal_load_mean_N", "load_cv", "foot_pitch_deg",
        "reference_patch_mm2", "visual_min_z_mm", "score")}, flush=True)
    # 左右对称，先优化前后各两个 pitch 角，roll 保持原值零。
    bounds = [(-.35, .05), (.10, .55), (-.35, .05), (.05, .55)]
    result = differential_evolution(
        lambda parameters: experiment.evaluate(parameters)["score"],
        bounds, seed=args.seed, popsize=8, maxiter=args.iterations,
        tol=.005, polish=False, x0=baseline_params, updating="immediate",
    )
    stable = sorted((row for row in experiment.records if row["stable"]),
                    key=lambda row: row["score"])
    if not stable:
        raise RuntimeError("搜索没有找到满足四脚承重约束的姿态")
    finalists = []
    # 对训练初始姿态，复测同样采用 20 秒，不要求长时间静态保持。
    for index, row in enumerate(stable[:8]):
        finalist = experiment.evaluate(row["parameters"], seconds=20.,
            label=f"finalist_{index}", full=True)
        finalists.append(finalist)
    finalists = sorted(finalists, key=lambda row: row["score"])
    best = next((row for row in finalists if row["stable"]), None)
    if best is None:
        raise RuntimeError("候选未通过短时初始化复测")
    optimized = deepcopy(experiment.cfg)
    optimized["default_angles"] = experiment.angles(best["parameters"]).round(5).tolist()
    config_path = ROOT / "deploy/deploy_mujoco/configs/race_optimized.yaml"
    config_path.write_text(
        "# 当前 MuJoCo 平地搜索候选，仅目标关节角改变；原配置另行保留。\n"
        + yaml.safe_dump(optimized, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    model_path = ROOT / "resources/race/mjcf/race_pd.xml"
    report = {
        "physics": f"MuJoCo {mujoco.__version__}",
        "seed": args.seed, "bounds_rad": bounds,
        "candidate_count": len(experiment.records),
        "stable_candidate_count": len(stable),
        "baseline": baseline, "best": best, "finalists": finalists,
        "config": str(config_path),
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "geometry_note": "1 mm 底部带投影面积为 CAD 几何代理指标，"
                         "球形碰撞仍为点接触，不是压力面积。",
        "scope": "保留碰撞体、PD、力矩限幅和被动项，仅搜索前后腿 pitch 角。",
    }
    (args.output / "results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("最终候选：", {key: best[key] for key in (
        "parameters", "normal_load_mean_N", "load_cv", "foot_pitch_deg",
        "reference_patch_mm2", "maximum_tilt_deg", "score")}, flush=True)
    print("姿态配置：", config_path, flush=True)


if __name__ == "__main__":
    main()
