"""固定 Race 机械模型，在 CPU 上比较候选碰撞几何的间隙与重叠。"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
CURRENT_XML = ROOT / "resources/race/mjcf/race_pd.xml"
ASSETS = Path(
    "/home/zsy/project/mujoco_playground/mujoco_playground/"
    "_src/locomotion/race/assets"
)
OUT = ROOT / "logs/race_collision_comparison"
LEGS = ("f_l", "f_r", "b_l", "b_r")
DEFAULT = np.tile([0.0, -0.12, 0.30], 4)
MECHANICAL_ARRAYS = (
    "body_parentid", "body_pos", "body_quat", "body_mass", "body_inertia",
    "body_ipos", "body_iquat", "body_gravcomp", "jnt_type", "jnt_bodyid",
    "jnt_pos", "jnt_axis", "jnt_limited", "jnt_range", "qpos0",
    "dof_damping", "dof_frictionloss", "dof_armature", "actuator_trntype",
    "actuator_trnid", "actuator_gear", "actuator_ctrlrange",
    "actuator_forcerange", "actuator_gainprm", "actuator_biasprm",
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def numbers(values):
    return " ".join(format(float(value), ".17g") for value in values)


def build_candidate(source_json, output_xml):
    """只替换碰撞体；候选自带的地形接触过滤在此不生效。"""
    tree = ET.parse(CURRENT_XML)
    xml = tree.getroot()
    # 派生文件放在日志目录，所有外部资产改成绝对路径。
    for element in xml.iter():
        if "file" in element.attrib:
            element.set(
                "file", str((CURRENT_XML.parent / element.get("file")).resolve())
            )
    old = [g for g in xml.iter("geom") if g.get("group") == "1"]
    assert len(old) == 26
    # 避免把“几何更好”与“关闭了自碰撞”混为一谈。
    contact_keys = (
        "contype", "conaffinity", "condim", "friction", "solref",
        "solimp", "margin", "gap", "priority", "solmix",
    )
    contact = {key: old[0].get(key) for key in contact_keys
               if old[0].get(key) is not None}
    for key, value in contact.items():
        assert all(g.get(key) == value for g in old), key
    for parent in xml.iter():
        for element in list(parent):
            if element.tag == "geom" and element.get("group") == "1":
                parent.remove(element)
    bodies = {body.get("name"): body for body in xml.iter("body")}
    source = json.loads(Path(source_json).read_text())
    for geom in source["geoms"]:
        assert geom["body"] in bodies, geom["body"]
        attributes = {
            "name": geom["name"], "type": geom["type"],
            "pos": numbers(geom["pos"]), "quat": numbers(geom["quat"]),
            "size": numbers(geom["size"]), "density": "0", "group": "1",
            "rgba": "0.2 0.7 0.8 0.5", **contact,
        }
        ET.SubElement(bodies[geom["body"]], "geom", attributes)
    output_xml.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_xml, encoding="utf-8", xml_declaration=True)
    return source


def mechanical_assertions(reference, candidate):
    assert (reference.nq, reference.nv, reference.nu, reference.nbody,
            reference.njnt) == (
                candidate.nq, candidate.nv, candidate.nu, candidate.nbody,
                candidate.njnt,
            )
    errors = {}
    for name in MECHANICAL_ARRAYS:
        old, new = getattr(reference, name), getattr(candidate, name)
        assert old.shape == new.shape, name
        error = float(np.max(np.abs(old - new), initial=0))
        assert error < 1e-12, (name, error)
        errors[name] = error
    assert np.array_equal(reference.opt.gravity, candidate.opt.gravity)
    assert reference.opt.timestep == candidate.opt.timestep
    return {"全部通过": True, "各数组最大绝对误差": errors}


def get_indices(model):
    joints = [model.joint(f"{kind}_leg_{leg}_joint").id
              for leg in LEGS for kind in ("roll", "pitch01", "pitch02")]
    qpos = model.jnt_qposadr[joints]
    body = model.body("base_link").id
    colliders = np.flatnonzero(model.geom_group == 1)
    base = [g for g in colliders if model.geom_bodyid[g] == body]
    assert len(base) == 1
    legs = {}
    hips = {}
    feet = {}
    for leg in LEGS:
        leg_bodies = [model.body(f"pitch0{i}_leg_{leg}_link").id
                      for i in (1, 2, 3)]
        legs[leg] = [g for g in colliders
                     if model.geom_bodyid[g] in leg_bodies]
        hip_candidates = [g for g in legs[leg]
                          if model.geom_bodyid[g] == leg_bodies[0]]
        assert len(hip_candidates) == 2
        # 髋壳位于 pitch01 坐标原点附近，大腿长杆距原点更远。
        hips[leg] = min(hip_candidates, key=lambda g: np.linalg.norm(
            model.geom_pos[g, [0, 2]]))
        feet[leg] = [g for g in legs[leg]
                     if model.geom_bodyid[g] == leg_bodies[-1]]
        assert len(feet[leg]) == 1
        feet[leg] = feet[leg][0]
        assert model.geom_type[feet[leg]] == mujoco.mjtGeom.mjGEOM_SPHERE
    return joints, qpos, base[0], legs, hips, feet


def distances(model, data, base, geoms):
    # MuJoCo 3.4 native CCD 对部分轻微旋转盒体返回不一致的零间隙：
    # 最近点线段仍有厘米长度。盒-盒先用分离轴法判断相交，其余用 native。
    # libccd 对大查询距离的盒-圆柱也有异常，不能全局替换后端。
    # 查询结束恢复设置，不改变动力学后端、模型参数或碰撞过滤。
    flags = model.opt.disableflags
    try:
        values = []
        for geom in geoms:
            is_box_pair = model.geom_type[base] == model.geom_type[geom] == (
                mujoco.mjtGeom.mjGEOM_BOX)
            if is_box_pair:
                rot_a = data.geom_xmat[base].reshape(3, 3)
                rot_b = data.geom_xmat[geom].reshape(3, 3)
                axes = np.vstack((rot_a.T, rot_b.T, np.cross(
                    rot_a.T[:, None, :], rot_b.T[None, :, :]).reshape(-1, 3)))
                norms = np.linalg.norm(axes, axis=1)
                axes = axes[norms > 1e-12] / norms[norms > 1e-12, None]
                delta = data.geom_xpos[geom] - data.geom_xpos[base]
                intervals = (
                    np.abs(axes @ delta)
                    - np.abs(axes @ rot_a) @ model.geom_size[base]
                    - np.abs(axes @ rot_b) @ model.geom_size[geom]
                )
                sat_gap = float(intervals.max())
                # 15条轴均重叠时，最大带符号间隙即最小平移穿透深度。
                if sat_gap <= 0.0:
                    values.append(sat_gap)
                    continue
            model.opt.disableflags = flags | int(
                mujoco.mjtDisableBit.mjDSBL_NATIVECCD) if is_box_pair else flags
            value = mujoco.mj_geomDistance(model, data, base, geom, 10.0, None)
            # 分离轴已证实分离时，不接受求解器的虚假负值。
            if is_box_pair and value <= 0:
                value = sat_gap
            values.append(value)
        return np.asarray(values)
    finally:
        model.opt.disableflags = flags


def set_pose(model, data, qpos, angles, height=1.0):
    data.qpos[:] = model.qpos0
    data.qpos[:3] = [0.0, 0.0, height]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qpos[qpos] = angles
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def summarize(values):
    values = np.asarray(values)
    return {
        "最小值_m": float(values.min()),
        "中位数_m": float(np.median(values)),
        "百分之一分位_m": float(np.quantile(values, .01)),
        "百分之五分位_m": float(np.quantile(values, .05)),
        "重叠比例": float(np.mean(values < -1e-6)),
        "重叠判定阈值_m": -1e-6,
        "间隙小于1mm比例": float(np.mean(values < .001)),
    }


def evaluate(model, poses):
    data = mujoco.MjData(model)
    joints, qpos, base, legs, hips, feet = get_indices(model)
    colliders = np.flatnonzero(model.geom_group == 1)
    result = {
        "碰撞体数量": len(colliders),
        "形状数量": dict(Counter(
            mujoco.mjtGeom(int(model.geom_type[g])).name
            for g in colliders
        )),
        "机器人总质量_kg": float(model.body_mass.sum()),
        "碰撞掩码": {
            "contype": sorted(set(model.geom_contype[colliders].tolist())),
            "conaffinity": sorted(set(
                model.geom_conaffinity[colliders].tolist())),
        },
        "四腿外展扫描": {},
        "平地足底间隙": {},
    }
    angles = [0.0]
    for angle in (.1, .2, .25, .27, .3, .4, .733):
        angles += [-angle, angle]
    for leg_index, leg in enumerate(LEGS):
        rows = []
        for angle in sorted(angles):
            pose = DEFAULT.copy()
            pose[leg_index * 3] = angle
            set_pose(model, data, qpos, pose)
            rows.append({
                "外展角_rad": angle,
                "机身到髋壳间隙_m": float(distances(
                    model, data, base, [hips[leg]])[0]),
                "机身到该腿任意碰撞体最小间隙_m": float(distances(
                    model, data, base, legs[leg]).min()),
            })
        # 解析扫描首次负间隙，精细二分得到默认髋/膝下的碰撞边界。
        boundaries = {}
        for direction in (-1, 1):
            probe = np.linspace(0, .733, 734)
            crossing = None
            previous = 0.0
            for magnitude in probe:
                pose = DEFAULT.copy()
                pose[leg_index * 3] = direction * magnitude
                set_pose(model, data, qpos, pose)
                gap = distances(model, data, base, [hips[leg]])[0]
                if gap < 0:
                    low, high = previous, magnitude
                    for _ in range(25):
                        middle = (low + high) / 2
                        pose[leg_index * 3] = direction * middle
                        set_pose(model, data, qpos, pose)
                        if distances(model, data, base, [hips[leg]])[0] < 0:
                            high = middle
                        else:
                            low = middle
                    crossing = direction * high
                    break
                previous = magnitude
            boundaries[str(direction)] = crossing
        result["四腿外展扫描"][leg] = {
            "髋壳碰撞体": model.geom(hips[leg]).name,
            "逐角度": rows, "首次几何接触边界_rad": boundaries,
        }
    for pose_name, pose in (("默认站姿", DEFAULT), ("全零关节", np.zeros(12))):
        set_pose(model, data, qpos, pose, height=0.0)
        relative_bottom = {
            leg: float(data.geom_xpos[geom, 2] - model.geom_size[geom, 0])
            for leg, geom in feet.items()
        }
        result["平地足底间隙"][pose_name] = {
            "四足相对基座原点最低点_m": relative_bottom,
            "最低足底刚好接地的基座高度_m": -min(relative_bottom.values()),
            "不同基座高度下最低足底间隙_m": {
                str(height): float(height + min(relative_bottom.values()))
                for height in (.335, .35, .36, .37)
            },
        }
    samples = np.empty((len(poses), 4))
    worst = None
    for pose_index, pose in enumerate(poses):
        set_pose(model, data, qpos, pose)
        for leg_index, leg in enumerate(LEGS):
            gaps = distances(model, data, base, legs[leg])
            samples[pose_index, leg_index] = gaps.min()
            if worst is None or gaps.min() < worst["距离_m"]:
                pair_index = int(gaps.argmin())
                worst = {
                    "样本序号": pose_index, "腿": leg,
                    "碰撞体": model.geom(legs[leg][pair_index]).name,
                    "距离_m": float(gaps.min()), "关节角_rad": pose.tolist(),
                }
    result["随机姿态扫描"] = {
        "样本数": len(poses),
        "任意一腿碰撞机身": summarize(samples.min(axis=1)),
        "各腿碰撞机身": {
            leg: summarize(samples[:, i]) for i, leg in enumerate(LEGS)
        },
        "最严重重叠样本": worst,
    }
    return result


def evaluate_pd(model, duration=10.0):
    """平地零命令纯 PD 旁证，不等同于 PhysX 的训练回放。"""
    data = mujoco.MjData(model)
    _, qpos, base_geom, legs, _, feet = get_indices(model)
    qvel = np.asarray([model.jnt_dofadr[model.joint(
        f"{kind}_leg_{leg}_joint").id]
        for leg in LEGS for kind in ("roll", "pitch01", "pitch02")])
    joint_ids = model.dof_jntid[qvel]
    actuators = [int(np.flatnonzero(
        (model.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT)
        & (model.actuator_trnid[:, 0] == joint))[0]) for joint in joint_ids]
    kp = np.tile([30.0, 48.0, 51.0], 4)
    kd = np.tile([1.15, 1.20, 1.00], 4)
    limits = np.tile([22.0, 22.0, 33.0], 4)
    speed = np.tile([31.4, 31.4, 21.0], 4)
    knee = speed * .45
    # XML 机械参数、重力、接触、碰撞掩码保持不变。
    model.opt.timestep = .001
    pd_steps = 5
    set_pose(model, data, qpos, DEFAULT, height=.36)
    base_body = model.geom_bodyid[base_geom]
    leg_bodies = set(model.geom_bodyid[g]
                     for geoms in legs.values() for g in geoms)
    foot_body_to_index = {
        model.geom_bodyid[feet[leg]]: i for i, leg in enumerate(LEGS)
    }
    rows = []
    max_self_force = 0.0
    max_self_force_event = None
    nonfoot_ground_steps = 0
    max_abs_torque = np.zeros(12)
    for step in range(round(duration / model.opt.timestep)):
        if step % pd_steps == 0:
            velocity = data.qvel[qvel]
            torque = kp * (DEFAULT - data.qpos[qpos]) - kd * velocity
            curve = np.clip((speed - np.abs(velocity)) / (speed - knee),
                            0.0, 1.0)
            torque = np.clip(torque, -limits * curve, limits * curve)
            data.ctrl[actuators] = torque
            max_abs_torque = np.maximum(max_abs_torque, np.abs(torque))
        mujoco.mj_step(model, data)
        assert np.isfinite(data.qpos).all()
        loads = np.zeros(4)
        has_nonfoot = False
        for contact_id in range(data.ncon):
            contact = data.contact[contact_id]
            b1, b2 = model.geom_bodyid[contact.geom]
            force = np.zeros(6)
            mujoco.mj_contactForce(model, data, contact_id, force)
            force_magnitude = float(np.linalg.norm(force[:3]))
            if (b1 == base_body and b2 in leg_bodies) or (
                    b2 == base_body and b1 in leg_bodies):
                if force_magnitude > max_self_force:
                    max_self_force = force_magnitude
                    max_self_force_event = {
                        "时间_s": float(data.time), "基座高度_m":
                        float(data.qpos[2]), "力_N": force_magnitude,
                        "碰撞体": [model.geom(g).name for g in contact.geom],
                    }
            if b1 == 0 or b2 == 0:
                robot_body = b2 if b1 == 0 else b1
                if robot_body in foot_body_to_index:
                    loads[foot_body_to_index[robot_body]] += max(0, force[0])
                else:
                    has_nonfoot = True
        nonfoot_ground_steps += int(has_nonfoot)
        rotation = data.xmat[base_body].reshape(3, 3)
        tilt = np.degrees(np.arccos(np.clip(rotation[2, 2], -1., 1.)))
        rows.append([data.time, data.qpos[2], tilt, *loads.tolist()])
    rows = np.asarray(rows)
    last = rows[rows[:, 0] >= duration - 1.0]
    height_peak_to_peak = np.ptp(last[:, 1])
    stable = (np.min(last[:, 1]) > .25 and np.max(last[:, 2]) < 15
              and height_peak_to_peak < .02 and nonfoot_ground_steps == 0)
    return {
        "说明": (
            "CPU MuJoCo动力学旁证，非PhysX训练验收。所有候选机械模型、"
            "碰撞掩码、控制完全相同；不施加随机化、命令或策略动作。"
            "只使用当前训练PD和力矩转速曲线，不叠加历史部署被动项。"
            "保留当前MJCF的armature和零被动项；与训练关节摩擦等不等价。"
        ),
        "时长_s": float(data.time),
        "控制": {"目标角_rad": DEFAULT.tolist(), "Kp": kp.tolist(),
                 "Kd": kd.tolist(), "力矩上限_Nm": limits.tolist(),
                 "转速上限_rad_s": speed.tolist(),
                 "满力矩拐点_rad_s": knee.tolist(),
                 "释放高度_m": .36, "积分步长_s": model.opt.timestep,
                 "PD刷新周期_s": model.opt.timestep * pd_steps,
                 "额外被动阻尼摩擦": False},
        "稳定": bool(stable),
        "稳定判据": "末1秒高度>0.25m、倾斜<15度、高度极差<2cm且无非足触地",
        "末1秒": {
            "平均高度_m": float(last[:, 1].mean()),
            "最小高度_m": float(last[:, 1].min()),
            "最大高度_m": float(last[:, 1].max()),
            "平均倾斜_deg": float(last[:, 2].mean()),
            "最大倾斜_deg": float(last[:, 2].max()),
            "四脚同时接地比例_大于1N": float(
                np.all(last[:, 3:] > 1, axis=1).mean()),
            "平均各脚法向力_N": last[:, 3:].mean(axis=0).tolist(),
        },
        "全程最大倾斜_deg": float(rows[:, 2].max()),
        "全程最低高度_m": float(rows[:, 1].min()),
        "非足触地积分步数": nonfoot_ground_steps,
        "最末关节角_rad": data.qpos[qpos].tolist(),
        "各关节最大绝对力矩_Nm": max_abs_torque.tolist(),
        "最大单接触机身对腿接触力_N": max_self_force,
        "最大机身对腿接触事件": max_self_force_event,
        "最末qpos": data.qpos.tolist(),
    }


def evaluate_small_range(model, poses):
    """仅比较实际关节位置的小范围几何扰动，不模拟观测误差。"""
    data = mujoco.MjData(model)
    _, qpos, base, legs, hips, _ = get_indices(model)
    leg_gaps = np.empty((len(poses), 4))
    hip_gaps = np.empty_like(leg_gaps)
    for index, pose in enumerate(poses):
        set_pose(model, data, qpos, pose)
        for leg_index, leg in enumerate(LEGS):
            leg_gaps[index, leg_index] = distances(
                model, data, base, legs[leg]).min()
            hip_gaps[index, leg_index] = distances(
                model, data, base, [hips[leg]])[0]
    return {
        "样本数": len(poses),
        "任意腿到机身最小间隙": summarize(leg_gaps.min(axis=1)),
        "任意髋壳到机身最小间隙": summarize(hip_gaps.min(axis=1)),
        "各腿到机身最小间隙": {
            leg: summarize(leg_gaps[:, index])
            for index, leg in enumerate(LEGS)
        },
        "各髋壳到机身最小间隙": {
            leg: summarize(hip_gaps[:, index])
            for index, leg in enumerate(LEGS)
        },
    }


def small_range_report(reference, sample_count, seed):
    """给三种独立假设分别采样，所有候选共用相同姿态。"""
    joints, *_ = get_indices(reference)
    rng = np.random.default_rng(seed)
    cases = (
        ("单独实际关节位置正负0.03rad", DEFAULT + rng.uniform(
            -.03, .03, (sample_count, 12)),
         "q=q_default+Uniform(-0.03,0.03)rad，各关节独立"),
        ("原版关节复位乘性范围", DEFAULT * rng.uniform(
            .5, 1.5, (sample_count, 12)),
         "q=q_default*Uniform(0.5,1.5)，各关节独立；外展默认0所以仍为0"),
        ("单独实际关节位置正负0.035rad", DEFAULT + rng.uniform(
            -.035, .035, (sample_count, 12)),
         "q=q_default+Uniform(-0.035,0.035)rad，各关节独立；几何假设"),
    )
    models = {"当前primitive26": reference}
    for name in ("primitive26_v1", "mjx26_v1"):
        model = mujoco.MjModel.from_xml_path(str(
            OUT / f"same_mechanics_{name}.xml"))
        mechanical_assertions(reference, model)
        models[name] = model
    report = {
        "说明": (
            "CPU MuJoCo静态几何核验；每种采样单独测试，没有互相叠加。"
            "所有候选共享相同姿态、机械模型、碰撞掩码。这里直接扰动实际"
            "关节位置；观测噪声只改变测量值，本身不会改变几何。"
        ),
        "与先前0.25rad正态扫描的区别": (
            "先前q_default+Normal(0,0.25rad)只是宽幅几何压力扫描，"
            "不是训练状态分布、不是复位分布，也不是观测噪声分布；"
            "其重叠率不能当作训练中的实际自碰概率。"
        ),
        "零位扰动解释": (
            "正负0.035rad独立测试只回答假设实际位置产生同幅位移时的"
            "几何间隙；电机零位/目标偏移并不等于瞬间改变实际位置。"
        ),
        "默认目标角_rad": DEFAULT.tolist(), "随机种子": seed,
        "MuJoCo版本": mujoco.__version__, "实验": {},
        "几何距离后端": (
            "盒-盒用15轴SAT判断相交和穿透深度，分离时用libccd距离，"
            "其余用native CCD，仅查询时临时切换。避免"
            "native CCD对部分轻微旋转盒体返回与最近点不一致的零间隙；"
            "也避免libccd在大查询距离下对盒-圆柱出现异常。"
        ),
    }
    for case_name, poses, definition in cases:
        poses = np.clip(poses, reference.jnt_range[joints, 0],
                        reference.jnt_range[joints, 1])
        report["实验"][case_name] = {
            "采样定义": definition,
            "实际每关节最小值_rad": poses.min(axis=0).tolist(),
            "实际每关节最大值_rad": poses.max(axis=0).tolist(),
            "模型": {
                name: evaluate_small_range(model, poses)
                for name, model in models.items()
            },
        }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--dynamics-only", action="store_true",
                        help="只执行10秒平地零命令纯PD同模型对照")
    parser.add_argument("--small-range-only", action="store_true",
                        help="只测试0.03rad、原版复位、0.035rad三组独立范围")
    args = parser.parse_args()
    assert args.samples >= 1000
    OUT.mkdir(parents=True, exist_ok=True)
    reference = mujoco.MjModel.from_xml_path(str(CURRENT_XML))
    if args.small_range_only:
        report = small_range_report(reference, args.samples, args.seed)
        target = OUT / "small_range_geometry.json"
        target.write_text(json.dumps(report, ensure_ascii=False, indent=2)
                          + "\n")
        for case_name, case in report["实验"].items():
            print(case_name)
            for name, result in case["模型"].items():
                stats = result["任意髋壳到机身最小间隙"]
                print(f"  {name}: 重叠={stats['重叠比例']:.3%}，"
                      f"最小髋壳间隙={stats['最小值_m'] * 1000:.4f}mm")
        print(f"小范围几何报告：{target}")
        return
    if args.dynamics_only:
        result = {"当前primitive26": evaluate_pd(reference)}
        for name in ("primitive26_v1", "mjx26_v1"):
            model = mujoco.MjModel.from_xml_path(str(
                OUT / f"same_mechanics_{name}.xml"))
            mechanical_assertions(reference, model)
            result[name] = evaluate_pd(model)
        target = OUT / "pd_dynamics.json"
        target.write_text(json.dumps(result, ensure_ascii=False, indent=2)
                          + "\n")
        for name, stats in result.items():
            print(name, stats["稳定"], stats["末1秒"],
                  stats["最大单接触机身对腿接触力_N"])
        print(f"动力学报告：{target}")
        return
    joints, *_ = get_indices(reference)
    rng = np.random.default_rng(args.seed)
    poses = np.clip(
        DEFAULT + .25 * rng.standard_normal((args.samples, 12)),
        reference.jnt_range[joints, 0], reference.jnt_range[joints, 1],
    )
    report = {
        "说明": (
            "CPU MuJoCo 几何预筛；同一机械模型、同一2000组随机姿态。"
            "仅替换group1碰撞几何，所有候选保留当前自碰撞掩码。"
            "负间隙为几何重叠，不等同于PhysX实际接触力或训练终止率。"
        ),
        "MuJoCo版本": mujoco.__version__, "随机种子": args.seed,
        "几何距离后端": (
            "盒-盒用15轴SAT判断相交和穿透深度，分离时用libccd距离，"
            "其余用native CCD，仅查询时切换，动力学不变"
        ),
        "采样": "默认站姿+[0.25 rad标准差]正态扰动，逐关节截断URDF限位",
        "采样用途": (
            "宽幅几何压力扫描，不能等同训练状态分布、复位分布、"
            "观测噪声分布或训练中实际自碰概率。"
        ),
        "原始模型路径": str(CURRENT_XML), "原始模型SHA256": digest(CURRENT_XML),
        "模型": {},
    }
    report["模型"]["当前primitive26"] = evaluate(reference, poses)
    for name in ("primitive26_v1", "mjx26_v1"):
        source_json = ASSETS / f"{name}.json"
        output_xml = OUT / f"same_mechanics_{name}.xml"
        source = build_candidate(source_json, output_xml)
        model = mujoco.MjModel.from_xml_path(str(output_xml))
        checks = mechanical_assertions(reference, model)
        result = evaluate(model, poses)
        result.update({
            "源文件": str(source_json), "源文件SHA256": digest(source_json),
            "派生同机械模型": str(output_xml), "机械参数未变断言": checks,
            "源文件接触过滤说明": source.get("contact_filter", source.get(
                "recipe", {}).get("contact_filter")),
        })
        report["模型"][name] = result
    target = OUT / "geometry.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    for name, result in report["模型"].items():
        stats = result["随机姿态扫描"]["任意一腿碰撞机身"]
        foot = result["平地足底间隙"]["默认站姿"]
        print(f"{name}: {result['形状数量']}")
        print(f"  任意腿碰机身比例={stats['重叠比例']:.3%}, "
              f"最小间隙={stats['最小值_m'] * 1000:.3f} mm")
        print(f"  默认站姿刚接地基座高度="
              f"{foot['最低足底刚好接地的基座高度_m']:.6f} m")
        print("  右前外展接触边界="
              f"{result['四腿外展扫描']['f_r']['首次几何接触边界_rad']}")
    print(f"完整报告：{target}")


if __name__ == "__main__":
    main()
