"""验证旧圆柱髋壳重挂到 roll 后的运动几何等价性及纯 PD 稳定性。"""

import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

import compare_race_collision_geometry as compare


OUT = compare.ROOT / "logs/race_collision_fix"


def make_copy():
    """仅改变四个零密度圆柱的所属刚体，保留其他 XML 内容。"""
    tree = ET.parse(compare.CURRENT_XML)
    root = tree.getroot()
    for element in root.iter():
        if "file" in element.attrib:
            element.set("file", str((compare.CURRENT_XML.parent
                                      / element.get("file")).resolve()))
    bodies = {body.get("name"): body for body in root.iter("body")}
    changes = []
    for leg in compare.LEGS:
        source_name = f"pitch01_leg_{leg}_link"
        target_name = f"roll_leg_{leg}_link"
        source, target = bodies[source_name], bodies[target_name]
        assert source in list(target)
        assert np.allclose(np.fromstring(source.get("pos"), sep=" "), 0)
        assert np.allclose(np.fromstring(source.get("quat"), sep=" "),
                           [1, 0, 0, 0])
        joint = source.find("joint")
        assert np.allclose(np.fromstring(joint.get("pos"), sep=" "), 0)
        assert np.allclose(np.fromstring(joint.get("axis"), sep=" "),
                           [0, 1, 0])
        geom = source.find(
            f"geom[@name='collision_pitch01_leg_{leg}_link_1']")
        assert geom.get("type") == "cylinder"
        assert geom.get("density") == "0"
        position = np.fromstring(geom.get("pos"), sep=" ")
        assert np.linalg.norm(position[[0, 2]]) < 1e-12
        source.remove(geom)
        target.append(geom)
        changes.append({"几何名称": geom.get("name"),
                        "原所属刚体": source_name, "新所属刚体": target_name})
    path = OUT / "same_shape_parent_hip_cylinders.xml"
    tree.write(path, encoding="utf-8", xml_declaration=True)
    return path, changes


def indices_with_moved_hips(model):
    """让共享 PD 诊断兼容圆柱重挂，而不修改共享工具文件。"""
    joints = [model.joint(f"{kind}_leg_{leg}_joint").id
              for leg in compare.LEGS
              for kind in ("roll", "pitch01", "pitch02")]
    legs, hips, feet = {}, {}, {}
    colliders = np.flatnonzero(model.geom_group == 1)
    for leg in compare.LEGS:
        body_ids = [model.body(f"{kind}_leg_{leg}_link").id
                    for kind in ("roll", "pitch01", "pitch02", "pitch03")]
        legs[leg] = [g for g in colliders if model.geom_bodyid[g] in body_ids]
        hips[leg] = model.geom(f"collision_pitch01_leg_{leg}_link_1").id
        feet[leg] = model.geom(f"collision_pitch03_leg_{leg}_link_0").id
    return (joints, model.jnt_qposadr[joints],
            model.geom("collision_base_link_0").id, legs, hips, feet)


def shape_equivalence(old, new, sample_count=2000, seed=20261014):
    joints, qpos, *_ = indices_with_moved_hips(old)
    other_qpos = indices_with_moved_hips(new)[1]
    rng = np.random.default_rng(seed)
    poses = rng.uniform(old.jnt_range[joints, 0],
                        old.jnt_range[joints, 1], (sample_count, 12))
    old_data, new_data = mujoco.MjData(old), mujoco.MjData(new)
    names = [old.geom(g).name for g in np.flatnonzero(old.geom_group == 1)]
    moved = {f"collision_pitch01_leg_{leg}_link_1" for leg in compare.LEGS}
    stats = {name: {"中心最大误差_m": 0.0,
                    "对称轴最大向量误差": 0.0,
                    "半径最大误差_m": 0.0,
                    "半长最大误差_m": 0.0}
             for name in moved}
    other_pose_error = 0.0
    for pose in poses:
        compare.set_pose(old, old_data, qpos, pose)
        compare.set_pose(new, new_data, other_qpos, pose)
        for name in names:
            a, b = old.geom(name).id, new.geom(name).id
            assert old.geom_type[a] == new.geom_type[b]
            assert np.array_equal(old.geom_size[a], new.geom_size[b])
            center_error = np.linalg.norm(old_data.geom_xpos[a]
                                          - new_data.geom_xpos[b])
            if name not in moved:
                other_pose_error = max(
                    other_pose_error, float(center_error),
                    float(np.max(np.abs(old_data.geom_xmat[a]
                                        - new_data.geom_xmat[b]))))
                continue
            old_axis = old_data.geom_xmat[a].reshape(3, 3)[:, 2]
            new_axis = new_data.geom_xmat[b].reshape(3, 3)[:, 2]
            axis_error = min(np.linalg.norm(old_axis - new_axis),
                             np.linalg.norm(old_axis + new_axis))
            stats[name]["中心最大误差_m"] = max(
                stats[name]["中心最大误差_m"], float(center_error))
            stats[name]["对称轴最大向量误差"] = max(
                stats[name]["对称轴最大向量误差"], float(axis_error))
            assert center_error < 1e-12
            # 来源 XML 圆柱四元数的末位误差导致约 1e-8 的轴向差异。
            assert axis_error < 1e-8
    assert other_pose_error < 1e-12
    for name, result in stats.items():
        g = old.geom(name).id
        result["半径_m"] = float(old.geom_size[g, 0])
        result["半长_m"] = float(old.geom_size[g, 1])
        result["形状边界误差保守上界_m"] = (
            result["中心最大误差_m"]
            + (result["半径_m"] + result["半长_m"])
            * result["对称轴最大向量误差"])
    return {
        "样本数": sample_count, "随机种子": seed,
        "采样定义": "12关节按各自URDF完整限位独立均匀采样",
        "全部通过": True, "各髋壳": stats,
        "未移动碰撞体的中心与旋转最大绝对误差": other_pose_error,
        "说明": (
            "圆柱绕自身对称轴的转角不改变世界占据形状。比较中心、"
            "无方向对称轴、半径、半长，而不要求完整旋转矩阵相等。"
            "来源圆柱四元数存在约1e-9级精度残差，故几何等价按数值容差。"
        ),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    path, changes = make_copy()
    old = mujoco.MjModel.from_xml_path(str(compare.CURRENT_XML))
    new = mujoco.MjModel.from_xml_path(str(path))
    mechanical_checks = compare.mechanical_assertions(old, new)
    geometric_checks = shape_equivalence(old, new)
    original_indices = compare.get_indices
    try:
        compare.get_indices = indices_with_moved_hips
        pd = {"当前模型": compare.evaluate_pd(old),
              "仅重挂四个髋壳": compare.evaluate_pd(new)}
    finally:
        compare.get_indices = original_indices
    report = {
        "说明": (
            "只测试派生MJCF，不改原URDF及正式训练配置。圆柱零密度且"
            "机械惯量来自显式inertial，重挂不改变质量惯量。"
            "本测试证明世界占据形状等价，但相邻刚体自碰过滤会随所属"
            "刚体变化，这是计划调整的接触语义，不能称物理行为完全不变。"
        ),
        "源模型": str(compare.CURRENT_XML),
        "源模型SHA256": compare.digest(compare.CURRENT_XML),
        "派生模型": str(path), "变更": changes,
        "机械参数断言": mechanical_checks,
        "随机姿态几何等价": geometric_checks,
        "10秒纯PD_CPU旁证": pd,
        "新版盒形髋壳能否同样无损重挂": {
            "可以": False,
            "原因": (
                "盒体没有绕Y轴的连续旋转对称性；新版髋壳中心也有非零"
                "X/Z偏移。绕pitch01旋转会改变占据形状，直接重挂不等价。"
            ),
        },
    }
    target = OUT / "parent_hip_cylinders_cpu.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print("2000个完整关节限位随机姿态：圆柱世界占据形状等价检查通过。")
    for name, result in pd.items():
        print(name, "稳定：", result["稳定"], "末1秒：", result["末1秒"])
    print(f"完整报告：{target}")


if __name__ == "__main__":
    main()
