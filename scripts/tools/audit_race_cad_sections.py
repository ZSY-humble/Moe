"""只读检查 Race 机身 CAD 的精确切面与单盒包络误差。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
import trimesh


ROOT = Path(__file__).resolve().parents[2]


def origin_transform(element):
    result = np.eye(4)
    if element is not None:
        result[:3, 3] = np.fromstring(element.get("xyz", "0 0 0"), sep=" ")
        angles = np.fromstring(element.get("rpy", "0 0 0"), sep=" ")
        result[:3, :3] = Rotation.from_euler("xyz", angles).as_matrix()
    return result


def clip_segments_z(segments, lower, upper):
    """裁剪三角面切线，保留边与高度带边界的真实交点。"""
    points = []
    for segment in segments:
        start, end = segment
        dz = end[2] - start[2]
        t_lower, t_upper = 0.0, 1.0
        if abs(dz) < 1e-12:
            if not lower <= start[2] <= upper:
                continue
        else:
            first, last = sorted(
                ((lower - start[2]) / dz, (upper - start[2]) / dz)
            )
            t_lower = max(t_lower, first)
            t_upper = min(t_upper, last)
        if t_lower <= t_upper:
            points.extend(
                (start + t_lower * (end - start),
                 start + t_upper * (end - start))
            )
    return np.asarray(points).reshape(-1, 3)


def summarize_points(points, box_lower, box_upper):
    if not len(points):
        return {"surface_intersection_found": False}
    lower, upper = points.min(axis=0), points.max(axis=0)
    return {
        "surface_intersection_found": True,
        "bounds_m": [lower.tolist(), upper.tolist()],
        "y_width_mm": float(1000 * (upper[1] - lower[1])),
        "z_height_mm": float(1000 * (upper[2] - lower[2])),
        "box_extra_left_positive_y_mm": float(
            1000 * (box_upper[1] - upper[1])
        ),
        "box_extra_right_negative_y_mm": float(
            1000 * (lower[1] - box_lower[1])
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "logs/race_cad_audit/sections.json",
    )
    args = parser.parse_args()
    urdf = ROOT / "resources/race/urdf/race.urdf"
    mesh_path = ROOT / "resources/race/meshes/base_link.stl"
    tree = ET.parse(urdf).getroot()
    base = tree.find("link[@name='base_link']")
    collision = base.find("collision")
    box_pose = origin_transform(collision.find("origin"))
    assert np.allclose(box_pose[:3, :3], np.eye(3))
    size = np.fromstring(collision.find("geometry/box").get("size"), sep=" ")
    box_center = box_pose[:3, 3]
    box_lower, box_upper = box_center - size / 2, box_center + size / 2
    mesh = trimesh.load(mesh_path, force="mesh")

    # 在四个髋轴，以及机身前后端与宽窄过渡区域做精确平面切割。
    x_values = sorted(set(
        np.arange(-0.24, 0.260001, 0.01).round(8).tolist()
        + [-0.2488, -0.245, -0.1825, 0.1825]
    ))
    sections = []
    for x in x_values:
        segments = trimesh.intersections.mesh_plane(
            mesh, plane_normal=[1, 0, 0], plane_origin=[x, 0, 0]
        )
        all_points = segments.reshape(-1, 3)
        band_points = clip_segments_z(segments, box_lower[2], box_upper[2])
        sections.append({
            "x_m": x,
            "triangle_plane_intersection_segments": len(segments),
            "full_height": summarize_points(
                all_points, box_lower, box_upper
            ),
            "within_collision_box_height": summarize_points(
                band_points, box_lower, box_upper
            ),
        })

    report = {
        "read_only_asset_audit": True,
        "mesh_file": str(mesh_path),
        "mesh_sha256": hashlib.sha256(mesh_path.read_bytes()).hexdigest(),
        "urdf_sha256": hashlib.sha256(urdf.read_bytes()).hexdigest(),
        "units": "米；报告中显式标注 mm 的数字为毫米",
        "mesh": {
            "vertices": len(mesh.vertices),
            "triangles": len(mesh.faces),
            "bounds_m": mesh.bounds.tolist(),
            "is_watertight": bool(mesh.is_watertight),
            "is_winding_consistent": bool(mesh.is_winding_consistent),
            "connected_components": len(mesh.split(only_watertight=False)),
        },
        "collision_box": {
            "size_m": size.tolist(),
            "bounds_m": [box_lower.tolist(), box_upper.tolist()],
        },
        "sections": sections,
        "limitations_zh": [
            "截面宽度来自三角面与平面真实交线，未使用稀疏顶点切片。",
            "106 mm 是前髋轴截面限定在原碰撞盒高度带后的宽度；"
            "不应称为该处全部高度的 CAD 总宽。",
            "STL 不是封闭实体且含多个组件，空切面表示没有表面交线，"
            "不能在没有实体重建的情况下直接判定内部全部为空。",
            "主盒各侧外扩是截面包络差，不等于整条腿与机身的最小间隙。",
        ],
    }

    # 重建已记录的右后大腿接触姿态，定位盒子的最近点。
    sample_path = ROOT / "logs/race_collision_fix/filtered_seed42.json"
    if sample_path.exists():
        sample = json.loads(sample_path.read_text())["first_ten_reset_samples"][2]
        q = dict(zip(sample["joint_names"], sample["joint_pos_rad"]))
        link_transforms = {"base_link": np.eye(4)}
        for joint in tree.findall("joint"):
            transform = origin_transform(joint.find("origin"))
            if joint.get("type") == "revolute":
                axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
                rotation = np.eye(4)
                rotation[:3, :3] = Rotation.from_rotvec(
                    axis * q.get(joint.get("name"), 0.0)
                ).as_matrix()
                transform = transform @ rotation
            parent = joint.find("parent").get("link")
            child = joint.find("child").get("link")
            link_transforms[child] = link_transforms[parent] @ transform
        thigh = tree.find(
            "link[@name='pitch01_leg_b_r_link']/"
            "collision[@name='pitch01_leg_b_r_link_contact_0']"
        )
        thigh_pose = link_transforms["pitch01_leg_b_r_link"] @ origin_transform(
            thigh.find("origin")
        )
        thigh_half = np.fromstring(
            thigh.find("geometry/box").get("size"), sep=" "
        ) / 2
        base_half = size / 2

        def objective(values):
            base_point = box_center + values[:3]
            thigh_point = thigh_pose[:3, 3] + thigh_pose[:3, :3] @ values[3:]
            return float(np.sum((base_point - thigh_point) ** 2))

        solution = minimize(
            objective, np.zeros(6), method="SLSQP",
            bounds=list(zip(-base_half, base_half))
            + list(zip(-thigh_half, thigh_half)),
            options={"ftol": 1e-18, "maxiter": 1000},
        )
        assert solution.success, solution.message
        base_point = box_center + solution.x[:3]
        thigh_point = (
            thigh_pose[:3, 3] + thigh_pose[:3, :3] @ solution.x[3:]
        )
        cad_point, cad_distance, face = trimesh.proximity.closest_point_naive(
            mesh, np.asarray([base_point])
        )
        report["recorded_contact_sample"] = {
            "source": str(sample_path),
            "sample_index_zero_based": 2,
            "sample_base_contact_n": sample["base_contact_n"],
            "obb_distance_mm": float(np.sqrt(solution.fun) * 1000),
            "closest_base_box_point_m": base_point.tolist(),
            "closest_thigh_box_point_m": thigh_point.tolist(),
            "base_box_point_to_base_cad_surface_mm": float(
                cad_distance[0] * 1000
            ),
            "closest_base_cad_point_m": cad_point[0].tolist(),
            "closest_cad_triangle_index": int(face[0]),
            "interpretation_zh": "最近碰撞盒点位于机身后右下角；"
            "该点距 CAD 表面很远，说明这次接近发生在单盒多包出的区域。"
            "这不是完整大腿与完整机身的 CAD 最小间隙，"
            "也不是接触历史窗口中的确切碰撞瞬间。",
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(args.output)
    for section in sections:
        if abs(abs(section["x_m"]) - 0.1825) < 1e-9:
            print(json.dumps(section, ensure_ascii=False))
    if "recorded_contact_sample" in report:
        print(json.dumps(report["recorded_contact_sample"], ensure_ascii=False))


if __name__ == "__main__":
    main()
