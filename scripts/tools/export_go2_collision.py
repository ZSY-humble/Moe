"""从实际 MJCF 和 URDF 提取 Go2 碰撞几何，生成离线交互 HTML。"""

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]


def transform(position, rotation):
    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = position
    return result


def origin_transform(origin):
    if origin is None:
        return np.eye(4)
    position = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
    roll, pitch, yaw = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rotation = np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])
    return transform(position, rotation)


def axis_rotation(axis, angle):
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return (
        np.eye(3) * np.cos(angle)
        + (1 - np.cos(angle)) * np.outer(axis, axis)
        + np.sin(angle) * skew
    )


def record(name, link, shape, size, world):
    return {
        "name": name,
        "link": link,
        "shape": shape,
        # 统一约定：盒子为半尺寸，圆柱为半长，球为半径。
        "size": np.asarray(size).round(7).tolist(),
        "position": world[:3, 3].round(7).tolist(),
        "rotation": world[:3, :3].round(7).flatten().tolist(),
    }


def mjcf_geometry(cfg, model_path=None):
    model_path = model_path or ROOT / "resources/go2/flat.xml"
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    data.qpos[:3] = cfg["base_init_pos"]
    data.qpos[3:7] = cfg["base_init_quat"]
    for name, angle in zip(cfg["mujoco_joint_names"], cfg["default_angles"]):
        data.qpos[model.joint(name).qposadr[0]] = angle
    mujoco.mj_forward(model, data)
    shapes = {
        mujoco.mjtGeom.mjGEOM_BOX: "box",
        mujoco.mjtGeom.mjGEOM_CYLINDER: "cylinder",
        mujoco.mjtGeom.mjGEOM_SPHERE: "sphere",
    }
    result = []
    for i in range(model.ngeom):
        if model.geom_bodyid[i] == 0:
            continue
        if model.geom_contype[i] == 0 and model.geom_conaffinity[i] == 0:
            continue
        shape = shapes.get(model.geom_type[i])
        if shape is None:
            raise ValueError(f"未支持的碰撞几何类型：{model.geom_type[i]}")
        result.append(record(
            model.geom(i).name or f"geom_{i}",
            model.body(model.geom_bodyid[i]).name,
            shape, model.geom_size[i],
            transform(data.geom_xpos[i], data.geom_xmat[i].reshape(3, 3)),
        ))
    return result


def urdf_geometry(cfg, urdf_path=None):
    urdf_path = urdf_path or ROOT / "resources/go2/urdf/go2.urdf"
    robot = ET.parse(urdf_path).getroot()
    links = {link.get("name"): link for link in robot.findall("link")}
    joints = robot.findall("joint")
    child_names = {joint.find("child").get("link") for joint in joints}
    roots = set(links) - child_names
    if len(roots) != 1:
        raise ValueError("URDF 必须包含一个根连杆")
    root_quat = np.asarray(cfg["base_init_quat"], dtype=float)
    rotation = np.zeros(9)
    mujoco.mju_quat2Mat(rotation, root_quat)
    poses = {
        roots.pop(): transform(cfg["base_init_pos"], rotation.reshape(3, 3))
    }
    angles = dict(zip(cfg["mujoco_joint_names"], cfg["default_angles"]))
    pending = joints.copy()
    while pending:
        progress = False
        for joint in pending.copy():
            parent = joint.find("parent").get("link")
            if parent not in poses:
                continue
            pose = poses[parent] @ origin_transform(joint.find("origin"))
            if joint.get("type") in ("revolute", "continuous"):
                axis_node = joint.find("axis")
                axis = np.fromstring(
                    axis_node.get("xyz", "1 0 0")
                    if axis_node is not None else "1 0 0", sep=" "
                )
                pose = pose @ transform(
                    [0, 0, 0], axis_rotation(axis, angles.get(joint.get("name"), 0))
                )
            poses[joint.find("child").get("link")] = pose
            pending.remove(joint)
            progress = True
        if not progress:
            raise ValueError("URDF 连杆树不连通或包含环")
    result = []
    for name, link in links.items():
        for index, collision in enumerate(link.findall("collision")):
            geometry = list(collision.find("geometry"))[0]
            shape = geometry.tag
            if shape == "box":
                size = np.fromstring(geometry.get("size"), sep=" ") / 2
            elif shape == "cylinder":
                size = [float(geometry.get("radius")),
                        float(geometry.get("length")) / 2, 0]
            elif shape == "sphere":
                size = [float(geometry.get("radius")), 0, 0]
            else:
                raise ValueError(f"未支持的 URDF 碰撞几何：{shape}")
            result.append(record(
                collision.get("name") or f"{name}_collision_{index}",
                name, shape, size,
                poses[name] @ origin_transform(collision.find("origin")),
            ))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "resources/go2/go2_collision.html"
    )
    args = parser.parse_args()
    config = ROOT / "deploy/deploy_mujoco/configs/go2.yaml"
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    data = {
        "sources": {
            "mjcf": {"label": "MuJoCo / MJCF", "path": "resources/go2/go2.xml",
                     "geometries": mjcf_geometry(cfg)},
            "urdf": {"label": "IsaacLab / URDF",
                     "path": "resources/go2/urdf/go2.urdf",
                     "geometries": urdf_geometry(cfg)},
        },
        "height": cfg["base_init_pos"][2],
        "joints": dict(zip(cfg["mujoco_joint_names"], cfg["default_angles"])),
    }
    template = (ROOT / "resources/go2/collision_viewer.template.html").read_text(
        encoding="utf-8"
    )
    encoded = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        template.replace("__GO2_MODEL_DATA__", encoded.replace("<", "\\u003c")),
        encoding="utf-8",
    )
    print(f"已生成：{args.output}")
    for source in data["sources"].values():
        print(f"{source['label']}：{len(source['geometries'])} 个碰撞几何体")


if __name__ == "__main__":
    main()
