"""根据移植的 Race URDF 和原参数快照生成平地纯 PD 模型。"""

import hashlib
from copy import deepcopy
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

import yaml


ROOT = Path(__file__).resolve().parents[2]


def numbers(values):
    """保留 URDF 的数值精度，包括小质量和惯量。"""
    return " ".join("{:.17g}".format(float(x)) for x in values)


def origin(element):
    node = element.find("origin")
    if node is None:
        return np.zeros(3), np.zeros(3)
    return (np.fromstring(node.get("xyz", "0 0 0"), sep=" "),
            np.fromstring(node.get("rpy", "0 0 0"), sep=" "))


def rpy_rotation(rpy):
    """URDF 的固定轴 Rz(yaw) Ry(pitch) Rx(roll)。"""
    r, p, y = rpy
    cr, cp, cy = np.cos([r, p, y])
    sr, sp, sy = np.sin([r, p, y])
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def rpy_quaternion(rpy):
    r, p, y = np.asarray(rpy) / 2
    cr, cp, cy = np.cos([r, p, y])
    sr, sp, sy = np.sin([r, p, y])
    return [cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy]


def _parameter(name, mapping):
    keys = [key for key in mapping if key in name]
    if len(keys) != 1:
        raise ValueError("关节参数匹配不唯一：" + name)
    return mapping[keys[0]]


def _scene_appearance(mjcf, asset, world):
    """直接复用 Go2 平地的纹理、背景、灯光和显示设置。"""
    scene_path = ROOT / "resources/go2/flat.xml"
    scene = ET.parse(scene_path).getroot()
    for tag in ("statistic", "visual"):
        element = scene.find(tag)
        if element is not None:
            mjcf.append(deepcopy(element))
    for element in scene.findall("asset/*"):
        asset.append(deepcopy(element))
    for light in scene.findall("worldbody/light"):
        world.append(deepcopy(light))
    floor = scene.find("worldbody/geom[@name='floor']")
    if floor is None or floor.get("type") != "plane":
        raise ValueError("Go2 的参考场景必须包含 floor 平面")
    return {
        key: floor.get(key) for key in ("size", "material", "rgba")
        if floor.get(key) is not None
    }


def _terrain(world, terrain, floor_visual):
    """沿用来源 MuJoCo 平地接触设置，仅提供模型查看场景。"""
    if terrain != "plane":
        raise ValueError("当前只移植机器人模型，场景仅支持平地")
    ET.SubElement(
        world, "geom", name="ground", type="plane", **floor_visual,
        contype="1", conaffinity="2", condim="3", group="0",
        friction="1 0.005 0.0001", solref="0.004 1",
        solimp="0.95 0.99 0.001", margin="0",
    )
    return dict(origin_z=0.0, terrain="plane", level=0,
                terrain_representation="plane", exit_x=4.5)


def build_model(terrain="plane", level=0, timestep=0.001, output=None, *,
                urdf_path=None, control_cfg=None):
    """编译唯一的 26 碰撞体 Race，动力学参数只读本仓库 URDF/配置。"""
    if not (0 < timestep <= 0.005):
        raise ValueError("MuJoCo 内部时间步应在 (0, 0.005] 秒")
    if abs(0.005 / timestep - round(0.005 / timestep)) > 1e-8:
        raise ValueError("内部时间步必须整除 SLR 的 5 ms PD 周期")
    urdf_path = (Path(urdf_path) if urdf_path is not None else
                 ROOT / "resources/race/urdf/race.urdf")
    source = ET.parse(urdf_path).getroot()
    links = {link.get("name"): link for link in source.findall("link")}
    joints = source.findall("joint")
    children = {name: [] for name in links}
    child_names = set()
    for joint in joints:
        children[joint.find("parent").get("link")].append(joint)
        child_names.add(joint.find("child").get("link"))
    root_names = set(links) - child_names
    if root_names != {"base_link"}:
        raise ValueError("Race 根连杆不是唯一的 base_link")
    config_path = ROOT / "deploy/deploy_mujoco/configs/race.yaml"
    cfg = (deepcopy(control_cfg) if control_cfg is not None else
           yaml.safe_load(config_path.read_text(encoding="utf-8")))
    mjcf = ET.Element("mujoco", model="simtosim_race")
    ET.SubElement(mjcf, "compiler", angle="radian", autolimits="true",
                  inertiafromgeom="false", fusestatic="false")
    ET.SubElement(mjcf, "option", timestep=str(timestep),
                  gravity=numbers(cfg["gravity"]), integrator="implicitfast",
                  solver="Newton", iterations="50", tolerance="1e-10",
                  cone="pyramidal", jacobian="auto")
    ET.SubElement(mjcf, "size", nconmax="1500", njmax="6000")
    asset = ET.SubElement(mjcf, "asset")
    world = ET.SubElement(mjcf, "worldbody")
    floor_visual = _scene_appearance(mjcf, asset, world)
    metadata = _terrain(world, terrain, floor_visual)
    actuators = ET.SubElement(mjcf, "actuator")
    efforts = {}
    collision_count = [0]

    def add_link(parent, name, parent_joint=None):
        link = links[name]
        xyz, rpy = origin(parent_joint) if parent_joint is not None else (
            np.asarray(cfg["base_init_pos"]) + [0., 0., metadata["origin_z"]],
            np.zeros(3))
        body = ET.SubElement(parent, "body", name=name, pos=numbers(xyz),
                             quat=numbers(rpy_quaternion(rpy)))
        inertial = link.find("inertial")
        if inertial is not None:
            ipos, irpy = origin(inertial)
            values = inertial.find("inertia").attrib
            inertia = np.array([
                [float(values["ixx"]), float(values["ixy"]),
                 float(values["ixz"])],
                [float(values["ixy"]), float(values["iyy"]),
                 float(values["iyz"])],
                [float(values["ixz"]), float(values["iyz"]),
                 float(values["izz"])],
            ])
            rot = rpy_rotation(irpy)
            inertia = rot @ inertia @ rot.T
            ET.SubElement(body, "inertial", pos=numbers(ipos),
                          mass=inertial.find("mass").get("value"),
                          fullinertia=numbers(inertia[[0, 1, 2, 0, 0, 1],
                                                     [0, 1, 2, 1, 2, 2]]))
        if parent_joint is None:
            ET.SubElement(body, "freejoint", name="floating_base_joint")
        elif parent_joint.get("type") != "fixed":
            joint_name = parent_joint.get("name")
            if joint_name not in cfg["joint_names"]:
                raise ValueError("出现非策略驱动关节：" + joint_name)
            limit = parent_joint.find("limit")
            efforts[joint_name] = float(limit.get("effort"))
            ET.SubElement(
                body, "joint", name=joint_name, type="hinge", pos="0 0 0",
                axis=parent_joint.find("axis").get("xyz"), limited="true",
                range="{} {}".format(limit.get("lower"), limit.get("upper")),
                damping="0", frictionloss="0",
                armature=str(_parameter(joint_name, dict(zip(cfg["joint_names"], cfg["armature"])))))
            ET.SubElement(actuators, "motor", name="motor_" + joint_name,
                          joint=joint_name, gear="1", ctrllimited="false",
                          forcelimited="false")
        for i, visual in enumerate(link.findall("visual")):
            geom = visual.find("geometry/mesh")
            if geom is None:
                continue
            mesh_name = "visual_{}_{}".format(name, i)
            path = (urdf_path.parent / geom.get("filename")).resolve()
            ET.SubElement(asset, "mesh", name=mesh_name, file="../meshes/" + path.name,
                          scale=geom.get("scale", "1 1 1"))
            vpos, vrpy = origin(visual)
            ET.SubElement(body, "geom", name=mesh_name, type="mesh",
                          mesh=mesh_name, pos=numbers(vpos),
                          quat=numbers(rpy_quaternion(vrpy)),
                          contype="0", conaffinity="0", group="2",
                          density="0", rgba="0.75 0.75 0.75 1")
        for i, collision in enumerate(link.findall("collision")):
            cpos, crpy = origin(collision)
            shape = list(collision.find("geometry"))[0]
            attrs = dict(pos=numbers(cpos), quat=numbers(rpy_quaternion(crpy)),
                         contype="2", conaffinity="3",
                         condim="3", group="1", density="0",
                         friction="1 0.005 0.0001", solref="0.004 1",
                         solimp="0.95 0.99 0.001", margin="0",
                         rgba="0.2 0.7 0.8 0.5")
            geom_name = "collision_{}_{}".format(name, i)
            if shape.tag == "box":
                attrs.update(type="box", size=numbers(
                    np.fromstring(shape.get("size"), sep=" ") / 2))
            elif shape.tag == "sphere":
                attrs.update(type="sphere", size=shape.get("radius"))
            elif shape.tag == "cylinder":
                attrs.update(type="cylinder", size=numbers([
                    float(shape.get("radius")), float(shape.get("length")) / 2]))
            else:
                raise ValueError("未实现 URDF 几何：" + shape.tag)
            ET.SubElement(body, "geom", name=geom_name, **attrs)
            collision_count[0] += 1
        for child_joint in children[name]:
            add_link(body, child_joint.find("child").get("link"), child_joint)

    add_link(world, "base_link")
    output = Path(output) if output else ROOT / "resources/race/mjcf/race_pd.xml"
    # 网格路径相对于模型文件，移植后无需访问原 SLR 目录。
    if output.parent.resolve() != (ROOT / "resources/race/mjcf").resolve():
        raise ValueError("输出应保存在 resources/race/mjcf，以保持网格相对路径")
    ET.indent(mjcf, space="  ")
    xml = ET.tostring(mjcf, encoding="unicode")
    output.write_text(xml + "\n", encoding="utf-8")
    model = mujoco.MjModel.from_xml_path(str(output))
    metadata.update(
        base_name="base_link", joint_names=list(cfg["joint_names"]),
        torque_limits_policy=[efforts[n] for n in cfg["joint_names"]],
        urdf=str(urdf_path.relative_to(ROOT)),
        urdf_sha256=hashlib.sha256(urdf_path.read_bytes()).hexdigest(),
        self_collisions=True,
        collision_geoms=collision_count[0],
        control_dt=cfg["pd_dt"], pd_dt=cfg["pd_dt"],
        physics_dt=float(timestep), friction=1.0,
        total_mass=float(model.body_mass.sum()), mjcf_xml=xml,
        contact_note="MuJoCo 接触求解与 PhysX 不同，不声称数值等价。")
    if model.nu != 12 or len(efforts) != 12:
        raise ValueError("MuJoCo 驱动关节数量应为 12")
    if collision_count[0] != 26:
        raise ValueError("仅保留 26 碰撞体 Race，实际数量不符")
    return model, metadata


if __name__ == "__main__":
    model, metadata = build_model()
    print("已生成：resources/race/mjcf/race_pd.xml")
    print(f"主动关节：{model.nu}；碰撞体：{metadata['collision_geoms']}；"
          f"总质量：{metadata['total_mass']:.6f} kg")
