"""按训练配置核对 Race 检查点来源，禁止静默加载 Go2 同维度权重。"""

from __future__ import annotations

from collections.abc import Mapping
import math
from pathlib import Path

import numpy as np
import yaml
from yaml.nodes import MappingNode, ScalarNode


JOINT_NAMES = [
    f"{kind}_leg_{leg}_joint"
    for leg in ("f_l", "f_r", "b_l", "b_r")
    for kind in ("roll", "pitch01", "pitch02")
]
OBSERVATION_ORDER = [
    "base_ang_vel", "projected_gravity", "velocity_commands",
    "joint_pos", "joint_vel", "actions",
]
GROUP_FIELDS = {
    "concatenate_terms", "concatenate_dim", "enable_corruption",
    "history_length", "flatten_history_dim",
}


class RaceConfigLoader(yaml.SafeLoader):
    """只为安全加载器增加 IsaacLab 保存的 Python 元组格式。"""


def _construct_tuple(loader, node):
    return tuple(loader.construct_sequence(node, deep=True))


RaceConfigLoader.add_constructor(
    "tag:yaml.org,2002:python/tuple", _construct_tuple
)


def _mapping_nodes(node, location):
    """访问普通映射节点；不构造未使用的 Python 对象标签。"""
    if not isinstance(node, MappingNode) or node.tag != "tag:yaml.org,2002:map":
        raise ValueError(f"Race 检查点配置 {location} 必须为普通映射")
    entries = {}
    for key, value in node.value:
        if not isinstance(key, ScalarNode) or key.tag != "tag:yaml.org,2002:str":
            raise ValueError(f"Race 检查点配置 {location} 的字段名必须为字符串")
        if key.value in entries:
            raise ValueError(f"Race 检查点配置包含重复字段：{location}.{key.value}")
        entries[key.value] = value
    return entries


def _saved_node(root, path):
    node = root
    for index, name in enumerate(path):
        location = ".".join(path[:index]) or "根节点"
        entries = _mapping_nodes(node, location)
        if name not in entries:
            raise ValueError(f"Race 检查点配置缺少必要字段：{'.'.join(path)}")
        node = entries[name]
    return node


def _current_value(root, path):
    value = root
    for name in path:
        try:
            value = value[name] if isinstance(value, Mapping) else getattr(value, name)
        except (KeyError, AttributeError, TypeError) as error:
            raise ValueError(f"当前 Race 配置缺少必要字段：{'.'.join(path)}") from error
    return value


def _number(value, location):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Race 参数 {location} 必须为有限数值")
    if not math.isfinite(float(value)):
        raise ValueError(f"Race 参数 {location} 必须为有限数值")
    return float(value)


def _numbers_equal(saved, current, location):
    try:
        left = np.asarray(saved, dtype=float)
        right = np.asarray(current, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Race 检查点参数 {location} 不是有效数值") from error
    if left.shape != right.shape or not (
        np.isfinite(left).all() and np.isfinite(right).all()
        and np.allclose(left, right, rtol=1e-6, atol=1e-8)
    ):
        raise ValueError(f"Race 检查点与当前配置的 {location} 不一致")


def _joint_mapping_equal(saved, current, location):
    for label, values in (("检查点", saved), ("当前", current)):
        if not isinstance(values, Mapping) or set(values) != set(JOINT_NAMES):
            raise ValueError(
                f"{label} Race 配置 {location} 必须按标准 12 个关节名完整映射"
            )
    for name in JOINT_NAMES:
        _numbers_equal(
            _number(saved[name], f"{location}.{name}"),
            _number(current[name], f"{location}.{name}"),
            f"{location}.{name}",
        )


def _function_name(value):
    if callable(value):
        return f"{value.__module__}:{value.__qualname__}"
    return value


def _plain_config_equal(saved, current, location):
    """比较完整执行器字段，兼容标量参数及逐关节参数两种接口。"""
    if isinstance(saved, Mapping) and isinstance(current, Mapping):
        if set(saved) != set(current):
            raise ValueError(f"Race 检查点与当前 {location} 的字段不一致")
        for name in saved:
            _plain_config_equal(saved[name], current[name], f"{location}.{name}")
    elif isinstance(saved, (tuple, list)) and isinstance(current, (tuple, list)):
        if len(saved) != len(current):
            raise ValueError(f"Race 检查点与当前 {location} 的长度不一致")
        for index, (left, right) in enumerate(zip(saved, current)):
            _plain_config_equal(left, right, f"{location}.{index}")
    elif (isinstance(saved, (int, float)) and not isinstance(saved, bool)
          and isinstance(current, (int, float))
          and not isinstance(current, bool)):
        _numbers_equal(saved, current, location)
    elif _function_name(saved) != _function_name(current):
        raise ValueError(f"Race 检查点与当前 {location} 不一致")


def verify_race_checkpoint(checkpoint_path, env_cfg):
    """核对检查点和当前 Race 接口，允许平地与粗糙地形之间续训。

    只读取检查点旁的训练配置，不反序列化权重。配置中的无关 slice
    标签仅作为 YAML 节点存在；构造器只处理所需字段及安全元组。
    地形、并行数量、随机种子、观测噪声和域随机化不参与接口匹配。
    """
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"找不到 Race 检查点：{checkpoint}")
    config_path = checkpoint.parent / "params/env.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"缺少检查点来源配置 {config_path}；无法确认机器人和控制接口，"
            "请保留原训练目录中的 params/env.yaml。"
        )
    current_cfg = env_cfg.to_dict() if hasattr(env_cfg, "to_dict") else env_cfg
    loader = RaceConfigLoader(config_path.read_text(encoding="utf-8"))
    try:
        root = loader.get_single_node()

        def saved(*path):
            return loader.construct_object(_saved_node(root, path), deep=True)

        def current(*path):
            return _current_value(current_cfg, path)

        action_path = ("actions", "joint_pos")
        for label, names in (
            ("检查点", saved(*action_path, "joint_names")),
            ("当前", current(*action_path, "joint_names")),
        ):
            if names != JOINT_NAMES:
                raise ValueError(
                    f"{label}动作关节不是 Race 标准 12 关节顺序；"
                    "禁止将 Go2 或其他机器人同维度权重加载为 Race。"
                )
        for name in ("preserve_order", "use_default_offset"):
            if saved(*action_path, name) is not True or (
                current(*action_path, name) is not True
            ):
                raise ValueError(f"Race 动作配置 {name} 必须为 true")
        _plain_config_equal(
            saved(*action_path, "scale"), current(*action_path, "scale"),
            "actions.joint_pos.scale",
        )
        saved_clip = saved(*action_path, "clip")
        current_clip = current(*action_path, "clip")
        if not isinstance(saved_clip, Mapping) or not isinstance(current_clip, Mapping):
            raise ValueError("Race 目标角限位必须按关节类型映射")
        if set(saved_clip) != set(current_clip):
            raise ValueError("Race 检查点与当前目标角限位字段不一致")
        for name in saved_clip:
            _numbers_equal(saved_clip[name], current_clip[name], f"目标角限位.{name}")

        pose_path = ("scene", "robot", "init_state", "joint_pos")
        _joint_mapping_equal(saved(*pose_path), current(*pose_path), "默认关节角")
        actuator_path = ("scene", "robot", "actuators")
        _plain_config_equal(
            saved(*actuator_path), current(*actuator_path), "执行器",
        )
        for name in ("pos", "rot"):
            path = ("scene", "robot", "init_state", name)
            _numbers_equal(saved(*path), current(*path), f"初始姿态.{name}")

        for group, expected_history in (("policy", 10), ("single_obs", 1)):
            group_path = ("observations", group)
            if saved(*group_path, "history_length") != expected_history or (
                current(*group_path, "history_length") != expected_history
            ):
                raise ValueError(f"Race {group} 历史长度必须为 {expected_history}")
            entries = _mapping_nodes(_saved_node(root, group_path), f"观测.{group}")
            saved_terms = [name for name in entries if name not in GROUP_FIELDS]
            current_group = current(*group_path)
            if not isinstance(current_group, Mapping):
                current_group = vars(current_group)
            current_terms = [
                name for name in current_group if name not in GROUP_FIELDS
                and not name.startswith("_")
            ]
            if saved_terms != OBSERVATION_ORDER or current_terms != OBSERVATION_ORDER:
                raise ValueError(f"Race {group} 的观测项目或拼接顺序不一致")
            for name in ("concatenate_terms", "flatten_history_dim"):
                if saved(*group_path, name) is not True or (
                    current(*group_path, name) is not True
                ):
                    raise ValueError(f"Race {group}.{name} 必须为 true")
            for term in OBSERVATION_ORDER:
                term_path = (*group_path, term)
                saved_scale = saved(*term_path, "scale")
                current_scale = current(*term_path, "scale")
                _numbers_equal(
                    1.0 if saved_scale is None else saved_scale,
                    1.0 if current_scale is None else current_scale,
                    f"观测.{group}.{term}.scale",
                )
                _numbers_equal(
                    saved(*term_path, "clip"), current(*term_path, "clip"),
                    f"观测.{group}.{term}.clip",
                )
                if _function_name(saved(*term_path, "func")) != (
                    _function_name(current(*term_path, "func"))
                ):
                    raise ValueError(f"Race {group}.{term} 的观测函数不一致")
            for term in ("joint_pos", "joint_vel"):
                entity_path = (*group_path, term, "params", "asset_cfg")
                if saved(*entity_path, "joint_names") != JOINT_NAMES or (
                    current(*entity_path, "joint_names") != JOINT_NAMES
                ):
                    raise ValueError(f"Race {group}.{term} 观测关节顺序不一致")
                if saved(*entity_path, "preserve_order") is not True or (
                    current(*entity_path, "preserve_order") is not True
                ):
                    raise ValueError(f"Race {group}.{term} 必须保留关节顺序")

        _numbers_equal(saved("sim", "dt"), current("sim", "dt"), "仿真周期")
        _numbers_equal(saved("decimation"), current("decimation"), "控制降频倍数")
    except yaml.YAMLError as error:
        raise ValueError(
            f"无法安全解析 Race 来源配置 {config_path}；"
            "只允许标准 YAML 类型和 Python 元组，禁止执行 Python 对象标签。"
        ) from error
    finally:
        loader.dispose()
    print(f"[INFO] Race 检查点来源与控制接口核验通过：{config_path}")
    return {"checkpoint": str(checkpoint), "config": str(config_path), "robot": "race"}
