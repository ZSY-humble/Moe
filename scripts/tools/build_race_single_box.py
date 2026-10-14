"""保留原 Race 头腿几何，仅将主机身改为按 Go2 比例收紧的单盒。"""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh

from check_race_bodyfit_v2 import clip_triangles
from compare_race_collision_geometry import (
    ROOT, CURRENT_XML, DEFAULT, build_candidate, mechanical_assertions,
    get_indices, distances, set_pose,
)

OUT = ROOT/'logs/race_single_box'
SOURCE = ROOT/'resources/race/urdf/race.urdf'
TARGET = SOURCE.with_name('race_go2_single_box.urdf')
JSON_PATH = ROOT/'resources/race/collision_candidates/go2_single_box.json'
DIMENSIONS = np.array([.420, .100, .100])


def extract_geometries(robot):
    """将URDF几何转为诊断JSON，保留原碰撞名称以复用精确髋壳过滤。"""
    result = []
    for link in robot.findall('link'):
        for collider in link.findall('collision'):
            origin = collider.find('origin')
            xyz = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
            rpy = np.fromstring(origin.get('rpy', '0 0 0'), sep=' ')
            x, y, z, w = Rotation.from_euler('xyz', rpy).as_quat()
            shape = list(collider.find('geometry'))[0]
            if shape.tag == 'box':
                size = np.fromstring(shape.get('size'), sep=' ')/2
            elif shape.tag == 'sphere':
                size = np.array([float(shape.get('radius'))])
            elif shape.tag == 'cylinder':
                size = np.array([float(shape.get('radius')),
                                 float(shape.get('length'))/2])
            else:
                raise ValueError(shape.tag)
            result.append({'body': link.get('name'),
                           'name': collider.get('name'),
                           'type': shape.tag, 'pos': xyz.tolist(),
                           'quat': [w, x, y, z], 'size': size.tolist()})
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tree = ET.parse(SOURCE)
    original = deepcopy(tree.getroot())
    base = tree.getroot().find("link[@name='base_link']")
    assert len(base.findall('collision')) == 1
    base.find('collision/geometry/box').set(
        'size', ' '.join(f'{v:.8f}' for v in DIMENSIONS))
    # 清除唯一允许改变的尺寸后整棵树必须一致，含全部视觉、惯量和关节。
    expected = deepcopy(tree.getroot())
    expected.find("link[@name='base_link']/collision/geometry/box").set(
        'size', original.find(
            "link[@name='base_link']/collision/geometry/box").get('size'))
    assert ET.tostring(expected) == ET.tostring(original)
    # 保留原文件格式，便于直接审查唯一一行尺寸差异。
    text = SOURCE.read_text()
    old_line = '<box size="0.51800000 0.20600000 0.15700000"/>'
    assert text.count(old_line) == 1
    TARGET.write_text(text.replace(old_line,
        '<box size="0.42000000 0.10000000 0.10000000"/>'))
    assert ET.tostring(ET.parse(TARGET).getroot()) == ET.tostring(tree.getroot())
    geoms = extract_geometries(tree.getroot())
    assert len(geoms) == 26
    payload = {
        'schema_version': 1, 'mode': 'go2_single_box',
        'source': '原Race URDF，仅主机身盒改为420×100×100mm；头部与腿部原样保留。',
        'source_urdf_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'training_urdf': 'resources/race/urdf/race_go2_single_box.urdf',
        'contact_filter': '自碰撞开启，仅过滤机身与四个髋壳；原终止规则。',
        'geoms': geoms,
    }
    JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2)+'\n')
    xml = OUT/'same_mechanics_single_box.xml'
    build_candidate(JSON_PATH, xml)
    reference = mujoco.MjModel.from_xml_path(str(CURRENT_XML))
    model = mujoco.MjModel.from_xml_path(str(xml))
    checks = mechanical_assertions(reference, model)
    data = mujoco.MjData(model)
    _, qpos, base_id, legs, hips, _ = get_indices(model)
    set_pose(model, data, qpos, DEFAULT)
    gap = min(distances(model, data, base_id,
                        [g for g in legs[leg] if g != hips[leg]]).min()
              for leg in legs)
    mesh = trimesh.load_mesh(ROOT/'resources/race/meshes/base_link.stl')
    core = clip_triangles(mesh.triangles, 2, .06, True).reshape(-1, 3)
    core_dimensions = np.ptp(core, axis=0)
    go2 = json.loads((ROOT/'logs/go2_envelope/geometry.json').read_text())
    go2_dimensions = np.ptp(go2['visual_base_mesh_bounds_m'], axis=0)
    center = np.array(geoms[0]['pos'])
    report = {
        'source_urdf': str(SOURCE), 'training_urdf': str(TARGET),
        'json_path': str(JSON_PATH),
        'json_sha256': hashlib.sha256(JSON_PATH.read_bytes()).hexdigest(),
        'training_urdf_sha256': hashlib.sha256(TARGET.read_bytes()).hexdigest(),
        'only_base_box_size_changed': True,
        'base_box_count': 1, 'total_collider_count': len(geoms),
        'new_box_dimensions_m': DIMENSIONS.tolist(),
        'box_center_m': center.tolist(),
        'box_bounds_m': [(center-DIMENSIONS/2).tolist(),
                         (center+DIMENSIONS/2).tolist()],
        'race_core_visual_dimensions_m': core_dimensions.tolist(),
        'race_box_to_core_dimension_ratios': (DIMENSIONS/core_dimensions).tolist(),
        'go2_box_to_full_visual_dimension_ratios': (
            np.array([.3762, .0935, .114])/go2_dimensions).tolist(),
        'proportion_assumption': 'Race仅取Z≤60mm机身主体，排除顶部附件；Go2视觉含头部，因此只按大致比例设计，不宣称各功能部位严格相似。',
        'mechanical_checks': checks,
        'default_base_to_leg_gap_m': float(gap),
        'full_cad_coverage_required': False,
    }
    (OUT/'geometry.json').write_text(json.dumps(
        report, ensure_ascii=False, indent=2)+'\n')
    render(mesh, center)
    render_full_robot()
    make_contact_probes(model)
    print(json.dumps({k: report[k] for k in [
        'new_box_dimensions_m', 'race_box_to_core_dimension_ratios',
        'go2_box_to_full_visual_dimension_ratios', 'default_base_to_leg_gap_m',
        'only_base_box_size_changed']}, ensure_ascii=False, indent=2))


def make_contact_probes(model):
    """记录限位内大腿间隙，分别用倒地和侧倒确认机身接触仍会终止。"""
    _, qpos, base, legs, hips, _ = get_indices(model)
    data = mujoco.MjData(model)
    thighs = [g for g in legs['f_r'] if g != hips['f_r'] and
              model.geom_bodyid[g] == model.body('pitch01_leg_f_r_link').id]
    best = (float('inf'), None, None)
    for roll in np.linspace(-.733, .733, 39):
        for pitch in np.linspace(-2.2689, 2.2689, 61):
            pose = DEFAULT.copy()
            pose[3:5] = [roll, pitch]
            set_pose(model, data, qpos, pose)
            gap = float(distances(model, data, base, thighs).min())
            if gap < best[0]:
                best = (gap, float(roll), float(pitch))
    cases = json.loads((ROOT/'logs/race_bodyfit_v2/contact_probes.json').read_text())
    cases['cases'][1] = {
        'name': 'base_side_ground', 'height_m': .025,
        'quat_wxyz': [2**-.5, 2**-.5, 0., 0.]}
    # 新盒上表面仅Z=31.5mm，倒置时将根放到10mm以明确压入地面。
    cases['cases'][2]['height_m'] = .010
    cases['right_front_thigh_grid'] = {
        'grid_samples': 39*61,
        'negative_gap_found': best[0] < -1e-6,
        'minimum_sat_or_surface_gap_m': best[0],
        'scope': '仅右前大腿盒与机身盒，外展/髋俯仰在关节限位内取样；非连续范围证明。'}
    (OUT/'contact_probes.json').write_text(json.dumps(
        cases, ensure_ascii=False, indent=2)+'\n')


def render(mesh, center):
    """与用户看到的Go2图相同，用三视图展示单盒的欠包络。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.font_manager import FontProperties
    from matplotlib.patches import Rectangle

    font = FontProperties(
        fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    plt.rcParams.update({'font.family': font.get_name(), 'font.size': 10,
                         'axes.unicode_minus': False})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8),
                             gridspec_kw={'width_ratios': [1.4, 1.4, 1]})
    for ax, dims, title in zip(axes, [[0, 1], [0, 2], [1, 2]],
                              ['俯视：主体明显收窄', '侧视：长度、高度均收紧',
                               '正视：只用一个机身盒']):
        ax.add_collection(PolyCollection(mesh.triangles[:, :, dims]*1000,
                                         facecolor='#c4c9ce', edgecolor='none',
                                         antialiased=False, rasterized=True))
        c, size = center[dims]*1000, DIMENSIONS[dims]*1000
        ax.add_patch(Rectangle(c-size/2, *size, facecolor='#bb493b',
                               edgecolor='#bb493b', linewidth=1.6, alpha=.3))
        if dims[0] == 0:
            ax.set_xlim(-275, 290)
            ax.set_xlabel('X（mm）  ← 后 / 前 →')
            for x in [-182.5, 182.5]:
                ax.axvline(x, color='#65727a', linestyle='--', linewidth=.7)
        else:
            ax.set_xlim(-135, 135)
            ax.set_xlabel('Y（mm）')
        ax.set_ylim((-130, 130) if dims[1] == 1 else (-130, 190))
        ax.set_ylabel('Y（mm）' if dims[1] == 1 else 'Z（mm）')
        ax.set(title=title, aspect='equal')
        ax.grid(alpha=.14)
    fig.suptitle('Race：单机身盒 420 × 100 × 100 mm，灰色为原始视觉网格',
                 fontsize=14, y=.97)
    fig.text(.025, .04, '仅改变主机身盒尺寸，盒中心仍为(10, 0, −18.5)mm；'
             '头部、腿部、质量惯量和训练规则保持原值。\n'
             '按Go2的大致比例主动简化；不是完整CAD外包络。'
             '虚线为前后髋轴位置；本图只画机身，不含独立头部和腿部。',
             fontsize=9, linespacing=1.5)
    fig.tight_layout(rect=(0, .16, 1, .92), w_pad=2)
    fig.savefig(OUT/'single_box_envelope.png', dpi=170)
    plt.close(fig)


def render_full_robot():
    """补充与Go2相同表达的整狗侧视图，让单盒与头腿的位置关系可见。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from matplotlib.patches import Patch
    from export_go2_collision import origin_transform, urdf_geometry
    from render_go2_envelope import link_poses, collider_mesh, show_projection

    robot = ET.parse(TARGET).getroot()
    angles = {f'{kind}_leg_{leg}_joint': angle
              for leg in ['f_l', 'f_r', 'b_l', 'b_r']
              for kind, angle in zip(['roll', 'pitch01', 'pitch02'],
                                     [0., -.12, .30])}
    poses = link_poses(robot, angles)
    visuals = []
    for link in robot.findall('link'):
        for visual in link.findall('visual'):
            node = visual.find('geometry/mesh')
            mesh = trimesh.load_mesh(TARGET.parent/node.get('filename'))
            mesh.apply_transform(poses[link.get('name')]
                                 @ origin_transform(visual.find('origin')))
            visuals.append(mesh)
    cfg = {'base_init_pos': [0., 0., 0.],
           'base_init_quat': [1., 0., 0., 0.],
           'mujoco_joint_names': list(angles),
           'default_angles': list(angles.values())}
    geometries = urdf_geometry(cfg, urdf_path=TARGET)
    for geom in geometries:
        if geom['link'] == 'base_link':
            geom['link'] = 'base'
        elif geom['link'].startswith('head'):
            geom['link'] = 'Head'
    colliders = [(g, collider_mesh(g)) for g in geometries]
    bounds = np.concatenate([m.bounds for m in visuals]
                            + [m.bounds for _, m in colliders])*1000
    font = FontProperties(
        fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    plt.rcParams.update({'font.family': font.get_name(), 'font.size': 11,
                         'axes.unicode_minus': False})
    fig, ax = plt.subplots(figsize=(11, 6.2))
    show_projection(ax, visuals, colliders, [0, 2])
    ax.set(title='Race 整狗侧视：主机身仅一个 420 × 100 × 100 mm 盒体',
           xlabel='X（mm）  ← 后 / 前 →', ylabel='Z（mm）',
           xlim=(bounds[:, 0].min()-25, bounds[:, 0].max()+25),
           ylim=(bounds[:, 2].min()-25, bounds[:, 2].max()+25))
    fig.legend(handles=[Patch(facecolor='#c4c9ce', label='灰：视觉外壳'),
                        Patch(facecolor='#bb493b', alpha=.5,
                              label='红：单机身盒'),
                        Patch(facecolor='#24759b', alpha=.5,
                              label='蓝：原头部碰撞体'),
                        Patch(facecolor='#438460', alpha=.5,
                              label='绿：原腿部碰撞体')],
               loc='lower center', bbox_to_anchor=(.5, .025),
               ncol=4, frameon=False)
    fig.tight_layout(rect=(0, .10, 1, .98))
    fig.savefig(OUT/'single_box_full_robot.png', dpi=170)
    plt.close(fig)


if __name__ == '__main__':
    main()
