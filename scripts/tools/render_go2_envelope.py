"""按当前 URDF 坐标叠加 Go2 视觉网格与全部碰撞体，生成包围情况图。"""

from collections import Counter
from functools import lru_cache
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Patch
import numpy as np
from scipy.spatial import ConvexHull
import trimesh

from export_go2_collision import (
    axis_rotation, origin_transform, transform, urdf_geometry,
)
from investigate_go2_collision import read_dae

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'logs/go2_envelope'
COLORS = {'body': '#bb493b', 'head': '#24759b', 'leg': '#438460'}


def link_poses(robot, angles):
    """计算关节树正运动学，统一到机身坐标系。"""
    joints = robot.findall('joint')
    roots = {x.get('name') for x in robot.findall('link')} - {
        x.find('child').get('link') for x in joints}
    assert len(roots) == 1
    poses = {roots.pop(): np.eye(4)}
    pending = joints.copy()
    while pending:
        before = len(pending)
        for joint in pending.copy():
            parent = joint.find('parent').get('link')
            if parent not in poses:
                continue
            pose = poses[parent] @ origin_transform(joint.find('origin'))
            if joint.get('type') in ['revolute', 'continuous']:
                axis = np.fromstring(joint.find('axis').get('xyz'), sep=' ')
                rotation = axis_rotation(axis, angles[joint.get('name')])
                pose = pose @ transform([0, 0, 0], rotation)
            poses[joint.find('child').get('link')] = pose
            pending.remove(joint)
        assert len(pending) < before, 'URDF关节树未连通'
    return poses


@lru_cache(maxsize=None)
def mesh_from_dae(path):
    return read_dae(path)


def get_visual_meshes(robot, poses):
    result = {}
    for link in robot.findall('link'):
        for visual in link.findall('visual'):
            node = visual.find('geometry/mesh')
            assert node is not None
            relative = node.get('filename').removeprefix('package://go2/')
            mesh = mesh_from_dae(str(ROOT/'resources/go2'/relative)).copy()
            if node.get('scale'):
                mesh.apply_scale(np.fromstring(node.get('scale'), sep=' '))
            mesh.apply_transform(poses[link.get('name')]
                                 @ origin_transform(visual.find('origin')))
            result[link.get('name')] = mesh
    return result


def collider_mesh(geom):
    size = geom['size']
    if geom['shape'] == 'box':
        mesh = trimesh.creation.box(extents=2*np.asarray(size))
    elif geom['shape'] == 'cylinder':
        mesh = trimesh.creation.cylinder(radius=size[0], height=2*size[1],
                                         sections=64)
    elif geom['shape'] == 'sphere':
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=size[0])
    else:
        raise ValueError(geom['shape'])
    mesh.apply_transform(transform(geom['position'],
                                    np.asarray(geom['rotation']).reshape(3, 3)))
    return mesh


def show_projection(ax, meshes, colliders, dims):
    """绘制真实三角面投影与凸碰撞体投影，避免用视觉外接盒代替外壳。"""
    for mesh in meshes:
        ax.add_collection(PolyCollection(mesh.triangles[:, :, dims]*1000,
                                         facecolor='#c4c9ce', edgecolor='none',
                                         antialiased=False, rasterized=True))
    for geom, mesh in colliders:
        points = mesh.vertices[:, dims]*1000
        hull = ConvexHull(points)
        color = COLORS['body' if geom['link'] == 'base' else
                       'head' if geom['link'].startswith('Head') else 'leg']
        ax.add_collection(PolyCollection([points[hull.vertices]],
                                         facecolor=color, edgecolor=color,
                                         linewidth=1.5, alpha=.23))
        polygon = points[np.r_[hull.vertices, hull.vertices[0]]]
        ax.plot(*polygon.T, color=color, linewidth=1.3)
    ax.grid(alpha=.13)
    ax.set_aspect('equal', adjustable='box')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    xml = ET.parse(ROOT/'resources/go2/urdf/go2.urdf').getroot()
    angles = {}
    for leg in ['FL', 'FR', 'RL', 'RR']:
        angles[f'{leg}_hip_joint'] = .1 if leg.endswith('L') else -.1
        angles[f'{leg}_thigh_joint'] = .8 if leg.startswith('F') else 1.
        angles[f'{leg}_calf_joint'] = -1.5
    cfg = {'base_init_pos': [0., 0., 0.],
           'base_init_quat': [1., 0., 0., 0.],
           'mujoco_joint_names': list(angles),
           'default_angles': list(angles.values())}
    poses = link_poses(xml, angles)
    visuals = get_visual_meshes(xml, poses)
    geometries = urdf_geometry(cfg)
    # 用独立正运动学再次核对全部27个碰撞体的位姿，防止绘制时错位。
    for link in xml.findall('link'):
        for i, collision in enumerate(link.findall('collision')):
            name = collision.get('name') or f'{link.get("name")}_collision_{i}'
            geom = next(g for g in geometries if g['name'] == name)
            pose = poses[link.get('name')] @ origin_transform(
                collision.find('origin'))
            assert np.allclose(pose[:3, 3], geom['position'], atol=1e-7)
            assert np.allclose(pose[:3, :3], np.asarray(geom['rotation']).reshape(
                3, 3), atol=1e-7)
    colliders = [(g, collider_mesh(g)) for g in geometries]
    body_colliders = [(g, m) for g, m in colliders
                      if g['link'] == 'base' or g['link'].startswith('Head')]
    assert len(colliders) == 27 and len(body_colliders) == 3
    base = visuals['base']
    box = next(mesh for geom, mesh in colliders if geom['link'] == 'base')
    report = {
        'source_urdf': 'resources/go2/urdf/go2.urdf',
        'coordinate_frame': '机身坐标系；X正向为前，Y正向为左，Z正向为上。',
        'visual_base_mesh_bounds_m': base.bounds.tolist(),
        'main_box_bounds_m': box.bounds.tolist(),
        'head_colliders': [g for g in geometries if g['link'].startswith('Head')],
        'collider_counts': dict(Counter(g['shape'] for g in geometries)),
        'collider_count': len(colliders), 'visual_link_count': len(visuals),
        'default_joint_angles_rad': angles,
        'fk_cross_checks_passed': True,
        'rear_box_extension_beyond_base_visual_m': float(
            base.bounds[0, 0]-box.bounds[0, 0]),
        'scope': '当前仓库的URDF/DAE几何展示。投影重合不证明三维表面全覆盖；未改训练模型。',
    }
    (OUT/'geometry.json').write_text(json.dumps(
        report, ensure_ascii=False, indent=2)+'\n')

    font = FontProperties(
        fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    plt.rcParams.update({'font.family': font.get_name(), 'font.size': 10,
                         'axes.unicode_minus': False})
    fig, axes = plt.subplots(2, 2, figsize=(14, 10.6))
    show_projection(axes[0, 0], [base], body_colliders, [0, 1])
    axes[0, 0].set(title='俯视：主机身盒明显比视觉外壳窄',
                    xlabel='X（mm）  ← 后 / 前 →', ylabel='Y（mm）',
                    xlim=(-240, 380), ylim=(-130, 130))
    show_projection(axes[0, 1], [base], body_colliders, [0, 2])
    axes[0, 1].set(title='侧视：前端另有圆柱和球形头部碰撞体',
                    xlabel='X（mm）  ← 后 / 前 →', ylabel='Z（mm）',
                    xlim=(-240, 380), ylim=(-130, 130))
    axes[0, 1].annotate('主机身盒后端：−188.1 mm', xy=(-188.1, 0),
                        xytext=(-225, 90), fontsize=9,
                        arrowprops={'arrowstyle': '->', 'color': COLORS['body']})
    axes[0, 1].annotate('独立头部碰撞体', xy=(300, 25), xytext=(160, 90),
                        fontsize=9, arrowprops={'arrowstyle': '->',
                                              'color': COLORS['head']})
    show_projection(axes[1, 0], [base], body_colliders, [1, 2])
    axes[1, 0].set(title='沿X方向看：外壳两侧未被主机身盒包住',
                    xlabel='Y（mm）', ylabel='Z（mm）',
                    xlim=(-145, 145), ylim=(-130, 120))
    axes[1, 0].annotate('主机身盒宽 93.5 mm', xy=(46.75, 57),
                        xytext=(-100, 90), fontsize=10,
                        arrowprops={'arrowstyle': '->', 'color': COLORS['body']})
    show_projection(axes[1, 1], list(visuals.values()), colliders, [0, 2])
    all_bounds = np.concatenate([m.bounds for m in visuals.values()]
                                + [m.bounds for _, m in colliders]) * 1000
    axes[1, 1].set(title='整狗侧视：27个碰撞体，四条腿也各自建模',
                    xlabel='X（mm）  ← 后 / 前 →', ylabel='Z（mm）',
                    xlim=(all_bounds[:, 0].min()-20,
                          all_bounds[:, 0].max()+20),
                    ylim=(all_bounds[:, 2].min()-20,
                          all_bounds[:, 2].max()+30))
    fig.suptitle('Go2 的包围情况：视觉网格与 URDF 碰撞体叠加',
                 fontsize=16, y=.975)
    fig.legend(handles=[Patch(facecolor='#c4c9ce', label='灰：视觉网格'),
                        Patch(facecolor=COLORS['body'], alpha=.6,
                              label='红：主机身长方体'),
                        Patch(facecolor=COLORS['head'], alpha=.6,
                              label='蓝：头部碰撞体'),
                        Patch(facecolor=COLORS['leg'], alpha=.6,
                              label='绿：腿部碰撞体')],
               loc='upper center', bbox_to_anchor=(.5, .94), ncol=4,
               frameon=False)
    fig.text(.03, .02, '数据：当前仓库 resources/go2/urdf/go2.urdf 与 meshes/*.dae。'
             '前三图仅显示 base 视觉网格和机身/头部碰撞体。\n'
             '按URDF原始形状绘制并应用坐标变换；这是投影展示，'
             '不能据此宣称完整三维覆盖。',
             fontsize=9, linespacing=1.6)
    fig.tight_layout(rect=(0, .085, 1, .89), h_pad=5.2, w_pad=2)
    fig.savefig(OUT/'go2_envelope_views.png', dpi=170)
    plt.close(fig)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print('包围情况图：', OUT/'go2_envelope_views.png')


if __name__ == '__main__':
    main()
