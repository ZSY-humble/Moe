"""绘制 CAD 与机身盒的包络差异，核对本地 Go1/Go2 几何间距。"""

import json
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection, LineCollection
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh

from compare_race_collision_geometry import distances
from audit_race_cad_sections import clip_segments_z

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'logs/race_cad_audit'


def clip_triangle_height(triangle, lower=-.097, upper=.06):
    """裁剪表面三角形的高度带，避免依赖实体封口或额外几何库。"""
    polygon = list(triangle)
    for boundary, sign in [(lower, 1), (upper, -1)]:
        clipped = []
        for index, start in enumerate(polygon):
            end = polygon[(index + 1) % len(polygon)]
            inside_start = sign * (start[2] - boundary) >= 0
            inside_end = sign * (end[2] - boundary) >= 0
            if inside_start:
                clipped.append(start)
            if inside_start != inside_end:
                ratio = (boundary - start[2]) / (end[2] - start[2])
                clipped.append(start + ratio * (end - start))
        polygon = clipped
    return np.asarray(polygon).reshape(-1, 3)


def origin_matrix(element):
    """URDF 零关节角下的局部齐次变换。"""
    result = np.eye(4)
    if element is not None:
        result[:3, 3] = np.fromstring(element.get('xyz', '0 0 0'), sep=' ')
        result[:3, :3] = Rotation.from_euler(
            'xyz', np.fromstring(element.get('rpy', '0 0 0'), sep=' ')
        ).as_matrix()
    return result


def compare_reference(path, base_name, thigh_name):
    """仅查询零角度几何间距，静态 MJCF 不用于动力学。"""
    urdf = ET.parse(path).getroot()
    parents = {j.find('child').get('link'): j for j in urdf.findall('joint')}

    def frame(name):
        if name not in parents:
            return np.eye(4)
        joint = parents[name]
        return frame(joint.find('parent').get('link')) @ origin_matrix(
            joint.find('origin'))

    xml = ET.Element('mujoco')
    world = ET.SubElement(xml, 'worldbody')
    sizes = []
    for index, name in enumerate((base_name, thigh_name)):
        link = urdf.find(f"link[@name='{name}']")
        collision = link.find('collision')
        size = np.fromstring(collision.find('geometry/box').get('size'), sep=' ')
        transform = frame(name) @ origin_matrix(collision.find('origin'))
        x, y, z, w = Rotation.from_matrix(transform[:3, :3]).as_quat()
        body = ET.SubElement(world, 'body', name=name)
        ET.SubElement(body, 'geom', name=f'box_{index}', type='box',
                      size=' '.join(map(str, size / 2)),
                      pos=' '.join(map(str, transform[:3, 3])),
                      quat=' '.join(map(str, (w, x, y, z))))
        sizes.append(size)
    model = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding='unicode'))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return {
        'source_urdf': str(Path(path).resolve()),
        'reference_pose': '所有主动关节零角度；仅几何参考，不是默认站姿',
        'base_box_size_mm': (sizes[0] * 1000).tolist(),
        'right_front_thigh_box_size_mm': (sizes[1] * 1000).tolist(),
        'right_front_thigh_box_to_base_mm': float(
            distances(model, data, 0, [1])[0] * 1000),
        'note': '不同机器人的尺寸与结构不同，间距不能直接用作Race目标。',
    }


def main():
    font = FontProperties(
        fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    plt.rcParams.update({'font.family': font.get_name(), 'font.size': 10,
                         'axes.unicode_minus': False})
    mesh = trimesh.load_mesh(ROOT / 'resources/race/meshes/base_link.stl')
    polygons = [clip_triangle_height(triangle) for triangle in mesh.triangles]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6),
                             gridspec_kw={'width_ratios': [1.8, 1]})
    ax = axes[0]
    ax.add_patch(Rectangle((-249, -103), 518, 206,
                           facecolor='#fbe7e4', edgecolor='#b53b32',
                           linewidth=1.8, label='当前机身碰撞盒'))
    ax.add_collection(PolyCollection(
        [p[:, :2] * 1000 for p in polygons if len(p) >= 3],
        facecolor='#626e78',
        edgecolors='none', rasterized=True, label='同高度带 CAD 投影'))
    for x in [-182.5, 182.5]:
        ax.axvline(x, color='#263746', linestyle='--', linewidth=.9)
    ax.plot(-249, -103, 'o', color='#b53b32', markersize=6)
    ax.annotate('记录中的盒角接触位置', (-249, -103), (-235, -142),
                arrowprops={'arrowstyle': '->', 'color': '#b53b32'})
    ax.set(xlim=(-280, 295), ylim=(-165, 155), aspect='equal',
           xlabel='纵向 X（mm）', ylabel='横向 Y（mm）',
           title='Race：全长宽盒填入前后端的空隙')
    ax.legend(loc='upper center', frameon=False, ncol=2)
    ax = axes[1]
    cross = trimesh.intersections.mesh_plane(
        mesh, [1, 0, 0], [.1825, 0, 0])
    cross = clip_segments_z(cross, -.097, .06).reshape(-1, 2, 3)
    ax.add_patch(Rectangle((-103, -97), 206, 157,
                           facecolor='#fbe7e4', edgecolor='#b53b32',
                           linewidth=1.8, label='碰撞盒截面'))
    ax.add_collection(LineCollection(cross[:, :, 1:3] * 1000,
                                     colors='#263746', linewidths=1.2,
                                     label='CAD 截面交线'))
    ax.annotate('盒宽 206 mm', (0, 43), ha='center')
    ax.annotate('CAD 截面宽约 106 mm', (0, -115), ha='center')
    ax.set(xlim=(-125, 125), ylim=(-130, 80), aspect='equal',
           xlabel='横向 Y（mm）', ylabel='高度 Z（mm）',
           title='前髋位置 X = +182.5 mm')
    fig.suptitle('虚接触来源：机身用一个大盒，局部轮廓却明显收窄',
                 fontsize=14)
    fig.text(.02, .015,
             '来源：Race 原始 STL 与当前 URDF；仅分析 Z∈[-97,60] mm。'
             '灰色为 CAD 表面投影/交线，非封闭实体重建。', fontsize=9)
    fig.tight_layout(rect=(0, .10, 1, .94))
    fig.savefig(OUT / 'base_envelope.png', dpi=180)
    plt.close(fig)
    report = {
        'Go2': compare_reference(ROOT / 'resources/go2/urdf/go2.urdf',
                                  'base', 'FR_thigh'),
        'Go1': compare_reference(Path('/home/zsy/project/walk-these-ways/'
                                      'resources/robots/go1/urdf/go1.urdf'),
                                  'trunk', 'FR_thigh'),
        'Race': compare_reference(ROOT / 'resources/race/urdf/race.urdf',
                                   'base_link', 'pitch01_leg_f_r_link'),
    }
    (OUT / 'unitree_comparison.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(OUT / 'base_envelope.png')


if __name__ == '__main__':
    main()
