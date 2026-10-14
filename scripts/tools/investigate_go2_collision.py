"""核对 Go2 视觉网格、碰撞盒与腿部间隙，输出可复现的局部几何证据。"""

import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
import trimesh

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'logs/race_bodyclearance_v3'
NS = {'c': 'http://www.collada.org/2005/11/COLLADASchema'}


def read_dae(path):
    """读取本仓库单节点 COLLADA，应用场景矩阵后再比较坐标。"""
    xml = ET.parse(path).getroot()
    geometries = xml.findall('.//c:library_geometries/c:geometry', NS)
    nodes = xml.findall('.//c:visual_scene/c:node', NS)
    assert len(geometries) == len(nodes) == 1
    geometry, node = geometries[0], nodes[0]
    assert node.find('c:instance_geometry', NS).get('url') == (
        '#' + geometry.get('id'))
    assert float(xml.find('c:asset/c:unit', NS).get('meter')) == 1.
    mesh = geometry.find('c:mesh', NS)
    source = mesh.find(
        'c:vertices/c:input[@semantic="POSITION"]', NS).get('source')[1:]
    positions = np.fromstring(mesh.find(
        f'c:source[@id="{source}"]/c:float_array', NS).text,
        sep=' ').reshape(-1, 3)
    faces = []
    for triangles in mesh.findall('c:triangles', NS):
        inputs = triangles.findall('c:input', NS)
        stride = max(int(i.get('offset')) for i in inputs) + 1
        offset = int(next(i for i in inputs
                          if i.get('semantic') == 'VERTEX').get('offset'))
        indices = np.fromstring(triangles.find('c:p', NS).text,
                                sep=' ', dtype=int).reshape(-1, stride)
        faces.append(indices[:, offset].reshape(-1, 3))
    transform = np.fromstring(node.find('c:matrix', NS).text,
                              sep=' ').reshape(4, 4)
    positions = (np.c_[positions, np.ones(len(positions))] @ transform.T)[:, :3]
    return trimesh.Trimesh(positions, np.concatenate(faces), process=False)


def box_gap(center, rotation, half, other_center, other_rotation, other_half):
    """分离轴法：负值证明重叠，正值只表示分离轴上的间隙。"""
    axes = np.vstack((rotation.T, other_rotation.T, np.cross(
        rotation.T[:, None, :], other_rotation.T[None, :, :]).reshape(-1, 3)))
    norms = np.linalg.norm(axes, axis=1)
    axes = axes[norms > 1e-12] / norms[norms > 1e-12, None]
    gaps = (np.abs(axes @ (other_center-center))
            - np.abs(axes @ rotation) @ half
            - np.abs(axes @ other_rotation) @ other_half)
    return float(gaps.max())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    xml = ET.parse(ROOT / 'resources/go2/urdf/go2.urdf').getroot()
    base = xml.find("link[@name='base']")
    box = np.fromstring(base.find('collision/geometry/box').get('size'), sep=' ')
    assert base.find('visual/origin').get('xyz') == '0 0 0'
    assert base.find('visual/origin').get('rpy') == '0 0 0'
    mesh = read_dae(ROOT / 'resources/go2/meshes/base.dae')
    sections = {}
    for x in [0., .06, .12, .18]:
        points = trimesh.intersections.mesh_plane(
            mesh, [1., 0., 0.], [x, 0., 0.]).reshape(-1, 3)
        sections[str(x)] = {'bounds_m': [points.min(0).tolist(),
                                        points.max(0).tolist()],
                            'width_m': float(np.ptp(points[:, 1]))}
    hip = xml.find("joint[@name='FL_hip_joint']")
    thigh = xml.find("joint[@name='FL_thigh_joint']")
    origin = np.fromstring(hip.find('origin').get('xyz'), sep=' ')
    offset = np.fromstring(thigh.find('origin').get('xyz'), sep=' ')
    collider = xml.find("link[@name='FL_thigh']/collision")
    c = np.fromstring(collider.find('origin').get('xyz'), sep=' ')
    r = Rotation.from_euler('xyz', np.fromstring(
        collider.find('origin').get('rpy'), sep=' ')).as_matrix()
    half = np.fromstring(collider.find('geometry/box').get('size'), sep=' ')/2

    def gap(roll, pitch, body_width=None):
        rr = Rotation.from_rotvec([roll, 0., 0.]).as_matrix()
        rp = Rotation.from_rotvec([0., pitch, 0.]).as_matrix()
        dimensions = box.copy()
        if body_width is not None:
            dimensions[1] = body_width
        return box_gap(np.zeros(3), np.eye(3), dimensions/2,
                       origin+rr @ (offset+rp @ c), rr @ rp @ r, half)

    scans = {'original': [], 'width_only_expanded_to_visual': []}
    for roll in np.linspace(float(hip.find('limit').get('lower')),
                            float(hip.find('limit').get('upper')), 61):
        for pitch in np.linspace(float(thigh.find('limit').get('lower')),
                                 float(thigh.find('limit').get('upper')), 81):
            for name, width in [('original', None),
                                ('width_only_expanded_to_visual', .194)]:
                scans[name].append((gap(roll, pitch, width), roll, pitch))

    def summarize(values):
        worst = min(values)
        return {'sample_count': len(values),
                'overlapping_samples': sum(v[0] < -1e-6 for v in values),
                'minimum_sat_gap_m': worst[0],
                'minimum_gap_pose_rad': {'hip_roll': worst[1],
                                         'thigh_pitch': worst[2]}}
    result = {
        'go2_visual_bounds_m': mesh.bounds.tolist(),
        'go2_base_collision_dimensions_m': box.tolist(),
        'go2_exact_cross_sections': sections,
        'go2_zero_joint_thigh_base_axis_gap_m': gap(0., 0.),
        'go2_front_default_thigh_base_axis_gap_m': gap(.1, .8),
        'go2_thigh_static_joint_limit_grid': {
            'method': '外展61点×髋俯仰81点；左前大腿盒与机身盒，非连续全范围证明。',
            **{name: summarize(values) for name, values in scans.items()}},
        'scope': '只核对机身与左前大腿盒；静态几何，不是策略碰撞率或实机间隙。',
        'intent': '证明原模型存在简化与欠包络；没有作者设计意图的直接证据。',
    }
    (OUT / 'go2_geometry.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    render(mesh, box)


def render(go2_mesh, go2_box):
    """比较相同纵向切面的真实网格与碰撞盒，清楚标记单位。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.font_manager import FontProperties
    from matplotlib.patches import Rectangle
    from audit_race_cad_sections import clip_segments_z

    font = FontProperties(
        fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    plt.rcParams.update({'font.family': font.get_name(), 'font.size': 10,
                         'axes.unicode_minus': False})
    race_mesh = trimesh.load_mesh(ROOT / 'resources/race/meshes/base_link.stl')
    payload = json.loads((ROOT / 'resources/race/collision_candidates/'
                          'mjx_bodyclearance_v3.json').read_text())
    parent = payload['central_box_before']
    groups = [[{'pos': [0, 0, 0], 'size': (go2_box/2).tolist()}], [parent],
              [g for g in payload['geoms']
               if g['name'].startswith('clearance_base_3_')]]
    titles = ['Go2：中央视觉外壳未被完整包住',
              'Race v2：中央长盒填满底部空角',
              'Race v3 候选：下部阶梯收窄']
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5), sharey=True)
    for i, (ax, boxes, title) in enumerate(zip(axes, groups, titles)):
        mesh = go2_mesh if i == 0 else race_mesh
        cross = trimesh.intersections.mesh_plane(mesh, [1, 0, 0], [.06, 0, 0])
        cross = clip_segments_z(cross, -.13, .07).reshape(-1, 2, 3)
        for geom in boxes:
            center, half = np.asarray(geom['pos']), np.asarray(geom['size'])
            ax.add_patch(Rectangle((center[1:]-half[1:])*1000,
                                   half[1]*2000, half[2]*2000,
                                   facecolor='#287a88', edgecolor='#287a88',
                                   alpha=.23))
        ax.add_collection(LineCollection(cross[:, :, 1:]*1000,
                                         colors='#333d45', linewidths=1.1))
        ax.set(title=title, xlim=(-120, 120), ylim=(-115, 80),
               aspect='equal', xlabel='横向 Y（mm）')
        ax.grid(alpha=.15)
    axes[0].set_ylabel('高度 Z（mm）')
    fig.suptitle('同在 X=60 mm 处：灰黑线为视觉 CAD，蓝色为机身碰撞盒',
                 fontsize=13)
    fig.text(.02, .025, '来源：仓库 Go2 DAE/URDF、Race STL 与候选 JSON。'
             'v3 为有意收紧的近似模型，不要求覆盖全部 CAD；正式训练尚未替换。',
             fontsize=9)
    fig.tight_layout(rect=(0, .14, 1, .92))
    fig.savefig(OUT / 'go2_race_cross_sections.png', dpi=170)
    plt.close(fig)


if __name__ == '__main__':
    main()
