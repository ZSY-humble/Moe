"""将候选碰撞 JSON 导入独立测试 URDF/USD，保留当前机械与控制参数。"""

from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from scipy.spatial.transform import Rotation

from isaaclab.utils import configclass
from isaaclab.sim.utils import clone
from isaaclab.sim.spawners.from_files.from_files import _spawn_from_usd_file
from robot_lab.assets.race import RaceModelUrdfConverter, RaceModelUrdfFileCfg

ROOT = Path(__file__).resolve().parents[2]


class CandidateUrdfConverter(RaceModelUrdfConverter):
    """把占位圆柱精确改成 USD 胶囊，保留半径与圆柱段长度。"""

    def _convert_asset(self, cfg):
        super()._convert_asset(cfg)
        if not cfg.capsule_names:
            return
        from pxr import Usd, UsdGeom, Vt, Gf

        changed_names = set()
        for path in Path(self.usd_dir).rglob('*.usd'):
            stage = Usd.Stage.Open(str(path))
            changed = False
            for prim in stage.Traverse():
                if prim.IsInstanceProxy() or not prim.IsA(UsdGeom.Cylinder):
                    continue
                name = next((n for n in cfg.capsule_names
                             if n in str(prim.GetPath()).split('/')), None)
                if name is None:
                    continue
                cylinder = UsdGeom.Cylinder(prim)
                radius = cylinder.GetRadiusAttr().Get()
                height = cylinder.GetHeightAttr().Get()
                axis = cylinder.GetAxisAttr().Get()
                prim.SetTypeName('Capsule')
                capsule = UsdGeom.Capsule(prim)
                capsule.CreateRadiusAttr(radius)
                capsule.CreateHeightAttr(height)
                capsule.CreateAxisAttr(axis)
                # USD height 与 MuJoCo 2*size[1] 均指不含半球的圆柱段。
                half_extents = [radius, radius, radius]
                half_extents['XYZ'.index(str(axis).upper())] += height / 2
                extent = [tuple(-v for v in half_extents),
                          tuple(half_extents)]
                capsule.CreateExtentAttr(Vt.Vec3fArray(
                    [Gf.Vec3f(*v) for v in extent]
                ))
                changed_names.add(name)
                changed = True
            if changed:
                stage.GetRootLayer().Save()
        if changed_names != set(cfg.capsule_names):
            raise RuntimeError(f'胶囊转换缺失：{set(cfg.capsule_names)-changed_names}')


@clone
def spawn_candidate(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """使用独立转换器与缓存，不覆盖正式训练资产。"""
    converter = CandidateUrdfConverter(cfg)
    return _spawn_from_usd_file(prim_path, converter.usd_path, cfg,
                                translation, orientation)


@configclass
class CandidateUrdfCfg(RaceModelUrdfFileCfg):
    func = spawn_candidate
    capsule_names: list[str] = []


def configure_candidate(original, path):
    """只替换碰撞；原 URDF 的质量、惯量、关节、轴和限位原样保留。"""
    path = Path(path).resolve()
    payload = json.loads(path.read_text())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    folder = ROOT / 'logs/race_collision_comparison/assets' / (
        path.stem + '_' + digest
    )
    folder.mkdir(parents=True, exist_ok=True)
    source_path = Path(original.asset_path)
    tree = ET.parse(source_path)
    root = tree.getroot()
    links = {link.get('name'): link for link in root.findall('link')}
    inertials_before = {name: ET.tostring(link.find('inertial'))
                        for name, link in links.items()}
    joints_before = [ET.tostring(joint) for joint in root.findall('joint')]
    for mesh in root.findall('.//mesh'):
        mesh.set('filename', str((source_path.parent /
                                 mesh.attrib['filename']).resolve()))
    for link in links.values():
        for collision in list(link.findall('collision')):
            link.remove(collision)
    capsule_names = []
    for geom in payload['geoms']:
        collision = ET.SubElement(links[geom['body']], 'collision',
                                  name=geom['name'])
        w, x, y, z = geom['quat']
        rpy = Rotation.from_quat([x, y, z, w]).as_euler('xyz')
        ET.SubElement(collision, 'origin',
                      xyz=' '.join(map(str, geom['pos'])),
                      rpy=' '.join(map(str, rpy)))
        geometry = ET.SubElement(collision, 'geometry')
        kind, size = geom['type'], geom['size']
        if kind == 'box':
            ET.SubElement(geometry, 'box',
                          size=' '.join(str(2*v) for v in size))
        elif kind == 'sphere':
            ET.SubElement(geometry, 'sphere', radius=str(size[0]))
        elif kind in ('cylinder', 'capsule'):
            ET.SubElement(geometry, 'cylinder', radius=str(size[0]),
                          length=str(2*size[1]))
            if kind == 'capsule':
                capsule_names.append(geom['name'])
        else:
            raise ValueError(f'不支持候选形状：{kind}')
    assert inertials_before == {name: ET.tostring(link.find('inertial'))
                                for name, link in links.items()}
    assert joints_before == [ET.tostring(j) for j in root.findall('joint')]
    urdf = folder / 'candidate.urdf'
    tree.write(urdf, encoding='utf-8', xml_declaration=True)
    cfg = CandidateUrdfCfg()
    for name, value in vars(original).items():
        if name != 'func':
            setattr(cfg, name, deepcopy(value))
    cfg.asset_path = str(urdf)
    cfg.usd_dir = str(folder / 'usd')
    cfg.usd_file_name = 'candidate.usd'
    cfg.force_usd_conversion = True
    cfg.capsule_names = capsule_names
    cfg.replace_cylinders_with_capsules = False
    metadata = {
        'source_json': str(path),
        'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'collider_counts': dict(Counter(g['type'] for g in payload['geoms'])),
        'inertials_and_joints_unchanged': True,
        'candidate_urdf': str(urdf),
        'json_contact_filter': payload.get('contact_filter',
                            payload.get('recipe', {}).get('contact_filter')),
        'comparison_uses_explicit_self_collision_flag': True,
    }
    return cfg, metadata


def verify_candidate(stage, payload_path):
    """核对实际实例中的形状类型和尺寸，避免把胶囊误导成圆柱。"""
    from pxr import Usd, UsdGeom, UsdPhysics
    import numpy as np

    payload = json.loads(Path(payload_path).read_text())
    root = stage.GetPrimAtPath('/World/envs/env_0/Robot')
    colliders = [prim for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies())
                 if prim.HasAPI(UsdPhysics.CollisionAPI)]
    assert len(colliders) == len(payload['geoms'])
    cache = UsdGeom.XformCache()
    for geom in payload['geoms']:
        matches = [prim for prim in colliders
                   if geom['name'] in str(prim.GetPath()).split('/')]
        assert len(matches) == 1, (geom['name'], matches)
        prim = matches[0]
        kind, size = geom['type'], geom['size']
        body = stage.GetPrimAtPath(root.GetPath().AppendChild(geom['body']))
        transform = (cache.GetLocalToWorldTransform(prim)
                     * cache.GetLocalToWorldTransform(body).GetInverse())
        matrix = np.asarray(transform)
        scale = np.linalg.norm(matrix[:3, :3], axis=1)
        assert np.allclose(matrix[3, :3], geom['pos'], atol=1e-6), geom['name']
        w, x, y, z = geom['quat']
        rotation = Rotation.from_quat([x, y, z, w]).as_matrix()
        assert np.allclose((matrix[:3, :3] / scale[:, None]).T,
                           rotation, atol=1e-5), geom['name']
        if kind == 'box':
            assert prim.IsA(UsdGeom.Cube), prim
            actual = UsdGeom.Cube(prim).GetSizeAttr().Get() * scale
            assert np.allclose(actual, 2 * np.asarray(size), atol=1e-6)
        else:
            assert np.allclose(scale, 1., atol=1e-6), geom['name']
            cls = {'sphere': UsdGeom.Sphere, 'cylinder': UsdGeom.Cylinder,
                   'capsule': UsdGeom.Capsule}[kind]
            assert prim.IsA(cls), (geom['name'], prim.GetTypeName())
            obj = cls(prim)
            assert np.isclose(obj.GetRadiusAttr().Get(), size[0])
            if kind != 'sphere':
                assert np.isclose(obj.GetHeightAttr().Get(), 2*size[1])
    return {'actual_usd_shape_and_dimensions_verified': True,
            'count': len(colliders)}
