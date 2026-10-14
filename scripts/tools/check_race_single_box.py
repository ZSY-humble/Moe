"""核验单盒模型只改机身尺寸，并汇总PhysX对照与默认训练配置验收。"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'logs/race_single_box'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--include_default', action='store_true')
    args = parser.parse_args()
    geometry = json.loads((OUT/'geometry.json').read_text())
    source = Path(geometry['source_urdf'])
    target = Path(geometry['training_urdf'])
    payload_path = Path(geometry['json_path'])
    payload = json.loads(payload_path.read_text())
    digest = hashlib.sha256(payload_path.read_bytes()).hexdigest()
    assert digest == geometry['json_sha256']
    assert hashlib.sha256(target.read_bytes()).hexdigest() == (
        geometry['training_urdf_sha256'])
    assert hashlib.sha256(source.read_bytes()).hexdigest() == (
        payload['source_urdf_sha256'])
    old, new = ET.parse(source).getroot(), ET.parse(target).getroot()
    new_box = new.find("link[@name='base_link']/collision/geometry/box")
    assert new_box.get('size') == '0.42000000 0.10000000 0.10000000'
    new_box.set('size', old.find(
        "link[@name='base_link']/collision/geometry/box").get('size'))
    assert ET.tostring(old) == ET.tostring(new)
    logs = OUT/'physx_logs'
    logs.mkdir(exist_ok=True)
    names = ['single_seed42', 'single_seed43', 'single_seed44',
             'single_uniform_003', 'single_zero_10s', 'single_probes',
             'probe_diagnosis']
    if args.include_default:
        names.append('default_seed42')
    reports = {}
    for name in names:
        path = OUT/(name+'.json')
        report = json.loads(path.read_text())
        log_name = 'single_probe_diagnosis' if name == 'probe_diagnosis' else name
        log = Path('/tmp')/('race_'+log_name+'.log')
        content = log.read_text(errors='replace')
        assert '[诊断完成]' in content, name
        assert not re.search(r'Traceback|CUDA error|PhysX error|'
                             r'failed to allocate|GPU error|eOUT_OF_MEMORY',
                             content), name
        assert log.stat().st_mtime >= path.stat().st_mtime
        shutil.copyfile(log, logs/log.name)
        metadata = report['candidate_metadata']
        assert metadata['source_sha256'] == digest
        assert metadata['actual_usd_shape_and_dimensions_verified']
        assert metadata['count'] == 26
        assert report['base_hip_housing_filter_enabled']
        filters = report['collision_filter_relationships']
        assert len(filters) == 1 and len(filters[0]['targets']) == 4
        if name.startswith('default_'):
            assert metadata['verification_only']
            assert Path(metadata['active_asset_path']).resolve() == target
        else:
            assert metadata['inertials_and_joints_unchanged']
        reports[name] = report
    rows = []
    for seed in [42, 43, 44]:
        baseline = json.loads((ROOT/'logs/race_collision_fix'/
                               f'filtered_seed{seed}.json').read_text())
        current = reports[f'single_seed{seed}']
        for key in ['seed', 'num_envs', 'steps', 'action_std', 'action_seed',
                    'action_distribution', 'total_environment_steps',
                    'training_randomization_enabled', 'self_collisions']:
            assert baseline[key] == current[key], (seed, key)
        assert current['self_collisions']
        assert current['termination_counts']['time_out'] == 0
        rows.append({'seed': seed,
                     'original_large_box': baseline['termination_counts'][
                         'illegal_contact'],
                     'new_single_box': current['termination_counts'][
                         'illegal_contact']})
    assert reports['single_uniform_003']['joint_delta_bound_rad'] == .03
    assert reports['single_uniform_003']['completed_episodes'] == 0
    zero = reports['single_zero_10s']
    assert zero['self_collisions']
    assert zero['last_step']['time_s'] == 10.
    assert zero['first_base_contact_over_1n'] is None
    assert all(.33 < h < .35 for h in zero['last_step']['base_height_m'])
    probes = reports['single_probes']
    for name, force, terminated in zip(
            probes['env_case_order'], probes['max_base_contact_n_per_env'],
            probes['any_illegal_contact_per_env']):
        # 倒置低位样本由腿部先受力撑起，机身净接触力为零，不满足原终止条件。
        expected = name == 'base_side_ground'
        assert terminated == expected
        assert (force > 1.) == expected
    diagnosis = reports['probe_diagnosis']
    for name, force, terminated in zip(
            diagnosis['env_case_order'],
            diagnosis['max_base_contact_n_per_env'],
            diagnosis['any_illegal_contact_per_env']):
        expected = name in ['base_side_ground', 'base_upright_ground',
                            'base_roll30_ground', 'base_inverted_low',
                            'base_pitch90_ground']
        assert terminated == expected, name
        assert (force > 1.) == expected, name
    baseline_total = sum(r['original_large_box'] for r in rows)
    total = sum(r['new_single_box'] for r in rows)
    config = None
    if args.include_default:
        config = json.loads((OUT/'original_config_audit.json').read_text())
        assert config['passed'] and not config['unexpected_training_differences']
        assert config['default_num_envs'] == 16384
        assert config['save_interval'] == 1000
    report = {
        'verified_at_utc': datetime.now(timezone.utc).isoformat(),
        'local_single_box_checks_passed': True,
        'only_base_box_dimensions_changed': True,
        'mechanical_checks': geometry['mechanical_checks'],
        'geometry': geometry,
        'stress_test': {'rows': rows, 'original_total': baseline_total,
                        'new_total': total,
                        'reduction_fraction': 1-total/baseline_total,
                        'scope': '32环境×200步，每20ms独立高斯动作N(0,1)，缩放0.25rad；没有策略学习。'},
        'zero_action_10s': zero['last_step'],
        'small_action_termination_count': 0,
        'ground_contact_probes': {
            'labels': diagnosis['env_case_order'],
            'max_base_contact_n': diagnosis['max_base_contact_n_per_env'],
            'terminated': diagnosis['any_illegal_contact_per_env'],
            'inverted_case_without_base_force': 'base_ground根高度10mm倒置时，腿部先承受约4893N探针载荷并撑起，机身净接触力为零，不会触发原机身接触终止；更低根高度−15mm倒置时机身接触424N并终止。探针是故意重叠测试，力不是正常载荷。'},
        'default_training_asset_verified': args.include_default,
        'original_training_rules_audit_passed': bool(config),
        'legacy_urdf_preserved': True,
        'ppo_or_stair_success_tested': False,
        'limitations': ['比例参考的是视觉范围，Go2视觉含头部而Race主体排除顶部附件，不是严格结构相似缩放。',
                        '主动欠包络可能使视觉外壳先于碰撞盒接触障碍；此处仅验证简化模型行为。',
                        '原终止规则只检查机身净接触力，倒置而由腿部支撑的姿态可能不立即终止，未新增姿态终止。',
                        '压力测试的终止次数不代表训练成功率。'],
    }
    (OUT/'verification.json').write_text(json.dumps(
        report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'passed': True, 'stress_test': report['stress_test'],
                      'default_training_asset_verified': args.include_default},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
