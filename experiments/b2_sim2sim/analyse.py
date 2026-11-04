"""Summarise the completed diagnostic matrix, preserving every trial."""
from pathlib import Path
import json,hashlib,sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'diagnostics/b2_sim2sim_20261103'

def main():
    groups={};allrows=[]
    for backend in ['mujoco','mjlab','bullet']:
        folder=BASE/('bullet_com_corrected' if backend=='bullet' else 'paired_v1')
        assert (folder/f'{backend}_DONE.json').exists(),backend
        paths=sorted((folder/backend).glob('*/result.json'));assert len(paths)==14
        rows=[json.loads(p.read_text()) for p in paths]
        trials=[t for r in rows for t in r['trials']];assert len(trials)==56
        groups[backend]=dict(trials=56,conditions=14,
            **{k:sum(t[k] for t in trials) for k in ['completed','cycle','task_pass','actuator_pass','joint_pass']},
            speed_min_peak=min(t['peak_speed_rad_s'] for t in trials),speed_max_peak=max(t['peak_speed_rad_s'] for t in trials),
            power_min_peak=min(t['peak_power_w'] for t in trials),power_max_peak=max(t['peak_power_w'] for t in trials),
            speed_failed=sum(t['peak_speed_rad_s']>6 for t in trials),power_failed=sum(t['peak_power_w']>1501 for t in trials),
            cases=[dict(kg=r['kg'],case=r['case']['name'],**r['counts'],
                  speed_max=max(t['peak_speed_rad_s'] for t in r['trials']),power_max=max(t['peak_power_w'] for t in r['trials'])) for r in rows])
        for p,r in zip(paths,rows):
            allrows.append(dict(backend=backend,case=r['case'],kg=r['kg'],result=str(p),
                                result_sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    summary=dict(model='B2 C1 actor iteration 2000',weights_sha256='47e2a0ee384147c3682ce6c260d5d08985f8ea2e46fbdb358b5428e1ee91950c',
                 backends=groups,trial_count=168,case_count=42,training=False,hardware_connected=False,
                 full_acceptance=False,sources=allrows,
                 exclusions=['smoke and smoke_v2 were harness development; not physical evidence.',
                             'paired_v1 Bullet was superseded by bullet_com_corrected for consistent COM push.'],
                 limitations=['MuJoCo CPU and mjlab use the same MuJoCo physics family.',
                              'Bullet lacks the 0.1 generalized armature; no contact-compliance identification was done.',
                              'Measured initial joint/IMU state, with floor-aligned height and unmeasured root linear velocity set to zero.',
                              'Perfect simulated proprioception and root position; real odometry/noise errors are not reproduced.',
                              'No manufacturer 0x20 protection model or driver overcurrent/voltage model.',
                              'Four fixed trials per case; not full 48/96 conditions x 64 trials.'])
    (BASE/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['B2 C1 第 2000 次：冻结策略 sim2sim 验证结果','',
           '范围：0 kg / 5 kg；7 条件/负载，每条件 4 次，共 56 次/后端，三后端共 168 次。',
           '条件：参考初姿、实测初姿、摩擦 0.4、质量 1.1 倍、16 N 前向扰动（1.5–1.8 秒）、20 ms 目标延迟、上述组合。',
           '策略周期 20 ms、物理/PD 周期 2 ms、原 0.5 秒接管混合、完整 15.5 秒起立/保持/下降/恢复。',
           '模型、目标、PD/力矩与 1200 W 请求限幅冻结。未训练、未连接机器人、未改奖励或降低验收线。','',
           '结果（动作通过包含姿态、站稳/恢复 COM 加速度、速度、漂移等原判据；执行器单独要求速度≤6 rad/s、机械功率≤1501 W）：']
    for b,g in groups.items():
        lines.append(f'{b}: 未触发原终止判据 {g["completed"]}/56，完成规定姿态周期 {g["cycle"]}/56，动作严格通过 {g["task_pass"]}/56；执行器通过 {g["actuator_pass"]}/56，同时通过 {g["joint_pass"]}/56。')
        lines.append(f'  每次峰值速度范围 {g["speed_min_peak"]:.3f}–{g["speed_max_peak"]:.3f} rad/s；峰值功率 {g["power_min_peak"]:.1f}–{g["power_max_peak"]:.1f} W；速度不通过 {g["speed_failed"]}/56、功率不通过 {g["power_failed"]}/56。')
    lines += ['', '逐条件（每格分母 4）：','后端 | kg | 条件 | 完成姿态周期 | 动作严格 | 执行器 | 同时满足']
    for b,g in groups.items():
        for c in g['cases']:lines.append(f'{b} | {c["kg"]} | {c["case"]} | {c["cycle"]}/4 | {c["task_pass"]}/4 | {c["actuator_pass"]}/4 | {c["joint_pass"]}/4')
    lines += ['', '验证与解释边界：',
              '1. Actor 256 组输入对照原模型，最大误差 1.55e-6；原限幅 1024 组对照通过；电机顺序检查通过。评分器对原两组 64 次轨迹逐试次一致。',
              '2. Bullet 的关节刚体质量矩阵与 MuJoCo 去除 armature 后最大差异约 1.49e-8；质量及几何 FK 一致。但 Bullet 没有等价保留 generalized armature，接触柔顺性/求解器和 PD 积分仍不同。',
              '3. CPU MuJoCo 与 mjlab/Warp 属于同一物理家族，不能把它们说成两个独立物理理论的验证。',
              '4. 实测初姿来自尝试 1793687162374260969 的首个非负时刻。真实根高度未可靠测得，明确采用最低脚球触地对齐；根线速度设为零，不是逐状态复现整台实机。',
              '5. 模型名义质量为 83.4983 kg / 88.4983 kg，来自现有 MJCF；不是对此台机器实测称重。仿真观测无实机噪声/里程计误差。',
              '6. 本次没有建立未知 0x20 保护逻辑，不能用仿真继续出力解释固件为何在约 1.835 秒停止出力。',
              '7. 每条件只有 4 次，属于诊断而非完整离线验收；168 次结果不能替代原完整条件、每条件 64 次验收。',
              '', '结论：动作可完成与执行器达标是两回事。多个后端均显示当前冻结模型缺少满足既定速度/功率条件的证据；主要矛盾已经在仿真中可见，不能只归因为实机保护，也不能据此解除保护。',
              '本次不更换 Actor、不重训、不修改实机控制。后续可以先用该套对照隔离起立参考节奏、PD/执行器模型和初姿/延迟，再决定如何修正训练。',
              '', '视频：paired_v1/videos/，三列分别对应真实计算所得 CPU MuJoCo、mjlab/Warp、Bullet 轨迹；统一 MuJoCo 渲染器仅用于画面展示，未重新模拟或替换失败样本。',
              '固定展示 trial 0，另提供 combined 失败条件。全部原始轨迹、每个 2 ms 子步、初态、模型哈希和代码版本已保留。',
              '', '官方资料：',
              'https://github.com/mujocolab/mjlab',
              'https://mujoco.readthedocs.io/en/stable/mjwarp/index.html',
              'https://github.com/bulletphysics/bullet3']
    (BASE/'REPORT.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps(groups,indent=2))

if __name__=='__main__':main()
