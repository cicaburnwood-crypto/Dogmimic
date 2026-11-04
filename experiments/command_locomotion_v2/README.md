# 无参考动作 G1 启停：第二轮续训

本轮从第一轮自行训练的最佳 checkpoint（24,000 次迭代）继续学习。它不是新一轮随机参数训练；完整谱系仍然来自无参考动作、无外部预训练模型的第一轮。第一轮结果和文件保留不变。

实际失败原因：1.5 m/s 的 4 个失败全部超出横向位移门限；2 m/s 的 12 个失败全部有横向偏移过大，其中 1 个摔倒。原奖励在机身朝向坐标系衡量速度，而测试要求沿固定方向前进。仅加大原奖励不能可靠修复航向漂移。

本轮调整：

- 使用世界坐标系的目标速度，加入航向、横向位置和横向速度惩罚。
- 目标输入从 3 维扩展为 6 维：速度指令 3 维，加航向正弦、1−航向余弦、横向偏移。机器人因此能观测需要纠正的误差；没有动作轨迹、步态相位或专家姿态。
- 新增输入的初始权重为零。实际 GPU 检查确认扩展后的策略和价值输出与继承模型完全一致。世界模型及其优化器继续使用；策略/价值优化器和回放重新初始化，避免旧目标的动量干扰。
- 30% 指令段为停止；运动段中 60% 聚焦当前上限的 75%–100%，其余保持全速度范围覆盖。
- 初始上限 2 m/s；连续两个 checkpoint 均通过当前速度和全部较低速度后提高 0.5 m/s，封顶 4 m/s。速度上限由独立评估通过原子更新的 curriculum.json 控制。
- 每次 episode 从固定默认姿态、严格零线速度/角速度/关节速度开始。

评估保持第一轮 16 秒“停—前进—停—再前进—停”的验收标准。第二轮使用新随机种子、每个速度 32 个试验，速度集合为 0.3、0.5、1、1.5、2、2.5、3、3.5、4 m/s，并用相同新种子重新测试旧模型。选型按连续通过的速度范围排序，避免只通过某个高速却丢失低速启停能力。

服务器：super_5090。目录：/home/liudiwen/projects/supermimic_vibe。
本轮输出：runs/command_v2_heading；评估：runs/command_v2_heading_eval。
训练 GPU 4、5；评估 GPU 6。计划额外 30,000 次迭代，每 2,000 次独立评估。

本轮计时与第一轮分开保存，复用编译缓存；不能把本轮耗时称为完整从零训练耗时。

```bash
.venv/bin/python experiments/command_locomotion_v2/launch.py \
  --out runs/command_v2_heading --gpus 4,5 \
  --init-checkpoint runs/command_fresh_h32_eval/selected.pt \
  --cache-root runs/command_fresh_h32/cold_cache \
  --num-envs 2048 --wm-batch 1024 --pi-batch 2048 \
  --iterations 30000 --horizon 32 --ckpt-every 2000 --log-every 200 \
  --buffer-steps 500 --lr 0.00005 --action-noise 0.06 \
  --start-speed 2 --max-speed 4 --wall-hours 1.5 --seed 20261102
```

launch.json 保存来源 checkpoint 哈希、源码哈希和启动参数；launch_timing.json 保存本轮完整耗时；source_metadata 保存第一轮迭代数及耗时；后台训练退出后评估进程会在处理完最后的 checkpoint 后自行退出。
