# B2+A3 V24

- `train.py` 将物理采样交给 MuJoCo Warp，将 WM 拟合和策略更新交给官方 SuperMimic 学习器。
- `launch.py` 记录命令、运行目录和进程。它会检查所选 GPU 的当前占用。
- `official_adapter.py` 接入实际执行器滤波和延迟状态，并保持 V24 检查点的模型维度。
- `b2_a3.xml` 和 `leg_kinematics.xml` 描述既有模型。网格包含实际碰撞几何，训练需使用同一套原始资产。

将现有 V24 资产目录里的 `mesh_*.obj` 放入本目录的 `meshes/`。
仓库提供 XML 和网格校验清单。若有权访问原训练服务器，可从
`/home/liudiwen/projects/Dogmimic_official/tasks/b2_v24_dual_scratch/meshes/`
取得匹配网格。权重、轨迹和机器专用环境单独管理。

从仓库根目录核对网格并运行小批量流程验证。GPU 0 仅为示例。

```bash
TASK=experiments/b2_v24_dual_scratch
(cd "$TASK" && sha256sum -c mesh_sha256.txt)
PY=supermimic/.venv/bin/python
"$PY" "$TASK/train.py" --help
"$PY" "$TASK/launch.py" --gpu 0 --no-promote \
  --out "runs/v24-smoke-$(date +%Y%m%d_%H%M%S)" \
  --envs 16 --batch 8 --wm-batch 8 --horizon 32 --wm-horizon 8 \
  --updates 4 --warmup 1 --smoke --final-n 0
```

省略权重参数时，Actor、Critic 和世界模型随机初始化。
`--wm-donor` 读取兼容的 V24 世界模型权重和归一化状态，Actor 和 Critic
仍随机初始化。`--resume` 从 V24 检查点恢复三套模型、优化器、迭代和课程档位，
重新收集在线采样。应使用可信来源的检查点。

去掉 `--smoke` 后启用物理评测。撞地、关节极限和碰撞约束由仿真处理。
EE 验收要求夹爪尖端误差 ≤1.5 cm，且全系统同时停稳 1 秒。看各条件的实际物理
结果决定是否开始长训。完整最终评测可设置 `--final-n 64`，运行两个独立种子。
按运行预算设置 `--max-hours`，必要时通过 `--deadline-hk` 指定带时区的绝对截止时间。

`launch.json` 记录 PID 和命令，`status.json` 记录进度，`latest_eval.json`
记录最近物理评测，`DONE.json` 记录结束原因和验收状态。运行目录内创建
`STOP` 后，训练会在迭代边界保存并执行配置中的结尾评测。
