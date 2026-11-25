# Dogmimic

Dogmimic 的训练入口使用 [HybridRobotics 官方 SuperMimic](https://github.com/HybridRobotics/supermimic)。
上游训练器、世界模型、Actor–Critic/BPTT、配置、测试和 G1 工具均来自完整子模块。
B2+A3 V24 是本项目的独立任务适配，使用自己的仿真和奖励，通过官方学习器更新。
既有历史实验和部署代码仍可按各目录原有说明使用。

## 安装

需要 Git、uv、Python 3.13 和可用的 CUDA 环境。两个官方源码目录必须保持同级。

```bash
git clone --recurse-submodules https://github.com/cicaburnwood-crypto/Dogmimic.git
cd Dogmimic
# 已有克隆可在切到本版后运行：
git submodule update --init --recursive
(cd supermimic && uv sync)
```

本版固定 SuperMimic `5fb92297ab4fe3d86629d9f0c635334791bb0d5f` 和
mjlab_research `dd51a9fcc98791a2f0b018b3c7f71a3eefd5ecdd`。
mjlab_research 固定在已通过 V24 域随机化验证的版本。首次 `uv sync` 会按这个
依赖版本重新生成本地锁文件。可用 `git submodule status` 核对源码提交。官方任务说明见
[SuperMimic README](https://github.com/HybridRobotics/supermimic/tree/5fb92297ab4fe3d86629d9f0c635334791bb0d5f#readme)。

## 官方 G1

从 `supermimic/` 执行。将 GPU 编号和参考动作路径替换为实际值。

```bash
cd supermimic
uv run --locked src/supermimic/scripts/train.py SuperMimic-GenCoord-G1 --help
CUDA_VISIBLE_DEVICES=0 uv run --locked src/supermimic/scripts/train.py \
  SuperMimic-GenCoord-G1 --motion-file /absolute/path/motion.npz --runner.run_name g1_example
uv run --locked src/supermimic/scripts/play.py \
  SuperMimic-GenCoord-G1 --wandb-path ENTITY/PROJECT/RUN_ID
```

另有 `SuperMimic-MaxCoord-G1`，训练和回放须使用相同任务 ID。

## B2+A3 V24

从仓库根目录使用 [V24 训练指南](experiments/b2_v24_dual_scratch/README.md)。
任务使用一个 Actor 控制 12 个腿关节和 6 个臂关节，无末端目标时追踪
速度，有目标时追踪夹爪尖端的世界坐标 XYZ。发布的是仿真训练代码。
