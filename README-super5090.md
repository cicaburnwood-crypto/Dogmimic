# SuperMimic — super_5090 复现环境

服务器：`ssh super_5090`（主机名 `lightyear`，用户 `liudiwen`）。
工作目录：`/home/liudiwen/projects/supermimic_vibe`。
Python 虚拟环境：`/home/liudiwen/projects/supermimic_vibe/.venv`。

本环境使用用户指定的社区实现 [dhmbb2/supermimic_vibe](https://github.com/dhmbb2/supermimic_vibe)，固定提交 `072f0910dadc9d95c720e625b546120e9ce917c9`。算法源码保持该提交内容。额外的 `deployment/` 提供无需 Slurm 的启动和验证脚本。

## 进入环境

```bash
ssh super_5090
cd /home/liudiwen/projects/supermimic_vibe
source deployment/activate.sh
```

默认只使用 GPU 0。切换 GPU 时，在激活前或运行命令前设置 `CUDA_VISIBLE_DEVICES`。缓存全部放在项目内，依赖固定版本保存在服务器的 `requirements-super5090.lock.txt`。

## 单卡训练

```bash
CUDA_VISIBLE_DEVICES=0 ./deployment/train.sh \
  --out runs/lafan1_5090 --wall-hours 3 \
  --eval-every 4000 --ckpt-every 4000
```

脚本在输出目录已有 `last.pt` 时自动恢复。恢复时请保持环境数、批量和算法参数一致；`--wall-hours` 是检查点累计训练时间上限，如已有 3 小时检查点，继续到 6 小时请指定 `--wall-hours 6`。

后台运行：

```bash
mkdir -p logs
CUDA_VISIBLE_DEVICES=0 nohup ./deployment/train.sh \
  --out runs/lafan1_5090 --wall-hours 3 \
  --eval-every 4000 --ckpt-every 4000 \
  > logs/lafan1_5090.log 2>&1 &
tail -f logs/lafan1_5090.log
```

## B2 最新续训（2026-10-30）

在原 B2 第 4,000 轮模型上继续训练稳定性和完整往返流程：四足站立、起立、后腿直立保持、恢复四足、稳定停住。使用 GPU 6，代码和规则见 `experiments/b2_cycle/README.md`，新运行目录和 PID 见 `experiments/b2_cycle/latest_launch.json`。此任务替代旧 B2 任务的“仅起立和保持”目标，旧结果仍保留。

在服务器项目根目录运行 `.venv/bin/python experiments/b2_cycle/status.py` 查看当前 B2 状态；`experiments/three_skills/status.py` 仍显示此前三个独立实验的历史运行。

## 两卡训练

确认两张卡空闲后运行：

```bash
CUDA_VISIBLE_DEVICES=0,1 ./deployment/train.sh \
  --out runs/lafan1_2gpu --wall-hours 3 \
  --eval-every 4000 --ckpt-every 4000
```

启动器按可见 GPU 数运行 `torchrun`，上游代码自动平均梯度并按 GPU 数的平方根调整学习率。

## 评估

```bash
source deployment/activate.sh
python -m supermimic.evaluate --run runs/lafan1_5090 --out benchmark.jsonl
```

评估读取 `ckpt_*.pt`，结果写入所选输出 JSONL。每段动作从首帧开始运行，使用固定摩擦系数并关闭观测噪声；这些设置由上游评估代码控制。

## 数据

已准备 [LAFAN1 Retargeting Dataset](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset) 的 40 个 `g1/*.csv`，原始文件位于 `data/raw/lafan1_g1/`，下载大小及 SHA256 在 `data/lafan1-download-manifest.json` 中。50 Hz 转换文件位于 `data/processed/lafan1_50hz.npz`。

LAFAN1 数据使用其数据发布方规定的 CC BY-NC-ND 4.0 许可。这里只为复现准备用户工作目录内的数据。

论文全量 BONES-SEED 数据尚未准备：该数据集需用户 Hugging Face 账户获得访问授权。上游提供 `slurm/download_bones.slurm` 与 `code/scripts/filter_bones.py`，本次先验证公开 LAFAN1 路径。

## 验证记录

2026-10-27 已完成安装、运行时验证、默认批量短训练、检查点读取与评估。

- Python 3.11.15，PyTorch 2.14.1 / CUDA 13.0，支持 RTX 5090 的 `sm_120`。
- mjlab 1.2.0，MuJoCo / MuJoCo Warp 3.5.0，warp-lang 1.12.0。
- 150 个包通过 `uv pip check`；CUDA bf16 前反向、Warp CUDA、MuJoCo EGL 渲染通过。
- 动作库：40 段、441,120 帧、50Hz，约 2.451 小时。
- 双卡 NCCL `all_reduce` 在 GPU 1、2 上通过。多卡完整训练尚未进行。
- 单卡默认批量：4096 个环境、2048 个世界模型窗口、4096 条想象轨迹、H=32、1000 步回放缓存；100 次迭代完成 40 次模型更新和 671,744 个环境步（含预热）。
- 世界模型、策略、critic 的优化器均实际更新；所有训练日志数值和检查点张量均为有限值。
- PyTorch 最大分配约 13.884GB（不包含 Warp 分配）；稳定阶段约 0.18 秒/迭代，首次图编译耗时数分钟，缓存已经保留。
- `ckpt_0000100.pt` 读取成功，并完成 40 段动作的评估流程。归一化回报 0.001820，40/40 动作失败：这是只训练 100 步的环境验证检查点，并未达到策略复现结果。

安装、运行时检查、数据转换、短训练和评估日志位于服务器项目的 `logs/`。本地 `verification/` 保存关键验证记录与锁定依赖。

复验命令：

```bash
python deployment/verify_runtime.py
python deployment/verify_training.py runs/smoke_5090
```

`runs/smoke_5090` 仅为环境验证检查点；正式实验请使用新的 `--out` 目录。本次验证进程均已结束。

关键证据：本地 `verification/logs/runtime-report.json`、`verification/runs/smoke_5090/verification.json`、`verification/runs/smoke_5090/smoke_eval.jsonl`。服务器上对应目录相同（无 `verification/` 前缀）。
