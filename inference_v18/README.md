# V18 inference toolkit · B2-P4 + A3

这是 **V18 最佳累计 24500 次（本段 24000）** 的可迁移推理包，不是 V19，也不是最后一次训练权重。包含完整 PyTorch 检查点、无需 GPU 的 NumPy 导出、B2-P4 下层策略、仿真资产、仿真网页、实机网页、机械臂接口、狗端桥和离线测试。世界模型仅用于训练，推理不更新网络。

**状态：研究版本，未通过全部验收。** near 64/64、bridge 64/64、far 55/64；实机 10 cm 目标曾因 J6 跟随误差停止。请先阅读 [当前状态](docs/STATUS.md)、[推理契约](docs/PIPELINE.md) 和 [部署指南](docs/DEPLOYMENT.md)。历史仓库其他目录不是本分支 V18 的入口。

## 获取完整文件

```bash
git clone --branch Inference_V18 https://github.com/cicaburnwood-crypto/Dogmimic.git
cd Dogmimic
git lfs install
git lfs pull
cd inference_v18
```

检查点和网格使用 Git LFS。没有取回 LFS 实体文件时不能运行；不要仅复制指针文件。`SHA256SUMS.json` 列出模型与资产校验值。

## CPU 离线推理与验证（不连接机器人）

以下命令均在 `inference_v18/` 执行。CPU 工具在 Python 3.14 本机环境验证；仿真使用独立的 Python 3.11 环境。

```bash
python3 -m venv .venv-live
.venv-live/bin/pip install -r requirements-live.txt
OPENBLAS_NUM_THREADS=1 .venv-live/bin/python tools/verify.py
.venv-live/bin/python tools/infer.py
# 自定义 observation.json 是按 docs/PIPELINE.md 排列的 99 维数组
.venv-live/bin/python tools/infer.py --observation observation.json
```

最后一个工具只输出 Actor 动作和机体请求。六个机械臂动作不能直接当关节位置发送；必须经过有状态速度/加速度/几何投影及接收端跟随检查。

## 仿真（无硬件接口）

Linux + NVIDIA CUDA GPU，版本见 `requirements-sim.txt`，与 CPU 实机推理环境分开。当前仿真入口沿用训练环境/碰撞判据，不保证每个目标成功。

```bash
python3.11 -m venv .venv-sim
.venv-sim/bin/pip install -r requirements-sim.txt
CUDA_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=1 .venv-sim/bin/python simulation/smoke.py
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl .venv-sim/bin/python simulation/server.py --port 8878
```

打开 http://127.0.0.1:8878/，输入 XYZ（米）和 XYZ 欧拉角（度），提交目标，或重置。位置是原训练 TCP，而非自动改为夹爪指尖。GPU 编译首次加载可能较慢。需要可用的 EGL/OpenGL 驱动；`smoke.py` 不渲染，用于区分动力学/模型加载与图形驱动问题。

## 目录

| 路径 | 内容 |
|---|---|
| `checkpoints/v18_best24500.pt` | 原始 V18 manager、WM、优化器及元信息 |
| `checkpoints/p4_20000.pt` | 对应冻结下层训练检查点 |
| `apps/b2-v18-live/models/manager.npz` | 实机使用的 NumPy V18 Actor |
| `dog_bundle/models/locomotion/` | B2-P4 NumPy Actor、归一化参数、限位和滤波配置 |
| `assets/simulation/` | B2 平头版 + A3 + 连接板的 XML、网格、模型元信息 |
| `simulation/` | V18 原训练物理环境及独立仿真入口、采样和评测函数 |
| `apps/b2-v18-live/` | 本机联合控制接口、网页、运行时投影和跟随保护 |
| `apps/b2-v18-live/arm_console/` | A3 WebSocket 协议、夹爪、遥控器仲裁 |
| `apps/b2-v18-live/robot_console/` | 狗端 DDS 桥、遥控映射、反馈、租约 |
| `dog_bundle/` | 下层运行时、兼容模式模型、Unitree SDK 源码及原许可证 |
| `tools/` | 哈希/导出验证和纯推理示例 |

安装位置对应 B02/B03/B10/B11，前后孔距 67.5 mm；连接板沉孔侧朝狗。资产哈希固定，安装或 TCP 定义更改需要同步契约，不能仅改显示文本。

## 离线测试

```bash
OPENBLAS_NUM_THREADS=1 .venv-live/bin/python -m unittest discover -s apps/b2-v18-live/tests
OPENBLAS_NUM_THREADS=1 .venv-live/bin/python -m unittest discover -s apps/b2-v18-live/arm_console/tests
OPENBLAS_NUM_THREADS=1 .venv-live/bin/python -m unittest discover -s apps/b2-v18-live/robot_console/tests
node apps/b2-v18-live/tests/test_manual_goal.mjs
node apps/b2-v18-live/tests/test_keyboard_browser.mjs
```

这些是离线行为/数值回归检查，不代替真机动作验收。参见 [发布验证记录](docs/VERIFICATION.md)。

## 来源与许可

参见 [第三方来源](docs/THIRD_PARTY.md) 和仓库根目录 `THIRD_PARTY.md`。本分支明确增加自训练权重和用户提供模型资产，覆盖旧根目录文档中“未分发权重”的历史描述。没有增加笼统开源许可证；不同来源保留各自权利和原有许可证。
