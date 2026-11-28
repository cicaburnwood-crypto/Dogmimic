# Lightyear Dogmimic M1 平地与楼梯策略

本分支默认运行 M1 ASM09 的地形版 ONNX。上楼来自双前轮越阶训练，下楼来自带地形观测的平地与下行训练。两阶段均使用原版 SuperMimic。

## 启动仿真

在仓库的 experiments/m1_locomotion 目录执行。环境需要 Linux、Python 3.12、uv 和可用的 EGL。

~~~bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-sim.txt
M1_PYTHON="$PWD/.venv/bin/python" ./start.sh --native
~~~

打开 [本机仿真](http://localhost:8767/)。W/S 前后移动，A/D 侧移，Q/E 转向，空格停车，R 复位，方向键调节第三人称相机。页面可选平地和 18、20 厘米楼梯的上下行。下行场景会从上平台朝下复位，按 W 前进。

| 交付项 | 入口 |
| --- | --- |
| 当前模型 | [.runtime/native/m1-universal-3a27ea5132a1.onnx](.runtime/native/m1-universal-3a27ea5132a1.onnx) |
| 版本与回归索引 | [release.json](.runtime/native/release.json) |
| 训练命令、数据和两阶段奖励 | [训练复现](TRAINING.md) |
| 仿真验收、复测命令 | [测试说明](TESTING.md) |
| 资产来源与哈希 | [资产清单](ASSETS.md) |

此版验收范围为 MuJoCo。模型使用仿真计算的 terrain 输入。当前 M1 固件接口提供 obs 与 obs_history，使用此版本开展实机测试前须提供等价地形观测并重新验收。
