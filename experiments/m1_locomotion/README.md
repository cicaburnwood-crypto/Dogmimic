# Lightyear Dogmimic M1 Uni Exp

M1 ASM09 轮足机器人项目。Uni Exp 的当前版本使用原版 SuperMimic 重训平地组合转向，
在同一 ONNX 中保留原实机恢复版的标准直入楼梯策略。分支内的仿真用完整图和 M1 用 ARM
条件图共享同一训练检查点。2026 年 11 月 27 日的板端记录确认新版加载成功。

## 快速启动

需要 Linux、Python 3.12、uv 和可用的 EGL。以下命令从新签出的仓库开始。

~~~bash
git clone --branch Lightyear-Dogmimic --single-branch \
  https://github.com/cicaburnwood-crypto/Dogmimic.git
cd Dogmimic/experiments/m1_locomotion
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-sim.txt
M1_PYTHON="$PWD/.venv/bin/python" ./start.sh \
  --hardware-preview .runtime/native/contact-turn/uni-exp-contact-turn.onnx
~~~

打开 [本机仿真](http://localhost:8767/) 后，W/S 前后移动，A/D 侧移，Q/E 转向，
空格停车，R 复位。方向键调节第三人称视角。场景支持平地及 18、20 厘米楼梯上下行。
选择场景会复位机器人。端口已占用时可在启动命令末尾加上 --port 8768。

## 交付位置

| 内容 | 入口 |
| --- | --- |
| 版本资产、来源及 SHA-256 | [资产清单](ASSETS.md)、[校验文件](ASSETS.sha256) |
| 仿真、ONNX 等价、板端回放和实机试车 | [测试说明](TESTING.md) |
| 数据、官方学习器、检查点复现 | [训练说明](TRAINING.md) |
| 板端脚本、日志和回滚 | [实机交接](hardware/README.md) |
| Word 版本地交接快照 | [M1 Uni Exp 项目交接文档](docs/M1%20Uni%20Exp%20%E9%A1%B9%E7%9B%AE%E4%BA%A4%E6%8E%A5%E6%96%87%E6%A1%A3.docx) |

默认速度的仿真试验中，左前轮反复离地显著减少。右转还有少量抬轮和侧偏，倒车转向跟踪
仍需改进。18、20 厘米楼梯上行合计通过 56/60，下行通过 0/48。新版实机行走需现场验收。
