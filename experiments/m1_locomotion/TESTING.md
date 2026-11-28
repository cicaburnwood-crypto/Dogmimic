# M1 测试说明

从本实验目录执行。首次安装使用 [启动指南](README.md)。下面的回放使用已随分支交付的
两份 M1 教师权重和 ASM09 场景。

## 制品校验和动作等价

~~~bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
sha256sum -c ASSETS.sha256
.venv/bin/python tests/check_release.py
~~~

检查输入 obs [1, 49]、obs_history [1, 245] 和输出 actions [1, 16]。
固定随机采样后，对比网页图和 ARM 条件图的 3,876 帧转向观测，发布记录的动作最大绝对差
为 2.38 × 10⁻⁷。直入首阶的 50 组、550 帧动作与实机恢复版一致。

## 仿真回归

~~~bash
PY="$PWD/.venv/bin/python"
N=.runtime/native
R="$N/contact-turn"
M="$R/uni-exp-contact-turn.onnx"
mkdir -p results
"$PY" native_turn_replay.py "$N/hardware/uni-exp/classic-reference.onnx" \
  --candidate "$M" --suite final --out results/turns
"$PY" native_rollout.py --candidate "$M" --suite final --out results/terrain
"$PY" native_rollout.py --candidate "$M" --suite descent-final --out results/descent
~~~

用 [发布摘要](.runtime/native/contact-turn/preview-validation.json) 和逐场景日志对照。
在默认键盘速度用 W+Q、W+E 分别持续转弯，随后只松开转向键。观察左前轮接地、实际角速度
和松键过渡。反向重复，再确认空格停车、R 复位和方向键相机控制。

| 统计 | 旧网页预览 | 当前 |
| --- | ---: | ---: |
| W+Q 左前轮离地至少 1 厘米的次数 | 7 | 0 |
| W+E 左前轮离地至少 1 厘米的次数 | 7 | 3 |
| 168 个转向场景的左前轮抬起次数 | 361 | 13 |
| 倒车转弯平均角速度绝对误差 | 0.030 rad/s | 0.059 rad/s |
| 84 个松转向场景中任一轮离地超过 3 厘米 | 17 | 1 |

当前 180 个平地回归场景均完成。18 厘米上行 29/30，20 厘米上行 27/30，
与实机恢复版的同组回归一致。下行 0/48，其中 38 次越出测试车道，10 次跌倒。
下行以结果报告的 passed 字段作为验收依据。

## 板端和现场

2026 年 11 月 27 日的 [板端记录](hardware/board-validation.json) 包含 7,481 帧
离线回放和标准站立切换。切换首帧力矩跳变为 0，平滑交接为 800 ms。
需要现场复测时，先在平地验证站立、前后低速和停车，再检查左右组合转向及松键。
记录指令、关节、IMU、轮速、实际速度和角速度。实机与日志入口见 [实机交接](hardware/README.md)。
