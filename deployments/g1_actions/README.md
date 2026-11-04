# G1 两动作部署准备包

此包提供 **立定跳远 60 cm** 与 **Funky Chicken** 的真实训练权重、参考轨迹、NumPy 推理器、SDK2 执行接口和网页后端。不是轨迹直接播放，不含 locomotion，不使用官方策略。启动默认离线，不连接或触发任何机器人。

## 模型来源

- 跳远：本项目 `g1_jump_boundary_v11_20261102_171810` 第 13000 次，checkpoint SHA256 `55d47be9ead36fc6a0863d79c64701ea7c86d04b2f66595182eec7432c854a7b`。独立评测 501/672，完成 663/672，标准 32/32，未达到所有案例验收要求。当前视频中标准样本落地约 0.595 m。
- Funky Chicken：本项目 `g1_funky_chicken_20261030_211659` 第 8000 次，原仿真 32/32 成功，关节参考 RMSE 0.119 rad。目标是自建 Funky Chicken 风格动作，不是 BONES 官方动作复刻。
- 两个模型均未经过 G1 实机验证。原始评测保存在各 `models/*/simulation_eval.json`；每个目录包含模型来源哈希、PD/关节映射、参考轨迹及导出对照记录。

2026-11-03 的本机后端替换已完成；没有连接或更新 G1 实机。v11 同时导出足部运动学数据，按起始双脚七个碰撞几何中心的均值对齐 XY；保留竖直误差，避免旧根部锚定改变输入含义。128 组初姿与训练端对照通过。离线模式也加载相同 Actor，返回真实模型版本，不构造遥测。原文件备份与完整记录在 `diagnostics/webui_model_swap_20261103`。

## 运行与接入

运行需要 Python 3.10+、NumPy；设备接入另需 CycloneDDS 0.10.2 与包中固定版本的 SDK2。没有 Torch/GPU 依赖。CPU 物理重放另需与模型相同版本 MuJoCo 3.5。

从项目根目录运行离线页面：

```bash
G1_PYTHON=/usr/bin/python3 deployments/g1_actions/start_console.sh
```

接入设备时，在机器人或具有 DDS 网络直连的主机配置实际网卡：

```bash
export G1_INTERFACE=eth0   # 替换为实机连接网卡
export G1_ODOM_TOPIC=rt/odommodestate
export G1_PYTHON=python3  # 使用已装 NumPy/CycloneDDS 的解释器
# 一次只运行一个 8767 服务，先停止此前的离线服务。
deployments/g1_actions/start_console.sh
```

网页仍为 `http://127.0.0.1:8767/`。部署至 G1 时可通过 SSH 的 `-L 8767:127.0.0.1:8767` 转发。只有用户点击两个动作按钮才接管控制。网页必须收到稳定站姿才提交；不额外设置解锁、倒计时、确认框或遥控器门控。

## 执行链路

`POST /api/action` → `runner.Backend` → 当前姿态复核 → `MotionSwitcher.ReleaseMode` → 最新遥测 → 50 Hz 模型推理 → 500 Hz HG LowCmd/PR 模式 → 轨迹结束后 `SelectMode` 恢复原控制者。原控制者为空时使用 SDK2 `SwitchToInternalCtrl(LAST)` 请求回到原生。重复点击正在执行的动作不会重启动作，也不会并发执行另一个动作。

动作时长以模型元数据为准。进度代表参考轨迹执行进度，不代表实际成功率。记录 `results/<动作>_<时间>.jsonl` 包含时间、q/dq、tau_est、IMU、根部位置、传感器时间戳/新鲜度、目标 q、推理耗时及轨迹帧。执行异常与原生交接异常会在网页显示。

## 必须在连接后核对的设备事实

这不是额外解锁流程，而是当前尚无 G1 设备信息的适配边界：

- 目标为 **29 自由度 G1，腰部 roll/pitch 可动**。23DOF 或锁腰型号不能直接使用此包；不能只把缺失关节置零。
- 关节按官方 SDK 29 关节索引映射；以接入机器的型号/固件实际配置核对 `policy.json`。默认 IMU 为 pelvis 姿态/角速度，四元数 wxyz。
- 根部位置必须来自真实里程计，位置/速度与 odometry IMU 同世界坐标、参考点为 pelvis。首次姿态对齐处理世界坐标 yaw 差；v11 跳远按起始双脚中心对齐 XY，Z 保留参考和测量的差值。Funky Chicken 保留原根部锚定；不能把另一身体部位的位置当作 pelvis。
- `rt/odommodestate` 在退出原生控制后是否持续更新，需要实机验证。既不把固定零位置当测量，也不把参考轨迹位置冒充实际位置。数据停止时记录异常、停止本次动作并请求交回原控制者。
- 使用训练模型的逐关节 PD，未经额外目标限幅或动作平滑。物理模型的力矩限制记录在元数据中，SDK 的实机驱动器限制需要对照；软件没有声称已复刻驱动器饱和特性。
- 启动阶段只订阅，不更改控制权。原生接管/交还是否支持当前固件、500 Hz 发送线程与 50 Hz 推理在机载 CPU 上的实际时序均需连接后验证。

只读检查工具：设置上述 SDK 路径与网卡后运行 `python3 deployments/g1_actions/inspect_device.py`。它不会接管或发电机指令。

## 验证

```bash
/usr/bin/python3 deployments/g1_actions/selftest.py
/usr/bin/python3 -m unittest discover -s deployments/g1_actions/tests
# 有 MuJoCo 3.5 与 training_model.mjb 的服务器：
python3 deployments/g1_actions/replay.py
```

`export_audit.json` 比较原始 Torch CPU 网络和 NumPy 网络，包含 goal 变换对照。`results/replay.json` 是部署推理器的单次名义物理重放，不替代训练时多次验收，更不等于实机通过。

SDK 固定版本：`814556d15970dd2ecf1c9984e845ca02ab07e206`（保留 BSD-3-Clause 许可证）。控制消息与关节定义依据 [Unitree 官方 G1 低层示例](https://github.com/unitreerobotics/unitree_sdk2_python/blob/master/example/g1/low_level/g1_low_level_example.py)。不复用官方策略权重。
