# Inference pipeline 与坐标契约

```
世界/参考坐标目标位置+朝向
  → 实测 odom/IMU/腿与臂编码器、P4上下文
  → 99维 observation
  → V18 MLP (99→256 SiLU→256 SiLU→12 tanh)
  ├→ 6维机体请求 → 狗端P4历史/相位/滤波 → 12腿关节PD目标
  └→ 6维臂动作 → 速度/加速度/关节与碰撞投影 → A3关节跟随
```

## 99维输入（零起始索引）

| 区间 | 内容 |
|---|---|
| 0:3 | 机体坐标系 `(goal_position - actual_TCP)/0.5` |
| 3:12 | `actual_EE_rotation.T @ goal_rotation`，3×3行优先 |
| 12:15 | 机体系线速度 /2 |
| 15:18 | 机体系角速度 |
| 18:21 | `-body_rotation[2,:]` 投影重力方向 |
| 21:33 / 33:45 | 模型顺序12腿关节角 / 关节速度÷10 |
| 45:51 / 51:57 | 六臂关节角 / 角速度 |
| 57:74 | P4实际17维相位/命令上下文 |
| 74:86 | 上一帧12维管理策略动作 |
| 86:87 | 估计机体离地高度 |
| 87:93 / 93:99 | 上次臂关节目标 / 目标速度 |

米、弧度、秒；SDK腿顺序重排 `[1,0,3,2]`，详见 `policy.PhysicalGeometry.state`。参考原点取创建时odom XY，yaw也相对初始odom；Z独立使用腿FK及足半径32mm估高。姿态为IMU roll/pitch与相对odom yaw。网页XYZ欧拉角仅是目标输入表达，判据使用SO(3)相对旋转角。

输出0:6对应 vx、vy、yaw-rate、roll、pitch、height：平移合成≤2m/s、侧向≤1m/s、yaw-rate≤0.5rad/s；高度+5/-20cm。低蹲倾角包络：-20cm时5°、-15cm时10°、-10cm及以上20°。输出6:12是归一化臂速度意图，不是关节角。

实机臂目标步长20ms，硬速度0.6rad/s、加速度1.2rad/s²；期望速度随狗速度下降。关节额外余量 `[.030,.030,.025,.020,.025,.025]` rad。当前及候选几何检查、半步/终步投影、跟随误差和遥控器代次均参与控制。

P4由 NumPy `HeightLocomotionController`执行，10帧本体感知历史、20维goal；保留11Hz Q1 notch及命令过渡。manager不接受外部视觉或点云；模型中的TCP坐标不能直接当真实地面距离。

权重导出在 `models/manager.npz`，完整来源见manifest。`parity.npz`是8组仿真数值夹具，非实机日志。`tools/verify.py`核对Actor与机体请求；最新加速度修复的差异另由专项测试验证。
