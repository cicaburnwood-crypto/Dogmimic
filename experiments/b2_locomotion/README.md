# B2 无参考全向四足 locomotion

本轮按用户最新要求只训练与仿真评测，实机型号细节、固件、遥控器、质量及控制接口随后提供；没有连接实机或发送任何电机指令。

策略、世界模型、价值函数和回放均从零初始化。使用现有 SuperMimic 社区复现的在线世界模型、想象轨迹 BPTT 与价值函数训练，不切换为 PPO，不加载官方策略、官方动作、已有自训练策略或教师。官方 B2 MJCF 仅提供机器人几何、质量、关节和扭矩范围。标准站姿是静态关节设定 `[0, .67, -1.3] × 4`，不属于行走参考动作。

## 速度及起止流程

- 指令是机身水平朝向坐标系的 `(vx, vy, wz)`，支持后退、侧移、斜向及平移与旋转叠加。
- `sqrt(vx² + vy²) <= 2 m/s`，`abs(vy) <= .6 m/s`；vx 没有额外分量限幅，仍受合成速度上限约束。
- wz 不做硬裁剪。高斯训练分布从标准差 .35 rad/s 逐级增加至 1.5 rad/s，分布无有限截断。有限评测覆盖正负 .5/1/2/3 rad/s；不以此宣称任意大转速均能实现。
- 指令改变限制线加速度为 1.5 m/s²、角加速度为 3 rad/s²，仿真和预备部署代码共用同一逻辑。
- 所有 reset 均为标准站姿、根和关节速度严格为零。每个 22 秒训练回合：站立 1.5 秒；运动至 7.5 秒；站立至 10.5 秒；再次运动至 17 秒；站立至结束。运动段内部随机变速、变向，并混入停止指令。
- 不提供参考关节轨迹、步态时钟或接触顺序。奖励衡量速度、姿态、停稳、脚滑、足端净空、动作平滑和力矩正则。

## 可用于后续实机适配的观测

每帧 42 维：12 关节角相对站姿、12 关节速度、3 维机身陀螺仪、3 维机身投影重力、12 维上一动作。历史 10 帧，加 3 维速度指令。Actor 不接收仿真真值线速度、世界位置、脚接触真值或动力学随机参数。世界模型与 critic 可以使用仿真状态和随机化参数。

50 Hz 策略，200 Hz 仿真 PD；kp=250，kd=8，保留 B2 MJCF 的 200/200/300 N·m 电机限幅。训练随机化：摩擦 .5–1.25、机身质量/惯量 .8–1.2 倍、重心 XYZ ±1.5 cm、kp .85–1.15 倍、kd .8–1.2 倍、编码器偏置 ±.005 rad、IMU/关节噪声、0/5/10 ms 指令延迟。世界模型记录每一步的延迟历史动作，想象滚动时同步更新，避免隐藏的控制器状态不匹配。

资产总质量是 83.498 kg；这只是当前模型参数，不能当作用户实机质量。后续实机部署前须按实际硬件重新核对。

## 独立评测与保存

GPU 4、5 同步训练，各 2048 环境，共 4096；GPU 6 单独物理评测。每 2000 次迭代保存 checkpoint 和独立评测。

47 个指令案例，每例 16 次（一个标称物理配置、其余随机动力学）；所有案例均走完整 17 秒流程：站立 → 指令运动 → 停止 → 反向指令运动 → 停止。平移包含 .25/.5/1/1.5/2 m/s、前后侧向和斜向；还包含原地旋转和移动旋转。

成功同时要求：不跌倒、不重置；最低高度 >.4 m、倾斜 <.55 rad；运动稳定区间 XY 速度 RMSE <=max(.12,20%指令速度)，角速度 RMSE <=max(.15,20%指令转速)；停车稳定区间平均平移速度 <.10 m/s、峰值 <.22 m/s、平均偏航速度 <.15 rad/s、漂移 <.10 m、站姿关节 RMSE <.20 rad、四脚低于接触高度的比例 >=90%、脚速 <.10 m/s。

标称试验仍包含观测噪声。2 rad/s 以内旋转是当前有限验收集，±3 rad/s 单独报告为扩展探测。连续两次评测中所有核心案例 >=95% 才提前结束；不能把训练奖励当作效果。课程扩大需要连续两次当前范围各案例 >=87.5% 且标称通过。

`best.pt` 与 `best_eval.json` 保留物理评测最佳；`last.pt` 与 `latest_eval.json` 表示最新。`best_rollout.npz` 保存每例标称轨迹和按编号遇到的第一个失败轨迹。默认预算 60000 次迭代或 6 小时墙钟，先到为准。若评测进程异常退出，监督进程会让训练保存后停止。

`audit.py` 验证禁止创建动作库、零初速、合成速度限制、无角速度裁剪、物理随机化、延迟历史一致性和世界模型到新策略的梯度。正式训练不继承短训练 checkpoint，复用软件编译缓存。

## 运行与状态

远程项目 `/home/liudiwen/projects/supermimic_vibe`，SSH `super_5090`。

```bash
.venv/bin/python experiments/b2_locomotion/launch.py --out runs/b2_locomotion_fresh_YYYYMMDD_HHMMSS --gpus 4,5 --eval-gpu 6 --trials 16 --num-envs 2048 --wm-batch 1024 --pi-batch 2048 --wm-batches 4 --horizon 24 --buffer-steps 512 --iterations 60000 --warmup-steps 128 --log-every 200 --ckpt-every 2000 --eval-every 0 --lr .0001 --action-noise .04 --lr-decay-iters 60000 --wall-hours 6 --seed 20261101
.venv/bin/python experiments/b2_locomotion/status.py
```

`latest_launch.json` 记录训练和评测 PID、GPU、启动时间和源代码哈希。监督进程独立 session 运行，关闭 SSH 不影响训练。没有创建聊天定时监视任务。

`deploy/` 自动导出我们的 CPU TorchScript actor，归一化和动作边界与评测一致，并验证导出数值误差。`controller.py` / `deploy_sdk2.py` 是预备适配代码，目前未对实际硬件或网络测试，不能将其标为实机已验证。SDK2 文件仅作为消息和电机编号文档使用：

- https://github.com/unitreerobotics/unitree_sdk2_python/blob/master/example/b2/low_level/unitree_legged_const.py
- https://github.com/unitreerobotics/unitree_sdk2_python/blob/master/example/wireless_controller/wireless_controller.py

训练不依赖 SDK2，更不使用其高层行走接口。
