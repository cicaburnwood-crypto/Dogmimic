# 实机使用指南

这是部署工具包，不会自动安装、切换控制器或开机启动。保留狗端已有原生控制器，避免同时启动两套DDS发布器。发布代码在本机离线验证；真实设备连接、网卡名、DDS依赖和安装尺寸应在目标设备核对。

## 1. 环境与链路

PC：`requirements-live.txt`；B2机载Python：NumPy、SDK2 Python及其CycloneDDS依赖。SDK源码在 `dog_bundle/vendor/unitree_sdk2_python-*`，按其README安装匹配平台的DDS库。包中不包含厂商操作系统、原生运动服务或A3控制器固件。

默认地址：B2 `192.168.123.164`，PC同网段静态地址；A3 `ws://10.42.0.101:8090`，PC另一网卡同网段。不要把两块直连网卡设置为默认路由。改变A3地址可编辑 `arm_console/server.py` 中 `Arm(endpoint=...)` 默认值；端口和本地安全Origin在代码中显式定义，不能只改浏览器URL。

## 2. 狗端桥

把 `inference_v18/` 或其中 `apps/b2-v18-live/robot_console/` 与 `dog_bundle/`复制到狗端，后者可放任意目录并通过 `--bundle` 指定。安装SDK：

```bash
python3 -m pip install numpy
python3 -m pip install ./dog_bundle/vendor/unitree_sdk2_python-814556d15970dd2ecf1c9984e845ca02ab07e206
```

先用offline模式确认导出加载；该模式不连接DDS：

```bash
python3 apps/b2-v18-live/robot_console/server.py --backend offline --bundle "$PWD/dog_bundle" --port 8765
```

确认没有另一狗端桥占用后，在机器人稳定原生模式、遥控器可用时显式启动SDK模式（网卡名按实际设备）：

```bash
python3 apps/b2-v18-live/robot_console/server.py --backend sdk --bundle "$PWD/dog_bundle" --interface eth0 --port 8765
```

启动会创建传输与状态接口；自训练动作仍需要明确模式切换/联合待命操作。不要把该启动命令当作“完全不接触硬件”的检查命令。此包不自动修改systemd或厂商开机流程。已有成功部署应先比对版本和配置，不要覆盖运行中的控制器。

PC单独终端建立隧道（使用自己的SSH认证，不提供或上传密码）：

```bash
ssh -N -L 127.0.0.1:18865:127.0.0.1:8765 unitree@192.168.123.164
```

## 3. PC服务

分别在两个终端运行，工作目录为 `inference_v18/`：

```bash
OPENBLAS_NUM_THREADS=1 .venv-live/bin/python apps/b2-v18-live/arm_console/server.py
OPENBLAS_NUM_THREADS=1 .venv-live/bin/python apps/b2-v18-live/server.py
```

联合网页 http://127.0.0.1:8881/ ，机械臂独立控制台 http://127.0.0.1:8879/ 。服务启动/重连不自动重放任务。A3反馈须真实且在POSITION模式；工具坐标为identity，夹爪连接正常。不要同时使用多个页面控制同一设备。

## 4. 操作顺序

1. 原生稳定站立时用“进入联合待命”；已经P4零速稳定时用“P4原地进入预备位”。两者先检查臂路径，以5%控制器速度将臂送至待命姿态，再确认P4稳定，才开放目标控制。
2. 手动XYZ/朝向：填写全部六个字段→解算预览→确认执行。预览仅输入、当前姿态及首帧校验，**不是完整IK路径保证**。本包没有新增慢速极限触地轨迹功能。
3. 键盘：W/S参考X，A/D参考Y，R/F高度；↑/↓绕末端横轴俯仰，←/→参考Z偏航，O/P末端指向轴滚转；空格夹爪开/合。页面含精确方向提示。
4. Esc/停止终止联合动作，狗归零平移/yaw并保持姿态，不自动趴下。折叠按钮走独立检查路径，不改变狗模式。交回遥控器释放网页会话。
5. 遥控器始终优先；任意输入撤销旧目标，完全无输入2秒后才恢复其他操作资格。恢复不自动重放任务。R2+Y切原生/P4，R2+上下升降，R2+A回正/标准高度；具体支持状态由狗端报告。

页面目标范围水平3m、Z=0.1–1.6m。跟踪标准1cm/10°/1秒只是当前传感器估算，不代表外部实测精度。目标超时、J1–J6跟随错误、几何/关节约束会报具体原因并停止；不要反复重试未解决故障。

## 5. 故障排查

- API离线：先核对SSH隧道和A3地址、再看遥控仲裁及反馈时效。
- `JOINT_STANDBY_REQUIRED`：显式进入正确模式的联合待命，不绕过检查。
- `JOINT_FOLLOWING_ERROR`：检查具体关节、实际/目标角、负载、速度与执行链，不提高阈值掩盖故障。
- `GROUND_CLEARANCE`：模型地面/安装/TCP需要核对；已知约1cm偏差尚未修正。
- `POLICY_NOT_CONVERGED`：策略12秒未达到目标，不等于数学IK无解。
- 仿真只有空白画面：先运行不渲染的smoke，再检查EGL和GPU驱动；与实机连接无关。

本次发行不提供无人值守自动运动或强制绕过保护开关。
