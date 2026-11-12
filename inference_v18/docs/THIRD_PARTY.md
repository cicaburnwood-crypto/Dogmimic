# 来源和权利说明

- V18 manager与B2-P4为本项目自训练策略，非宇树官方策略。权重来源和SHA256见本包manifest及SHA256SUMS。
- `simulation/engine/supermimic`及其任务适配源自SuperMimic-vibe： https://github.com/dhmbb2/supermimic_vibe ，历史基线 `072f0910dadc9d95c720e625b546120e9ce917c9`。包含本项目训练/推理适配。上游未提供的许可不在此补造；根目录THIRD_PARTY.md仍适用。
- Unitree SDK2 Python： https://github.com/unitreerobotics/unitree_sdk2_python ，revision `814556d15970dd2ecf1c9984e845ca02ab07e206`，BSD-3-Clause；原许可证随源码分发。用于DDS传输与消息定义。
- B2平头版+A3及连接板几何来自用户提供的模型/STEP资料，出处和转换元数据见 `assets/simulation/model.json` 与 `simulation/mount_assembly.json`。未对这些资料追加MIT等开放许可；再分发者应保留来源并核对原权利。
- A3控制台依据已安装CVTE CARM控制器协议实现，不包含厂商固件、原厂上位机二进制或厂商私钥。
- NumPy、SciPy、MuJoCo、PyTorch、mjlab、Warp等作为外部依赖安装，各自遵循其许可证。

本次发行没有包含机器人的运行会话令牌、SSH密钥、账户凭据、原始实机日志或私人遥测。内网默认地址是可编辑的设备配置，不是凭据。
