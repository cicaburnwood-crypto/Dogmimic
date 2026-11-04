# B2 自训练策略部署记录

2026-11-03 本机更新：`models/hindstand` 已替换为 B2 C1 第 2000 次（checkpoint 总计数 2500），SHA256 `52ee319da4b4b97b73467f376a6cf415cbeda7584b0c00e476b0a5b3199aa829`，负载范围 0 / 5 kg。当前训练环境的目标裁剪和力矩 envelope 已接入 `apps/b2-console/pd_adapter.py`。Locomotion 模型文件全部保持原哈希。

本机原 WebUI：`bash apps/b2-console/start_console.sh`，地址 http://127.0.0.1:8765/ 。本次只更新本机模型与执行接口，没有向机器人发布命令或安装新版。导出、CPU 推理与接口检查通过不代表完整物理验收；C1 仍存在执行器超限，未 accepted。

当前验证和备份：`diagnostics/webui_model_swap_20261103/REPORT.txt`、`backups/`。旧 `deployment.tar.gz` 与以下部署收据属于历史版本，不能用于安装当前候选。

## 2026-11-01 历史安装记录（以下不是当前模型的验证结论）

目标：`unitree@192.168.123.164`，已确认主机 `unitree-b2-pc4`。

**当前完成的是模型安装、CPU 推理和真实传感器输入的影子运行。电机接管和实机动作验证尚未完成。** 本发布包没有电机命令发布入口，没有退出现有控制器，没有自启动服务。`b2ctl status` 显示的是本次部署检查快照，不是实时姿态。

## 已安装模型

| 技能 | 本聊天自训练来源 | 独立仿真结果 |
| --- | --- | --- |
| 后腿站立—保持—恢复四足 | `b2_payload_stage_recovery_v6_20261031_002818` 的 best，第 14,500 次 | 0 / 5 / 10 / 15 kg 额外负载分别严格通过 64 / 63 / 60 / 33 次，每组 64 次 |
| 四足 locomotion | `b2_locomotion_wtw_repair_20261101_110346` 的 best，第 78,000 次 | 严格 673/720，步态 705/720，完整完成 752/752；未达到全部核心案例验收目标 |

模型来源哈希、关节映射、增益、归一化和导出审计保存在各自 `models/<skill>/policy.json`、`export_audit.json`。未加载官方策略。Unitree SDK2 仅用于通信和消息定义，固定提交 `814556d15970dd2ecf1c9984e845ca02ab07e206`。没有安装官方策略权重，也没有执行 SDK 动作示例。

后腿站立使用我们的完整 15.5 秒参考轨迹及自己的早期策略作为冻结骨干，需要根部位置估计和 IMU。当前影子运行固定在起始参考帧，不代表机器人完成了整套动作。后腿策略原训练没有关节目标裁剪，部署推理增加关节限位内 0.02 rad 的裁剪；影子测试没有触发裁剪。

Locomotion 继承我们自己的训练权重；本轮之前有 82,000 次训练来源。其输入为关节、IMU、动作历史、速度命令及公开的内部步态时钟，不使用真值线速度、外部教师或官方策略。

## 机器人上的命令

安装目录：`/home/unitree/supermimic_b2/releases/20261101`，当前链接：`/home/unitree/supermimic_b2/current`。

```bash
python3 /home/unitree/supermimic_b2/current/b2ctl.py status
python3 /home/unitree/supermimic_b2/current/b2ctl.py selftest
python3 /home/unitree/supermimic_b2/current/b2ctl.py shadow --skill locomotion --seconds 20
python3 /home/unitree/supermimic_b2/current/b2ctl.py shadow --skill hindstand --seconds 20
```

加 `--remote-input` 只会将按住 L1 时的遥控速度命令送入影子推理，仍不会发送电机命令。摇杆方向尚未通过实机运动验证。结果写入 `/home/unitree/supermimic_b2/logs`。没有配置后台常驻进程或开机启动。

## 已完成的验证

- 两个 actor 各 256 组输入，与原始 Torch FP32 输出对照；机器人上的最大绝对动作误差分别为 1.67e-6 和 5.96e-7。
- 后腿站立 111 组未来参考坐标对照；locomotion 512 组时钟、速度限制、命令变化率对照。
- 在目标 B2 上各 1,200 次完整推理管线的合成输入计时。
- 真实传感器输入下，各运行 1,000 步、20 秒；P99 推理耗时后腿站立 1.405 ms、locomotion 0.768 ms，均没有 20 ms 超时。
- `rt/lowstate`、`rt/sportmodestate` 约 500 Hz，`rt/lf/sportmodestate` 约 20 Hz；只读 `CheckMode` 返回 `ai`。
- 此固件的 SportModeState 应用层 stamp 恒为零；采用持续递增的 DDS 源时间戳和本机接收时龄检查更新。发布端与 PC4 的时钟不一致，未将跨机绝对时间差误当作消息延迟。

这些结果验证软件数值与通信，不证明实机步态稳定或可直接承重执行。

## 阻止电机接管的实际问题

检查时关节呈趴卧姿态，标准站姿最大关节误差约 1.5 rad；若直接应用当前推理目标，最大关节跳变可达 2.25 rad。两个模型均从标准四足站姿训练，未训练从趴卧启动。

现有 `ai` 控制器仍占用控制权，本次没有调用 `ReleaseMode`。位置估计在该控制器运行时能读到，但其在低层接管后是否继续有效，以及高俯仰角下的估计精度，仍未验证。后腿站立不能把参考位置或固定高度伪装成测量值。

完整实机部署还需要现场支撑下的低层控制切换、关节方向/增益/遥控急停验证、标准站姿起步测试；后腿站立额外需要验证低层状态估计。Locomotion 目前未通过全任务仿真验收，也不能将已导出模型称为实机验证通过。没有放宽原验收标准或将模型 metadata 改为已通过。

SDK 的 B2 低层电机模式为 `0x0A`，不是旧通用脚本中的 `1`；此版本没有复用旧脚本直接输出电机控制量。后续电机接口需要独立验证，不能直接运行 vendor 中的运动示例。

## 本机网络与撤销

开发电脑使用已连接网线的 `enx6c1ff7c0a0b6`，创建内存临时连接 `codex-b2-temp-20261101`，地址 `192.168.123.222/24`，无网关、无自动连接。不改机器人 IP，也不改其他网卡默认路由。部署结束后仍保留此连接以便 SSH；撤销连接可在开发电脑执行：

```bash
nmcli connection down codex-b2-temp-20261101
nmcli connection delete codex-b2-temp-20261101
```

机器人侧仅新增 `/home/unitree/supermimic_b2` 下的发布包、日志和当前版本链接。没有改动 SDK、ROS、现有服务或全局 Python 环境。撤销部署只需移除该独立目录；目录内包含测试记录，移除前可先保存。
