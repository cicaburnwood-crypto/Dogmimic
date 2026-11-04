# G1 被动底层记录器

机器人目录：`/home/unitree/supermimic_g1_actions_20261101/deployments/g1_actions/results/dds_logs/`。
服务：`g1-dds-logger.service`（用户级 systemd，独立于网页控制器）。

持续订阅当前域发布的匹配通道，不创建 DataWriter、不调用控制 RPC。SDK 未包含的消息类型从 DDS 类型发现获取；未知类型及无数据通道均列入状态记录，不冒充已采集。当前实际捕获包括 HG lowstate/lowcmd、双 IMU、骨盆/躯干里程计、原生 FSM、电池、主板和控制交接/运动 RPC。遥控器按键与摇杆同时保留于 lowstate 的完整 wireless_remote 字节；独立遥控器 topic 有发布时也记录。足底压力、手部等仅在机器人发布这些通道时可采集。

每条记录保留完整 IDL 字段，包括全部 35 个电机槽位、reserve、CRC、状态码及嵌套传感器信息。时间为本机接收 Unix 纳秒、monotonic 纳秒、可用时 DDS source timestamp/publication handle。已知类型直接保存 CycloneDDS 接收的 CDR 二进制载荷，不先解码再序列化；动态类型用完整 JSON。这是 DDS 载荷记录，不是原始以太网抓包。未知固件内部信息无法通过本工具读取。

每 60 秒关闭一个 gzip 段；`.part` 是正在写入的文件，不宜离线解码。日志不抽样、不循环覆盖；服务启动创建新 session。状态每两秒写入 `results/logger_status.json` 和当前 session 的 `status.json`，包含订阅清单、各 topic 接收/写入数、DDS 报告的丢样、应用队列丢弃、读写错误。lowstate 的 tick 原样保存在载荷中，可离线检查；其间隔单位取决于固件，不能直接当作丢失报文数。没有丢样报告不等于链路绝对无丢包。

```bash
systemctl --user status g1-dds-logger.service
systemctl --user stop g1-dds-logger.service  # 只停止日志，不影响控制器
systemctl --user start g1-dds-logger.service
```

从项目根目录解码已关闭的分段（需要已安装 CycloneDDS，脚本自动使用包内 SDK）：

```bash
python3 deployments/g1_actions/logger/decode.py \
  deployments/g1_actions/results/dds_logs/<session>/segment_00000.cdr.gz \
  --out /tmp/g1_lowlevel.jsonl
```

可加 `--topic rt/lowstate` 或 `--topic rt/lowcmd` 过滤。每行包含 header 及 `data` 全部解码字段。`results/logger_roundtrip.json` 是实机数据反序列化验证记录。

`tau_est` 是固件估计力矩。记录到 lowcmd 表示 DDS 上发布了该指令，不单独证明驱动器接受或实际执行。模型推理输入/输出仍由 `runner.py` 的动作 JSONL 记录，可用 Unix 时间与本记录关联。

该日志不包含摄像头/麦克风流或无关服务请求。当前 logger 服务只依赖网络，不启动模型、不改变原生控制模式、不为动作增加门控。

早期试运行的全字段序列化版本存在吞吐不足和队列丢弃，相关 session 的 status.json 保留实际丢弃数量。现用原始 CDR 接收路径；不要把试运行记录当作无丢包记录。
