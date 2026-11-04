# G1 动作控制台

本地页面，与 B2 控制台独立。两个动作：`jump_60cm`（立定跳远 60 cm 目标）和 `funky_chicken`。不含 locomotion。

2026-11-03：跳远后端已替换为 `g1_jump_boundary_v11_20261102_171810/best.pt` 第 13000 次，checkpoint SHA256 `55d47be9ead36fc6a0863d79c64701ea7c86d04b2f66595182eec7432c854a7b`。执行接口保留 v11 的初始脚中心 XY 对齐，Funky Chicken 权重未变。页面版本来自实际加载的模型，可通过 `/api/status` 或 `/models.json` 核对。此次只更新本机，尚未部署至实机。

运行：

```sh
/usr/bin/python3 apps/g1-console/server.py --port 8767
```

打开 http://127.0.0.1:8767 。默认离线，不发布 DDS、不调用机器人、不生成模拟遥测。60 cm 是动作目标，不代表已验证的实机距离。两个实际训练模型与 SDK2 执行器已准备在 `deployments/g1_actions`；默认离线服务不会加载设备连接。

## 设备适配

`--backend module:factory` 注入无参数工厂。适配器提供：

- `status()`：JSON 可序列化字典；`connected`、`stable_standing` 为布尔值；未知姿态用 null。`stable_standing` 必须由当前 G1 遥测判断，数据不可用时不能返回 true。不可复用 B2 姿态阈值。
- 可选状态：`height`（m）、`velocity`（m/s）、`tilt`（度）、`battery`（%）、`action`（动作 ID）、`phase`（可读阶段）、`progress`（0–1，轨迹进度而非成功率）。没有数据返回 null。
- `execute(action)`：快速提交动作并返回 JSON 收据；启动时在设备端原子复核稳定站立，动作期间报告执行阶段与进度。设备运行和模型推理由适配器管理，不能阻塞 HTTP 状态读取。

只依据连接可用性及用户要求的稳定站立执行，不设额外解锁、等待时间、摇杆条件或确认弹窗。前端提交期间防止重复点击；后端串行检查当前姿态。HTTP 同源校验用于防止其他网页发起指令，不增加操作步骤。

接口：`GET /api/status`；`POST /api/action`，JSON `{"action":"jump_60cm"}` 或 `{"action":"funky_chicken"}`。不能通过 HTTP 伪造设备姿态。动作提交成功仅表示接受请求，不表示动作成功。

测试：`/usr/bin/python3 -m unittest discover -s apps/g1-console/tests`。

已实现的后端：`--backend backend:create_backend`，或使用 `deployments/g1_actions/start_console.sh`。具体模型、配置与验证见该目录 README。
