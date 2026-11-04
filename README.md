# Dogmimic

基于 **SuperMimic-vibe** 的 Unitree B2 / G1 运动学习、仿真评测和机器人控制代码。
本仓库保存现有工作区的代码快照，当前 B2 倾斜全向运动任务采用 vibe 实现。
这不是 SuperMimic 原作者的官方仓库；历史 NJU 对比实验保留在独立目录中。

## 主要内容

- **B2 倾斜全向运动**：前后 ±4 m/s、左右 ±1 m/s、合成平面速度不超过 4 m/s；roll/pitch 各 ±20°，可叠加；累计 yaw 转角不限，角速度在有限训练域内采样。
- **B2 姿态与后腿站立**：命令角度追踪、动作编排、起立/保持/恢复，以及历史负载实验。
- **G1 动作学习**：60 cm 立定跳远及其他历史动作实验。
- **机器人控制台**：网页、遥控器事件处理、原生/自训练控制交接、SDK 适配、模型导出和离线回归测试。

任务目标不等于已通过验收。此代码快照仍包含尚在排查的实机起步与控制交接问题，
不能把仿真结果、单元测试或本次上传当作完整实机验证。

## 目录

| 路径 | 内容 |
| --- | --- |
| `code/supermimic/` | 固定版本 SuperMimic-vibe 基础学习代码 |
| `experiments/b2_tilt_locomotion/` | B2 倾斜全向运动初版任务 |
| `experiments/b2_tilt_locomotion_repair/` | 采样覆盖与任务损失修正后的续训代码 |
| `experiments/b2_attitude/` | B2 四足姿态命令任务 |
| `experiments/` | B2/G1 历史实验、评测标准和任务适配 |
| `apps/b2-console/`、`apps/g1-console/` | 控制台和机器人外部接口 |
| `deployment/` | 训练环境与启动辅助脚本 |
| `deployments/` | 模型导出、推理、SDK 适配及原部署目录结构 |
| `upstream/` | 固定上游版本的本地源码镜像 |
| `vendor/nju_supermimic/` | 历史 NJU 后端对比代码，不是当前 B2 任务后端 |
| `demo-video/` | 视频生成脚本，不包含生成的视频 |
| `verification/` | 已记录的训练环境依赖锁定文件 |

## 环境

训练环境使用 Python 3.11 与 CUDA GPU。依赖保留在 `requirements.txt`，原环境完整锁定清单见
`verification/requirements-super5090.lock.txt`。

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
source deployment/activate.sh
```

`code/`、`env.sh`、`requirements.txt` 和 `setup_local_env.sh` 从固定的 `upstream/` 镜像原样复制，
用于恢复原训练服务器的根目录布局；任务算法代码未因上传而修改。

实验入口、命令范围、评测条件和初始化方式以各目录的 README、训练脚本与
`evaluation_manifest.json` 为准。旧启动脚本和 XML 中保留了原机器的路径、SSH 主机别名、GPU 编号及资源位置；
在新机器运行前需配置对应资源。续训入口还需要原检查点，不能仅凭本仓库复现某次已结束运行。

## 离线回归测试

B2 控制台测试使用伪后端和伪 SDK，不连接机器人。安装 NumPy 后在仓库根目录执行：

```bash
python -m unittest discover -s apps/b2-console/tests -p 'test_*.py'
# G1 的离线后端测试会加载模型；先补齐 deployments/g1_actions/models/ 下的原权重。
python -m unittest discover -s apps/g1-console/tests -p 'test_*.py'
node --check apps/b2-console/web/app.js
node --check apps/g1-console/web/app.js
```

本次快照检查：B2 121 项回归测试通过，两个控制台 JavaScript 语法检查通过。
G1 的 3 项控制逻辑测试通过；另 1 项离线后端测试因本快照不包含 `weights.npz` 而无法完成。
这不是训练、物理仿真或实机动作验收。

## 快照范围

本次上传包含源码、静态网页、机器人模型描述/网格、配置、第三方许可证，以及小型 PD 回归测试输入。
**不包含**训练检查点和导出权重、原始动作数据、实机日志、诊断目录、浏览器资料、运行状态、视频、虚拟环境和压缩部署包。
`deployments/**/models/*/policy.json` 等元数据被保留，对应权重需要单独提供。
`diagnostics/` 中尚未合入正式源码的候选修复也不包含在此快照中。

源文件摘要及固定版本见 `SOURCE_SNAPSHOT.json`。历史文档中的结果、路径和运行编号仅代表各自记录时点。

## 来源与许可

当前 vibe 基线：[`dhmbb2/supermimic_vibe`](https://github.com/dhmbb2/supermimic_vibe)，
提交 `072f0910dadc9d95c720e625b546120e9ce917c9`。

第三方来源、目录和许可说明见 [THIRD_PARTY.md](THIRD_PARTY.md)。原有版权声明和许可证保持原样；
本次快照没有给整个混合代码库额外授予统一开源许可证。
