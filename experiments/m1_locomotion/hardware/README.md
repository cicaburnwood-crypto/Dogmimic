# M1 板端交接

当前实机模型是上级目录 .runtime/native/contact-turn/hardware/uni-exp-conditional.onnx。
已有 M1 控制包的工作根目录为 /home/m1/dogmimic-uni-exp-3a27ea5132a1。
dg_fsm.service 从该目录的 scripts/ 运行，在启动时读取 models/uni_exp.onnx。
板端运行环境为 aarch64 和 ONNX Runtime 1.16.1。这里保留 Uni Exp 相关补丁和检查，
固件重新构建以操作者已有的 M1 原厂包和运行库为基础。

在对应的 M1 板端源码基线上，先应用 uni-exp.patch，再应用
gamepad-remap/HighLevelPath.patch。前者包含预加载和标准站立过渡，后者加入 Uni Exp
手柄重映射。preload.patch 和 stand-handoff.patch 用于分别查看这两项改动的历史差异。
两份发布补丁已在留存的板端基线副本上验证，生成的文件与部署源码一致。

部署脚本 activate-contact-turn.sh 对应这次交付的校验值，依赖现有板端离线回放程序、
站立读数以及启动校验脚本。它在折叠静止时备份旧权重、原子替换和重启控制服务。
来源记录和性能在 [board-validation.json](board-validation.json)。

标准站立姿态稳定、摇杆回中后，长按 L2 进入 Uni Exp。L1+R1 起立，长按 L1 选择
Loco Classic，长按 R1 选择 Loco Stairs，L2+R2 下蹲。实机试走范围见
[测试说明](../TESTING.md)。

~~~bash
systemctl status dg_fsm.service dogmimic-m1-debug.service --no-pager
sudo -n journalctl -b -u dg_fsm.service -n 100 --no-pager
tail -n 60 /var/log/dogmimic-m1/debug.jsonl
~~~

旧版备份相对实机根目录为
.deployment/contact-turn-fa09fc625a62/previous-5fcca3909d58.onnx。
从折叠静止恢复旧权重的带校验命令见 [Word 交接文档](../docs/M1%20Uni%20Exp%20%E9%A1%B9%E7%9B%AE%E4%BA%A4%E6%8E%A5%E6%96%87%E6%A1%A3.docx)。
