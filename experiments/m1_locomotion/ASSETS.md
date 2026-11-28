# M1 资产清单

所有路径相对本实验目录。[ASSETS.sha256](ASSETS.sha256) 记录运行、回放与训练制品
的完整 SHA-256。

| 路径 | 来源和用途 |
| --- | --- |
| .runtime/m1/robot/robot_model/M1-robotdog-asm09/ | 用户提供的 M1 原厂包 dg_fsm_m1-dev-develop_xwp.zip。包含 ASM09 URDF、MJCF、网格和纹理，flat.xml 为生成的平地场景。 |
| .runtime/m1/headless.xml | 由 ASM09 MJCF 派生的无视觉网格回放场景。 |
| .runtime/native/teachers/ | 同一原厂包的 Classic 平地权重 model_21000_08_07_18_05_11.onnx 和楼梯权重 policy_loco_stairs_new.onnx，另附训练用张量导出。 |
| .runtime/native/hardware/uni-exp/ | 板端 Classic 参考权重、部署前的 Uni Exp 恢复版和首阶回放索引。 |
| .runtime/native/contact-turn/uni-exp-contact-turn.onnx | 当前网页仿真模型，SHA-256 以 fa09fc625a62 开头。 |
| .runtime/native/contact-turn/hardware/uni-exp-conditional.onnx | 实机 ARM 条件图，SHA-256 以 1b1ec7244ed1 开头。 |
| .runtime/native/contact-turn/weights/ | run-74231 与 run-74234 的训练记录、选中检查点及继续训练所需的世界模型。 |
| .runtime/native/contact-turn/turn-*.npz 和配套 JSON | MuJoCo 生成的策略回放，包括板端 Classic 示范、独立留出集和两套闭环回采。 |
| .runtime/native/classic-posture/ | 前一版网页基线、直行训练与留出数据、首阶验证回放、板端 Classic 张量和训练初始检查点。 |
| .runtime/native/contact-turn/validated/ | 当前发布的逐场景结果、键盘、首阶和浏览器验证。 |
| hardware/ 与 hardware_logger/ | 项目的板端增量、装载脚本、本地日志服务和已记录的验证结果。 |

供应商文件保留原权属。仓库的第三方声明及原厂随包条款适用于相应制品。
实机回滚备份的位置和安装前提见 [实机交接](hardware/README.md)。
