# M1 资产清单

路径均相对本实验目录。运行和训练制品的 SHA-256 见 [ASSETS.sha256](ASSETS.sha256)。

| 制品 | 来源及用途 |
| --- | --- |
| .runtime/m1/robot/robot_model/M1-robotdog-asm09/ | 用户提供的 M1 原厂包 dg_fsm_m1-dev-develop_xwp.zip 中的 ASM09 URDF、MJCF、网格和纹理。flat.xml 和 .runtime/m1/headless.xml 为派生场景。 |
| .runtime/native/teachers/ | 同一原厂包的平地 Classic 权重 model_21000_08_07_18_05_11.onnx、楼梯权重 policy_loco_stairs_new.onnx 及训练张量。 |
| .runtime/native/bilateral/ | 双前轮上楼训练记录、检查点及回采数据。 |
| .runtime/native/terrain_training/ | 下行及平地训练记录、留出数据、检查点和世界模型。 |
| .runtime/native/m1-universal-fe53c0fe605b.onnx | 带双前腿上楼动作的下行训练基底。 |
| .runtime/native/m1-universal-3a27ea5132a1.onnx | 默认地形版模型。SHA-256 为 3a27ea5132a1be524a1aa5d89f95f098a681c1df1ad64c00a5fcc8795a2da02a。 |
| .runtime/native/standing/k110-validation/ | 与当前模型对应的仿真验收。 |

供方文件保留原权属。训练数据为 MuJoCo 回放。train.npz 分为三个 Git 文件，按文件名顺序拼接即可恢复，步骤见 [训练复现](TRAINING.md)。
