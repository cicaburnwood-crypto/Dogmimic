# M1 训练与导出

当前版使用 [HybridRobotics SuperMimic](https://github.com/HybridRobotics/supermimic) 的
WorldModelLearner 与 BpttPolicyLearner。M1 训练入口核查上游提交
abc124c1bd5976341034668af3d12c087f2f1f55。
为 M1 准备独立的 SuperMimic 工作目录，在同级放置
[mjlab_research](https://github.com/HybridRobotics/mjlab_research)，其记录提交为
dd51a9fcc98791a2f0b018b3c7f71a3eefd5ecdd。按照固定提交的安装配置同步官方环境。

以下命令从本实验目录运行。将 SM_PY 设为该 CUDA 训练环境的 Python。训练数据和每个
输入的来源记录随仓库交付。native_turn_train.py 检查训练集与留出集的种子互斥。

~~~bash
SM_PY=/absolute/path/to/supermimic/.venv/bin/python
N=.runtime/native
C="$N/classic-posture"
R="$N/contact-turn"
mkdir -p training-runs
export CUDA_VISIBLE_DEVICES=0 TORCH_COMPILE_DISABLE=1

"$SM_PY" native_turn_train.py \
  --train "$R/turn-train.npz" --holdout "$R/turn-holdout.npz" \
  --straight "$C/flat-train.npz" --classic "$C/board-classic.npz" \
  --initial "$C/supermimic/run-271132/policy-0192.npz" \
  --world-start "$C/supermimic/run-271132/dynamics.pt" \
  --seed 74231 --world-updates 760 --actor-updates 360 \
  --out training-runs/run-74231

"$SM_PY" native_turn_train.py \
  --train "$R/turn-train.npz" --holdout "$R/turn-holdout.npz" \
  --straight "$C/flat-train.npz" --classic "$C/board-classic.npz" \
  --initial training-runs/run-74231/policy-0360.npz \
  --world-start training-runs/run-74231/dynamics.pt \
  --extra "$R/turn-wide-on-policy.npz" --extra "$R/turn-clearance-on-policy.npz" \
  --seed 74234 --world-updates 600 --actor-updates 288 --policy-lr .016 \
  --support-weight 1.25 --tilt-weight .19 --yaw-weight 1.4 \
  --refine-contact --out training-runs/run-74234
~~~

第一轮学习平地转弯时的接地、机身垂向稳定和速度跟踪。第二轮固定平地网络，训练有限幅度
的逐轮接地和转向修正。发布记录选择 run-74234 的 policy-0096.npz。
下面从已选择的检查点复现发布 ONNX。

~~~bash
PY="$PWD/.venv/bin/python"
N=.runtime/native
R="$N/contact-turn"
B="$N/hardware/uni-exp"
mkdir -p results
for mode in dense conditional; do
  flags=()
  if test "$mode" = conditional; then flags+=(--conditional); fi
  "$PY" native_classic_export.py \
    "$B/uni-exp-hardware.onnx" "$B/classic-reference.onnx" \
    "$R/weights/run-74234/policy-0096.npz" "results/uni-exp-$mode.onnx" \
    --contact-turn --clearance .047 .091 --tilt .092 .17 --gyro .16 .41 \
    --release-slew .055 --release-contact-gain 18 "${flags[@]}"
done
sha256sum results/uni-exp-{dense,conditional}.onnx
~~~

将校验值与 [ASSETS.sha256](ASSETS.sha256) 对照。训练后用
[测试说明](TESTING.md) 的闭环回归进行候选筛选。
