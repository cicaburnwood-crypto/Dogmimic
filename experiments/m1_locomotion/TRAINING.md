# 上楼与下楼训练复现

上楼使用 native_train.py 的上行进度、接地、侧偏和双前轮越阶奖励。第 1124 轮在平地及楼梯教师、学生回采和双前腿数据上进行 BPTT。下楼使用 native_stability_train.py 的第 2418 轮，在上楼基底上优化平地稳定和下行。下行奖励含进度、速度跟踪、冲击、俯仰、横滚、偏航和车道偏移。当前 ONNX 保留两阶段结果。

训练源码使用 [HybridRobotics 原版 SuperMimic](https://github.com/HybridRobotics/supermimic) 的 abc124c1bd5976341034668af3d12c087f2f1f55，同级 mjlab_research 提交为 dd51a9fcc98791a2f0b018b3c7f71a3eefd5ecdd。SM_PY 指向该源码对应的 CUDA Python 环境。以下命令从本实验目录执行。

~~~bash
SM_PY=/absolute/path/to/supermimic/.venv/bin/python
N=.runtime/native
T="$N/terrain_training"
B="$N/bilateral"
mkdir -p results training-runs
cat "$N"/train.npz.part-* > results/train.npz
printf '%s  %s\n' 7ef4348580eb72e4f18893daa58c71bceb0c44cf878ba6cde7e978a379543d5e results/train.npz | sha256sum -c -
cp "$N/train.json" results/train.json
export CUDA_VISIBLE_DEVICES=0 TORCH_COMPILE_DISABLE=1
"$SM_PY" native_train.py \
  --train results/train.npz --holdout "$N/holdout.npz" --teachers "$N/teachers" \
  --extra "$N/aggregate.npz" --extra "$B/front-replay.npz" \
  --front-lift --init-policy "$N/projected/run-823/policy-0144.npz" \
  --seed 1124 --world-updates 2600 --actor-updates 512 --eager \
  --out training-runs/run-1124
"$SM_PY" native_stability_train.py \
  --train results/train.npz --holdout "$N/holdout.npz" \
  --base "$N/m1-universal-fe53c0fe605b.onnx" \
  --base-report "$B/run-1124/training.json" \
  --extra "$N/aggregate.npz" --extra "$B/front-replay.npz" \
  --extra "$T/level-stable.npz" --extra "$T/descent.npz" \
  --holdout-extra "$T/descent-holdout.npz" \
  --seed 2418 --world-updates 2800 --warm-up 480 --actor-updates 512 \
  --out training-runs/run-2418
~~~

下楼重训可直接从交付的上楼基底开始。训练报告位于 [上楼记录](.runtime/native/bilateral/run-1124/training.json) 和 [下楼记录](.runtime/native/terrain_training/run-2418/training.json)。发布选择第 2418 轮的 policy-0016.npz，再加平地巡航和静止站立修正。

~~~bash
PY="$PWD/.venv/bin/python"
N=.runtime/native
mkdir -p results
"$PY" native_stability_export.py \
  "$N/m1-universal-fe53c0fe605b.onnx" \
  "$N/terrain_training/run-2418/policy-0016.npz" results/terrain-learned.onnx
"$PY" native_cruise_fix.py results/terrain-learned.onnx results/terrain-cruise.onnx
"$PY" native_stand_fix.py results/terrain-cruise.onnx results/terrain-release.onnx
sha256sum results/terrain-release.onnx
~~~
