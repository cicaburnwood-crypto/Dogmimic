# 平地与楼梯版测试说明

在本实验目录创建 [仿真环境](README.md) 后执行。

~~~bash
sha256sum -c ASSETS.sha256
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
PY="$PWD/.venv/bin/python"
MODEL=.runtime/native/m1-universal-3a27ea5132a1.onnx
mkdir -p results
"$PY" native_rollout.py --candidate "$MODEL" --suite final --out results/terrain
"$PY" native_rollout.py --candidate "$MODEL" --suite descent-final --out results/descent
~~~

| 测试 | 完成数 |
| --- | ---: |
| 平地 | 64/64 |
| 平地接楼梯混合场景 | 30/30 |
| 18 厘米上行 | 30/30 |
| 20 厘米上行 | 29/30 |
| 18 厘米下行 | 23/24 |
| 20 厘米下行 | 24/24 |

对照 [平地及上行回归](.runtime/native/standing/k110-validation/final.json) 与 [下行回归](.runtime/native/standing/k110-validation/descent-final.json)。两份报告应使用同一个候选 SHA-256，验收字段均为 passed=true。

网页手动复测时，选择上行场景后按 W 直入楼梯。下行从场景下拉框选择对应高度，等待复位完成后按 W。最后回到平地复测 W/S、Q/E、空格及方向键。访问 /state 检查 ready、camera 及模型 SHA-256。

本版使用 MuJoCo 计算的地形输入，以上结论适用于仿真。平地转向问题的 180 场专门回归属于前一个分支提交。
