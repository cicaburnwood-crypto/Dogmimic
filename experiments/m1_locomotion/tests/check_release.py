"""Verify the published M1 assets and saved observation replays."""
from hashlib import file_digest
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
import onnx

APP = Path(__file__).resolve().parents[1]
NATIVE = APP / ".runtime/native"
RELEASE = NATIVE / "contact-turn"
sys.path.insert(0, str(APP))

from controllers import session


def verify_assets():
    count = 0
    for entry in (APP / "ASSETS.sha256").read_text().splitlines():
        expected, name = entry.split("  ", 1)
        path = APP / name
        with path.open("rb") as handle:
            actual = file_digest(handle, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"Unexpected SHA-256 for {name}")
        count += 1
    return count


def graph_nodes(graph):
    for node in graph.node:
        yield node
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.GRAPH:
                yield from graph_nodes(attribute.g)


def seeded_session(path, directory, seed):
    model = onnx.load(path)
    graph = model.graph
    inputs = {value.name: [axis.dim_value for axis in value.type.tensor_type.shape.dim] for value in graph.input}
    output, = graph.output
    expected = {"obs": [1, 49], "obs_history": [1, 245]}
    if inputs != expected or output.name != "actions" or [axis.dim_value for axis in output.type.tensor_type.shape.dim] != [1, 16]:
        raise ValueError(f"Unexpected controller interface in {path.name}")
    noise = sorted((node for node in graph_nodes(graph) if node.op_type == "RandomNormalLike"), key=lambda node: node.name)
    for index, node in enumerate(noise):
        if "seed" in [attribute.name for attribute in node.attribute]:
            raise ValueError(f"Expected a deployable policy in {path.name}")
        node.attribute.append(onnx.helper.make_attribute("seed", seed + index))
    seeded = directory / path.name
    onnx.save(model, seeded)
    return session(seeded)


def maximum_action_delta(sessions, obs, ticks):
    maximum = 0.
    for tick in ticks:
        inputs = {"obs": obs[tick:tick+1], "obs_history": obs[tick-5:tick].reshape(1, -1)}
        left, right = (actor.run(["actions"], inputs)[0] for actor in sessions)
        if not (np.isfinite(left).all() and np.isfinite(right).all()):
            raise ValueError("A policy emitted nonfinite actions")
        maximum = max(maximum, float(np.max(np.abs(left - right))))
    return maximum


def turn_frames(data):
    report = json.loads((RELEASE / "hardware/arm-turn-fixtures.json").read_text())
    selected = [index for index, row in enumerate(report["records"])
                if abs(row["command"][2]) == .35 and row["command"][0] in (.35, .55, -.34)
                and row["command"][1] == 0.]
    return np.concatenate([np.arange(data["start"][index]+45, data["start"][index]+368) for index in selected])


def stair_frames(data):
    path = NATIVE / "hardware/uni-exp/stair-recovery/contact-replay.json"
    rows = json.loads(path.read_text())["first_step"]
    offsets = {int(seed): int(start) for seed, start in zip(data["seed"], data["start"], strict=True)}
    samples = (offsets[row["seed"]] + round(row["first_contact_s"]/.02) for row in rows)
    return np.concatenate([np.arange(touch-5, touch+6) for touch in samples])


def main():
    asset_count = verify_assets()
    dense = RELEASE / "uni-exp-contact-turn.onnx"
    board = RELEASE / "hardware/uni-exp-conditional.onnx"
    restored = NATIVE / "hardware/uni-exp/uni-exp-hardware.onnx"
    with TemporaryDirectory(prefix="m1-release-check-") as work:
        scratch = Path(work)
        actors = [seeded_session(path, scratch, 730.) for path in (dense, board)]
        with np.load(RELEASE/"hardware/arm-turn-fixtures.npz", allow_pickle=False) as replay:
            frames = turn_frames(replay)
            turn_delta = maximum_action_delta(actors, replay["obs"], frames)
            if len(frames) != 3876 or turn_delta > 2e-5:
                raise ValueError(f"Dense and ARM policy mismatch: {len(frames)} frames, {turn_delta}")
        actors = [seeded_session(path, scratch, 287.) for path in (restored, dense)]
        with np.load(NATIVE/"classic-posture/stair-replay.npz", allow_pickle=False) as replay:
            frames = stair_frames(replay)
            stair_delta = maximum_action_delta(actors, replay["obs"], frames)
            if len(frames) != 550 or stair_delta > 1e-6:
                raise ValueError(f"Straight first-step parity failed: {len(frames)} frames, {stair_delta}")
    print(json.dumps({"assets_verified": asset_count, "turn_frames": 3876, "turn_max_action_delta": turn_delta,
                      "stair_frames": 550, "stair_max_action_delta": stair_delta}))


if __name__ == "__main__":
    main()
