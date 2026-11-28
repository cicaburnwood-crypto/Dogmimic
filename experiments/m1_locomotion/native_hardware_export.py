"""Export the M1 policy's two-input path for the board without terrain observations."""
from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


def export(source, destination):
    if source.resolve() == destination.resolve():
        raise ValueError("Use a separate path for the hardware policy")
    payload = source.read_bytes()
    model = onnx.load_model_from_string(payload)
    graph = model.graph
    nodes = {node.name: node for node in graph.node}
    terrain, = (tensor for tensor in graph.initializer if tensor.name == "terrain")
    if numpy_helper.to_array(terrain).any():
        raise ValueError("Expected an absent terrain profile by default")

    # Both terrain corrections have zero authority when terrain[0] is zero.
    level = nodes["m1_stability_level_stabilized"]
    stability = nodes["m1_stability_final_action"]
    cruise = nodes["m1_cruise_final_action"]
    stand = nodes["m1_stand_final_action"]
    if (list(level.input) != ["m1_stability_upstream_actions", "m1_stability_active_flat"] or
            list(stability.input) != [level.output[0], "m1_stability_active_descent"] or
            list(stability.output) != [cruise.input[0]] or
            list(cruise.output) != [stand.input[0]] or
            cruise.input[1] != "m1_cruise_active_delta"):
        raise ValueError("Unexpected terrain correction chain")
    stand.input[0] = level.input[0]

    producers = {output: index for index, node in enumerate(graph.node) for output in node.output}
    pending = [output.name for output in graph.output]
    values = set()
    live_nodes = set()
    while pending:
        value = pending.pop()
        if value in values:
            continue
        values.add(value)
        index = producers.get(value)
        if index is not None and index not in live_nodes:
            live_nodes.add(index)
            pending.extend(graph.node[index].input)
    if "terrain" in values:
        raise ValueError("Hardware graph still depends on terrain")

    keep_nodes = [node for index, node in enumerate(graph.node) if index in live_nodes]
    keep_inputs = [value for value in graph.input if value.name in values]
    keep_initializers = [tensor for tensor in graph.initializer if tensor.name in values]
    keep_value_info = [value for value in graph.value_info if value.name in values]
    del graph.node[:], graph.input[:], graph.initializer[:], graph.value_info[:]
    graph.node.extend(keep_nodes)
    graph.input.extend(keep_inputs)
    graph.initializer.extend(keep_initializers)
    graph.value_info.extend(keep_value_info)
    if [value.name for value in graph.input] != ["obs", "obs_history"]:
        raise ValueError("Expected the Classic hardware input names")
    note = model.metadata_props.add()
    note.key, note.value = "m1_hardware_base_sha256", sha256(payload).hexdigest()
    onnx.checker.check_model(model)
    destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, destination)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    export(args.source, args.destination)
