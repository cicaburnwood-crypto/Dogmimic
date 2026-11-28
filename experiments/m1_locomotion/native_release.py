"""Publish a single M1 policy after the independent final MuJoCo suite passes."""
from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path

from native_rollout import acceptance, descent_acceptance, suite
from native_world import NATIVE_RUNTIME, NATIVE_SHA256


def release(checkpoint, validation, training, motion_check=None, front_check=None, flat_check=None,
            level_check=None, descent_check=None, cruise_check=None, stand_check=None):
    payload = checkpoint.read_bytes()
    digest = sha256(payload).hexdigest()
    result = json.loads(validation.read_text())
    run = json.loads(training.read_text())
    final_seeds = {trial.seed for trial in suite("final")}
    records = result.get("records", [])
    if (result.get("suite") != "final" or result.get("policy") != "candidate"
            or len(records) != len(final_seeds) or {r["seed"] for r in records} != final_seeds
            or result.get("sha256", {}).get("candidate") != digest
            or not acceptance(result["metrics"])):
        raise ValueError("Expected a full passing final report for this ONNX")
    if (run.get("phase") != "complete" or not run.get("actor_updates") or run.get("teacher_policies") != NATIVE_SHA256
            or run.get("supermimic", {}).get("repo") != "HybridRobotics/supermimic"):
        raise ValueError("Expected a completed original SuperMimic run with the two native teachers")
    native = NATIVE_RUNTIME.resolve()
    motion = None
    if motion_check is not None:
        motion = json.loads(motion_check.read_text())
        if motion.get("candidate_sha256") != digest or not motion.get("passed"):
            raise ValueError("Expected passing motion regression results for this ONNX")
    if front_check is not None:
        front = json.loads(front_check.read_text())
        if (front.get("candidate_sha256") != digest or front.get("suite") != "final"
                or front.get("front_order") != ["LF", "RF"]
                or run.get("front_lift", {}).get("legs") != front["front_order"]):
            raise ValueError("Expected a matched two-foreleg validation report for this ONNX")
    if flat_check is not None:
        flat = json.loads(flat_check.read_text())
        if flat.get("candidate_sha256") != digest or not flat.get("passed"):
            raise ValueError("Expected passing flat keyboard validation for this ONNX")
    if level_check is not None:
        level = json.loads(level_check.read_text())
        if level.get("candidate_sha256") != digest or level.get("suite") != "final" or not level.get("passed"):
            raise ValueError("Expected passing level trunk and joint measurements for this ONNX")
    if descent_check is not None:
        descent = json.loads(descent_check.read_text())
        seeds = {trial.seed for trial in suite("descent-final")}
        runs = descent.get("records", [])
        if (descent.get("sha256", {}).get("candidate") != digest or descent.get("suite") != "descent-final"
                or len(runs) != len(seeds) or {r["seed"] for r in runs} != seeds
                or not descent_acceptance(descent["metrics"])):
            raise ValueError("Expected a passing held-out descent report for this ONNX")
    if cruise_check is not None:
        cruise = json.loads(cruise_check.read_text())
        if cruise.get("candidate_sha256") != digest or not cruise.get("passed"):
            raise ValueError("Expected passing sustained cruise validation for this ONNX")
    if stand_check is not None:
        stand = json.loads(stand_check.read_text())
        if stand.get("candidate_sha256") != digest or not stand.get("passed"):
            raise ValueError("Expected passing sustained standing validation for this ONNX")
    dest = native / f"m1-universal-{digest[:12]}.onnx"
    temp = dest.with_suffix(".partial")
    temp.write_bytes(payload)
    temp.replace(dest)
    manifest = {"checkpoint": dest.name, "sha256": digest, "name": "M1 平地与楼梯通用策略",
                "passed": True, "training": str(training.resolve().relative_to(native)),
                "validation": str(validation.resolve().relative_to(native)), "teacher_policies": NATIVE_SHA256,
                "seed": run["seed"], "supermimic": run["supermimic"]["commit"],
                "metrics": result["metrics"]}
    if motion is not None:
        manifest["motion_check"] = str(motion_check.resolve().relative_to(native))
        manifest["motion_base_sha256"] = motion["baseline_sha256"]
    if front_check is not None:
        manifest["front_check"] = str(front_check.resolve().relative_to(native))
    if flat_check is not None:
        manifest["flat_check"] = str(flat_check.resolve().relative_to(native))
    if level_check is not None:
        manifest["level_check"] = str(level_check.resolve().relative_to(native))
    if descent_check is not None:
        manifest["descent_check"] = str(descent_check.resolve().relative_to(native))
    if cruise_check is not None:
        manifest["cruise_check"] = str(cruise_check.resolve().relative_to(native))
    if stand_check is not None:
        manifest["stand_check"] = str(stand_check.resolve().relative_to(native))
    temp = native / "release.json.tmp"
    temp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
    temp.replace(native / "release.json")
    print(dest)


if __name__ == "__main__":
    p = ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("validation", type=Path)
    p.add_argument("training", type=Path)
    p.add_argument("--motion-check",type=Path)
    p.add_argument("--front-check", type=Path)
    p.add_argument("--flat-check", type=Path)
    p.add_argument("--level-check", type=Path)
    p.add_argument("--descent-check", type=Path)
    p.add_argument("--cruise-check", type=Path)
    p.add_argument("--stand-check", type=Path)
    args = p.parse_args()
    release(args.checkpoint,args.validation,args.training,args.motion_check,args.front_check,args.flat_check,
            args.level_check,args.descent_check,args.cruise_check,args.stand_check)
