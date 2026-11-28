"""Stage only the M1 assets, baseline weights and reference controller sources."""
from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
from xml.etree import ElementTree as ET
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parent
DEFAULT_ARCHIVE = Path.home() / "Downloads/dg_fsm_m1-dev-develop_xwp.zip"
ROBOT_DIR = "robot/robot_model/M1-robotdog-asm09/"
MODEL_NAME = "M1-robotdog-asm09-add-knee-half-volume.xml"
BASELINES = ("model_21000_08_07_18_05_11.onnx", "policy_cts.onnx")
SOURCE_FILES = (
    "robot_ctrl/rl_controller/RL_CTS_Loco_Dof_Vel.cpp",
    "robot_ctrl/rl_controller/RL_CTS_Loco_Dof_Vel.h",
    "robot_ctrl/rl_controller/RL_Loco_Classic.cpp",
    "robot_ctrl/rl_controller/RL_Loco_Classic.h",
    "utilities/types/hardware_types.h",
    "config/Config.h",
    "config/Control_Parameters.info",
)


def stage(archive, dest):
    manifest = {}
    with ZipFile(archive) as package:
        for entry in package.infolist():
            if entry.is_dir():
                continue
            parts = PurePosixPath(entry.filename).parts
            if len(parts) < 2 or parts[0] != "dg_fsm_m1-dev-develop_xwp" or ".." in parts:
                continue
            name = PurePosixPath(*parts[1:]).as_posix()
            runtime = name.startswith("third_party/x86_64/onnxruntime-1.16.1/")
            permitted = (
                name.startswith(ROBOT_DIR)
                or name in {f"models/{model}" for model in BASELINES}
                or name in SOURCE_FILES
                or runtime and ("/include/" in name or name.endswith("libonnxruntime.so.1.16.1"))
            )
            if permitted:
                data = package.read(entry)
                target = dest / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                manifest[name] = sha256(data).hexdigest()

    required = {ROBOT_DIR + MODEL_NAME, *(f"models/{m}" for m in BASELINES)}
    if required - manifest.keys():
        raise ValueError(f"Missing archive assets: {sorted(required - manifest.keys())}")
    base = dest / ROBOT_DIR
    robot = ET.parse(base / MODEL_NAME)
    root = robot.getroot()
    body = root.find("worldbody")
    asset = root.find("asset")
    ET.SubElement(asset, "texture", name="test_floor", type="2d", builtin="checker", width="256",
                  height="256", rgb1=".19 .24 .30", rgb2=".23 .29 .35", mark="edge", markrgb=".31 .38 .44")
    ET.SubElement(asset, "material", name="test_floor", texture="test_floor", texrepeat="2 2",
                  texuniform="true", reflectance="0")
    ground = ET.SubElement(body, "geom", name="ground", type="plane", size="0 0 0.05",
                           friction="0.9 0.005 0.0001", material="test_floor")
    ET.SubElement(body, "light", name="key", pos="0 -4 6", dir="0 0.6 -1",
                  diffuse="0.8 0.85 0.9")
    robot.write(base / "flat.xml", encoding="utf-8")
    # Mesh decorations have contype=conaffinity=0, and explicit body inertia is retained.
    del ground.attrib["material"]
    for parent in root.findall(".//body"):
        for geom in parent.findall("geom"):
            if geom.get("class") == "visual":
                parent.remove(geom)
    if any(geom.get("mesh") for geom in root.findall(".//worldbody//geom")):
        raise ValueError("A collision mesh is required by the headless model.")
    for assets in root.findall("asset"):
        root.remove(assets)
    robot.write(dest / "headless.xml", encoding="utf-8")
    (dest / "manifest.json").write_text(json.dumps(
        {"archive": archive.name, "archive_sha256": sha256(archive.read_bytes()).hexdigest(),
         "files": manifest}, indent=2) + "\n")


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--out", type=Path, default=ROOT / ".runtime/m1")
    args = parser.parse_args()
    stage(args.archive, args.out)
    print(args.out / "manifest.json")


if __name__ == "__main__":
    main()
