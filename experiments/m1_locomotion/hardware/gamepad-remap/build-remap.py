"""Build the changed control-path object with the running M1 build's recipe."""
from argparse import ArgumentParser
from pathlib import Path
import shlex
import subprocess


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("stage", type=Path)
    parser.add_argument("release", type=Path)
    args = parser.parse_args()
    stage, release = args.stage.resolve(), args.release.resolve()
    robot_build = stage / "build/robot"
    cmake_target = robot_build / "CMakeFiles/hardware_library.dir"
    flags = {}
    for line in (cmake_target / "flags.make").read_text().splitlines():
        key, equals, value = line.partition(" = ")
        if equals and key.startswith("CXX_"):
            flags[key] = shlex.split(value)
    compile_flags = sum((flags[key] for key in ("CXX_DEFINES", "CXX_INCLUDES", "CXX_FLAGS")), [])
    link = shlex.split((cmake_target / "link.txt").read_text())
    compiler = link[0]
    obj = release / "HighLevelPath.cpp.o"
    include_origin = stage / "robot/ControlPath/src"
    subprocess.run([compiler, *compile_flags, "-iquote", str(include_origin), "-c", release / "HighLevelPath.cpp",
                    "-o", obj], check=True)
    index, = (i for i, argument in enumerate(link) if argument.endswith("/ControlPath/src/HighLevelPath.cpp.o"))
    link[index] = str(obj)
    link[link.index("-o") + 1] = str(release / "libhardware_library.so")
    subprocess.run(link, check=True, cwd=robot_build)

    control = stage / "build/robot_ctrl"
    ort = stage / "third_party/aarch64/onnxruntime-1.16.1/lib"
    libs = (release, control, ort, stage / "robot/dds_comm/thirdparty/lib/aarch64", stage / "utilities/src")
    wrap = ("_ZN10Controller7computeERK13HardwareState8FsmStateRK11PathCommandb",
            "_ZN16StateVoicePlayer7requestENS_3CueE")
    subprocess.run([compiler, *compile_flags, "-I" + str(stage), release / "check-remap.cpp", obj,
                    *("-L" + str(path) for path in libs), "-lhardware_library", "-lrobot_ctrl_library", "-lonnxruntime",
                    "-Wl,-rpath," + ":".join(map(str, libs)), *("-Wl,--wrap=" + name for name in wrap),
                    "-o", release / "check-remap"], check=True)


if __name__ == "__main__":
    main()
