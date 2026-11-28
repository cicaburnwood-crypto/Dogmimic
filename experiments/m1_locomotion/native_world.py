"""M1 stair geometry shared by replay, world-model context and the sim viewer."""
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

from sim import M1_HEADLESS, RUNTIME

NATIVE_RUNTIME = RUNTIME / "native"
CLASSIC_FILE = "model_21000_08_07_18_05_11.onnx"
STAIRS_FILE = "policy_loco_stairs_new.onnx"
NATIVE_FILES = {"classic": CLASSIC_FILE, "stairs": STAIRS_FILE}
NATIVE_SHA256 = {
    "classic": "29d6582cd8febc7dbd1d9bfdabb4023677300d8b7d5a08685809ab43fb489cfb",
    "stairs": "d7c511f635d96e5f876088312a6bb9c4c0fda0f5c3924117f747611126a37685",
}
HARDWARE_CLASSIC_SHA256 = "1e158d791095d0e064e3ab92079eb749b3039affcd4adcf283ace7e1fabbd0ea"
CLASSIC_RETRAINED_KEYS = tuple(f"actor_critic.actor.{layer}.{part}" for layer in (4, 6) for part in ("weight", "bias"))
CONTACT_TRIM_KEYS = ("contact_gain", "steering_gain")
CONTACT_CLEARANCE = .004
CONTACT_ACTION_LIMIT = .42
CONTACT_TRAVEL = (.06, .13)
CONTACT_YAW = (.04, .13)
CONTACT_SIDE = (.025, .09)
CONTACT_DIRECTION = (-.10, .10)
ARCHIVE = Path.home() / "Downloads/dg_fsm_m1-dev-develop_xwp.zip"
ARCHIVE_ROOT = "dg_fsm_m1-dev-develop_xwp/"
MAX_STAIRS = 10
FLAT, STAIRS, MIXED, DESCENT = range(4)
FRONT_WHEELS = (0, 2)
FRONT_ACTIONS = (1, 2, 3, 9, 10, 11)
FRONT_HIDDEN = 96
FRONT_LIMIT = (.26, .34, .34)
# Ranges in the archived policy's normalized observation units.
FRONT_FORWARD = (.29, .54)
FRONT_PITCH = (.055, .22)
FRONT_STALL = (.07, .155)
FRONT_DEFICIT = (.08, .18)


def contact_action_maps():
    legs = np.zeros((4, 16), dtype=np.float32)
    for leg, sign in enumerate((1., -1., 1., -1.)):
        # Extend the knee while compensating longitudinal wheel motion at the hip.
        legs[leg, 4*leg+1] = -.57 * sign
        legs[leg, 4*leg+2] = sign
    wheels = np.zeros(16, dtype=np.float32)
    wheels[3::4] = (-1., -1., 1., 1.)
    return legs, wheels


def observation_reflection():
    order = np.arange(49)
    sign = np.ones(49, dtype=np.float32)
    sign[[0, 2, 4, 7, 8]] = -1
    for start, width in ((9, 3), (21, 1), (25, 4), (41, 2)):
        order[start:start + 4*width] = (start + width*np.array([2, 3, 0, 1])[:, None]
                                        + np.arange(width)).ravel()
        if width in (3, 4):
            sign[start:start + 4*width:width] = -1
    return order, sign


@dataclass(frozen=True)
class Course:
    rise: float = 0.
    tread: float = .25
    steps: int = 0
    edge: float = 1.1
    half_width: float = 1.25
    friction: float = .9

    @property
    def top(self):
        return self.rise * self.steps

    @property
    def end(self):
        return self.edge + self.tread * self.steps

    @property
    def descending_start(self):
        return self.end + 1.02

    def context(self):
        return np.asarray((self.rise, self.tread, self.steps, self.edge, self.half_width, self.friction), np.float32)

    def write(self, target, source=M1_HEADLESS):
        tree = ET.parse(source)
        root = tree.getroot()
        meshdir = root.find("compiler").get("meshdir")
        if meshdir:
            root.find("compiler").set("meshdir", str((source.parent / meshdir).resolve()))
        body = root.find("worldbody")
        friction = f"{self.friction} 0.005 0.0001"
        body.find("geom[@name='ground']").set("friction", friction)
        spacing = self.tread
        for i in range(self.steps):
            z = (i + 1) * self.rise / 2
            x = self.edge + (i + .5) * spacing
            color = ".41 .54 .65 1" if i % 2 == 0 else ".33 .44 .54 1"
            ET.SubElement(body, "geom", name=f"step_{i + 1}", type="box", rgba=color,
                          pos=f"{x} 0 {z}", size=f"{spacing / 2} {self.half_width} {z}", friction=friction)
        if self.steps:
            ET.SubElement(body, "geom", name="landing", type="box", rgba=".34 .50 .46 1",
                          pos=f"{self.end + 1.35} 0 {self.top / 2}",
                          size=f"1.35 {self.half_width} {self.top / 2}", friction=friction)
        tree.write(target, encoding="utf-8")
