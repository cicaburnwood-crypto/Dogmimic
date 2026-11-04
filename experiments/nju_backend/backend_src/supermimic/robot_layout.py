"""Task dimensions only; selected before importing the NJU training backend."""
import os
ROBOT = os.environ.get("SM_NJU_ROBOT", "G1")
assert ROBOT in ("B2", "G1")
NJ = 12 if ROBOT == "B2" else 29
NQ, NV = 7 + NJ, 6 + NJ
SD, DD, GF, OD = 13 + 2 * NJ, 12 + 2 * NJ, 10 + 2 * NJ, 6 + 3 * NJ
XD = NJ + 4 + (2 if ROBOT == "B2" else 0)
ROOT_NAME = "base_link" if ROBOT == "B2" else "pelvis"
