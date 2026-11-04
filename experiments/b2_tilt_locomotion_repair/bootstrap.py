import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
os.environ['SM_ROBOT'] = 'b2'
sys.path.insert(0, str(HERE / 'engine'))

