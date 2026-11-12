import os,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
os.environ['SM_ROBOT']='b2'
sys.path.insert(0,str(HERE/'engine'))
