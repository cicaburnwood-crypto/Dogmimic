import hashlib
import json
from pathlib import Path
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]

def write_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)

def sha256(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()

def tensor_hash(mapping):
    import torch
    h=hashlib.sha256()
    for key,value in sorted(mapping.items()):
        if isinstance(value,torch.Tensor):
            v=value.detach().cpu().contiguous()
            h.update(key.encode());h.update(str((v.shape,v.dtype)).encode())
            h.update(v.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()

def clean_environment(robot):
    import os
    env=os.environ.copy()
    for key in list(env):
        if key.startswith(('B2_','G1_','SM_TASK','SM_ROBOT','SMIMIC_PROFILE','TORCHELASTIC_')):
            env.pop(key)
        if key in ('RANK','LOCAL_RANK','WORLD_SIZE','LOCAL_WORLD_SIZE','GROUP_RANK',
                   'ROLE_RANK','ROLE_WORLD_SIZE','MASTER_ADDR','MASTER_PORT'):
            env.pop(key,None)
    env['SM_RESET_ROBOT']=robot
    env['PYTHONPATH']=str(HERE)
    env['OMP_NUM_THREADS']='2';env['MKL_NUM_THREADS']='2';env['PYTHONUNBUFFERED']='1'
    return env
