"""One-shot read-only snapshot; does not poll or create a scheduler."""
import json
import os
from pathlib import Path
import time
from common import HERE,sha256

def read(path):
    return json.loads(path.read_text()) if path.exists() else None

def main():
    launch=read(HERE/'latest_launch.json')
    result=dict(checked=time.time(),launch=launch,robots={})
    for robot,info in launch['robots'].items():
        out=Path(info['out']);pid=info['pid'];proc=Path('/proc')/str(pid)
        identity=dict(pid=pid,alive=proc.exists())
        if identity['alive']:
            try:identity.update(command=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode(),cwd=os.readlink(proc/'cwd'),session=os.getsid(pid))
            except OSError as e:identity['read_error']=str(e)
        best=read(out/'best_eval.json')
        result['robots'][robot]=dict(process=identity,status=read(out/'status.json'),
            initial=read(out/'initial_eval.json'),latest=read(out/'latest_eval.json'),best=best,
            done=(out/'DONE').exists(),eval_done=read(out/'EVAL_DONE.json'),error=read(out/'ERROR.json'),
            accepted=(out/'accepted.pt').exists(),timing=read(out/'timing.json'),
            best_hash_matches=(sha256(out/'best.pt')==best.get('best_sha256')) if best and (out/'best.pt').exists() else None,
            log_tail=(out/'train.log').read_text(errors='replace').splitlines()[-8:] if (out/'train.log').exists() else [])
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
