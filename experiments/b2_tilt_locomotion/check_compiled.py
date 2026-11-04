"""Inspect actual compiled smoke optimizer updates and export equivalence."""
import argparse,json
from pathlib import Path
import torch
from bootstrap import HERE
from common import write_json
from export import export

def check(run,out):
    run=Path(run);ck=torch.load(run/'last.pt',map_location='cpu',weights_only=False)
    meta=json.loads((run/'run_meta.json').read_text())
    initial=torch.load(run/'initial.pt',map_location='cpu',weights_only=False)
    finite=True;changed={}
    for name in ('policy','wm','critic'):
        a=ck[name];b=initial[name]
        finite=finite and all(torch.isfinite(v).all().item() for v in a.values())
        changed[name]=any(not torch.equal(a[k],b[k]) for k in a)
    assert all(changed.values()) and finite
    report=dict(iterations=ck['it'],updates=ck['updates'],finite=finite,changed=changed,
        num_envs=meta['cfg']['num_envs'],compile_wm=meta['cfg']['compile_wm'],
        export_max_error=export(run/'last.pt',run/'checked_policy.pt'),physical_acceptance=False)
    assert report['iterations']==10 and report['updates']>=4
    write_json(out,report);print(json.dumps(report))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--out',required=True);a=p.parse_args();check(a.run,a.out)
