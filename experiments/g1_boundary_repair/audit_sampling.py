import json,importlib.util,torch
from pathlib import Path
from sampling import sample_domain,HARD_CASES
root=Path(__file__).resolve().parents[2]
torch.set_num_threads(2);torch.manual_seed(20261102)
n=100000;friction=.15+1.35*torch.rand(n)
spec=importlib.util.spec_from_file_location('old_sampling',root/'experiments/g1_contact_repair/sampling.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
xo,mo,go=old.sample_domain(n,'cpu',friction)
x,mu,g=sample_domain(n,'cpu',friction)
counts={c['name']:int((g['hard_id']==i).sum()) for i,c in enumerate(HARD_CASES)}
assert all(4000<k<6000 for k in counts.values()),counts
for i,c in enumerate(HARD_CASES):
 m=g['hard_id']==i
 assert torch.allclose(mu[m],torch.full_like(mu[m],c.get('friction',.8)))
 for j,key in enumerate(['mass','kp','kd','effort']):assert torch.allclose(x[m,j],torch.full_like(x[m,j],c.get(key,1.)))
 assert torch.allclose(x[m,4],torch.full_like(x[m,4],c.get('delay_ms',0)/40))
 assert torch.allclose(x[m,6:],torch.tensor(c.get('force',[0,0,0]),dtype=x.dtype).expand(int(m.sum()),3))
 assert (g['measured_start'][m]==bool(c.get('measured',False))).all()
assert (g['nominal'].int()+g['continuous'].int()+g['isolated_weak'].int()+g['crossed_weak'].int()+(g['hard_id']>=0).int()==1).all()
continu=g['continuous'];assert mu[continu].min()<.151 and mu[continu].max()>1.499
identity=torch.zeros(9);identity[:4]=1
old_pure_low={str(v):int(((mo==v)&(xo==identity).all(-1)).sum()) for v in [.15,.2,.3]}
back=identity.clone();back[6]=-25
report=dict(passed=True,samples=n,targeted_counts=counts,old_isolated_low_counts=old_pure_low,old_pure_back_push=int(((xo==back).all(-1)&(mo==.8)).sum()),continuous_samples=int(continu.sum()),scope='Domain-coverage audit only; not physical skill evaluation or convergence')
p=root/'experiments/g1_boundary_repair/sampling_audit.json';p.write_text(json.dumps(report,indent=2));print(json.dumps(report))
