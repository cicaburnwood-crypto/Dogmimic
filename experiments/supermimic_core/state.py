"""Complete in-process forks: all Warp arrays, environment tensors, replay and RNG.

No reconstruction from qpos/qvel; CUDA graph allocations stay in place. Snapshot
restoration is usable only after a next-transition replay audit on that instance.
"""
import copy
import random
import numpy as np
import torch
import warp as wp


def plain(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if value is None or isinstance(value, (str, bool, int, float)):
        return copy.deepcopy(value)
    if isinstance(value, dict) and all(isinstance(k, (str,int)) for k in value):
        result={}
        for k,v in value.items():
            q=plain(v)
            if q is UNSUPPORTED:return UNSUPPORTED
            result[k]=q
        return result
    if isinstance(value,(list,tuple)):
        result=[plain(v) for v in value]
        if any(v is UNSUPPORTED for v in result):return UNSUPPORTED
        return type(value)(result)
    return UNSUPPORTED


UNSUPPORTED=object()


def capture_attributes(obj):
    return {k:c for k,v in vars(obj).items() if (c:=plain(v)) is not UNSUPPORTED}


def restore_value(current,value,device):
    if isinstance(value,torch.Tensor):
        if isinstance(current,torch.Tensor) and current.shape==value.shape and current.dtype==value.dtype:
            current.copy_(value.to(current.device));return current
        return value.to(device).clone()
    if isinstance(value,dict):
        return {k:restore_value(current.get(k) if isinstance(current,dict) else None,v,device) for k,v in value.items()}
    return copy.deepcopy(value)


def restore_attributes(obj,attributes,device):
    for k,v in attributes.items():
        setattr(obj,k,restore_value(getattr(obj,k,None),v,device))


def warp_arrays(obj,prefix='',seen=None):
    seen=set() if seen is None else seen
    if id(obj) in seen:return
    seen.add(id(obj))
    for k,v in vars(obj).items():
        key=prefix+k
        if isinstance(v,wp.array):
            if v.size:yield key,v
        elif hasattr(v,'__dict__') and type(v).__module__.startswith('mujoco_warp'):
            yield from warp_arrays(v,key+'.',seen)


def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state() if torch.cuda.is_available() else None)


def restore_rng(s):
    random.setstate(s['python']);np.random.set_state(s['numpy']);torch.set_rng_state(s['torch'])
    if s['cuda'] is not None:torch.cuda.set_rng_state(s['cuda'])


class EnvironmentSnapshot:
    def __init__(self,env):
        self.env=env
        self.attributes=capture_attributes(env)
        self.arrays={}
        for group,obj in [('model',env.sim.wp_model),('data',env.sim.wp_data)]:
            self.arrays[group]={key:wp.to_torch(arr).cpu().clone() for key,arr in warp_arrays(obj)}
        self.rng=rng_state()

    def restore(self):
        env=self.env
        with torch.no_grad():
            for group,obj in [('model',env.sim.wp_model),('data',env.sim.wp_data)]:
                current=dict(warp_arrays(obj))
                assert current.keys()==self.arrays[group].keys(),'Warp fields changed since snapshot'
                for key,value in self.arrays[group].items():wp.to_torch(current[key]).copy_(value)
            restore_attributes(env,self.attributes,env.device)
        restore_rng(self.rng)

    def serializable(self):
        return dict(attributes=self.attributes,arrays=self.arrays,rng=self.rng)


class TrainerSnapshot:
    def __init__(self,t):
        self.t=t
        self.env=EnvironmentSnapshot(t.env)
        self.attributes=capture_attributes(t)
        self.buffer=capture_attributes(t.buf)
        self.modules={k:copy.deepcopy(getattr(t,k).state_dict()) for k in ['policy','wm','critic']}
        self.optimizers={k:copy.deepcopy(getattr(t,k).state_dict()) for k in ['opt_pi','opt_wm','opt_v']}
        self.rng=rng_state()

    def restore(self):
        t=self.t
        self.env.restore()
        restore_attributes(t,self.attributes,t.device)
        restore_attributes(t.buf,self.buffer,t.device)
        for k,v in self.modules.items():getattr(t,k).load_state_dict(v,strict=True)
        for k,v in self.optimizers.items():getattr(t,k).load_state_dict(v)
        restore_rng(self.rng)

    def save(self,path):
        torch.save(dict(environment=self.env.serializable(),attributes=self.attributes,
                        buffer=self.buffer,modules=self.modules,optimizers=self.optimizers,rng=self.rng),path)

    @classmethod
    def load_into(cls,t,path):
        data=torch.load(path,map_location='cpu',weights_only=False)
        instance=cls.__new__(cls);instance.t=t
        instance.env=EnvironmentSnapshot.__new__(EnvironmentSnapshot);instance.env.env=t.env
        for k,v in data['environment'].items():setattr(instance.env,k,v)
        for k in ['attributes','buffer','modules','optimizers','rng']:setattr(instance,k,data[k])
        instance.restore()
        return instance


def verify_replay(env,policy,steps=8):
    snap=EnvironmentSnapshot(env)
    def segment(actions=None):
        out=[]
        with torch.no_grad():
            for step in range(steps):
                action=policy(*env.policy_inputs()) if actions is None else actions[step]
                s,info=env.step(action)
                out.append((action.clone(),{k:v.clone() for k,v in s.items()},
                            info['reward'].clone(),env.xi.clone(),env.frame.clone(),env.ep_id.clone()))
        return out
    first=segment();snap.restore()
    restored_exact=True
    for group,obj in [('model',env.sim.wp_model),('data',env.sim.wp_data)]:
        for k,v in warp_arrays(obj):
            current=wp.to_torch(v).cpu().numpy().tobytes()
            original=snap.arrays[group][k].numpy().tobytes()
            restored_exact=restored_exact and current==original
    second=segment();snap.restore()
    # GPU sparse contact reductions can change float32 low bits; BF16 closed-loop
    # policy outputs then amplify them. Assess restoration using the same actions
    # as well, and retain closed-loop noise for every D0 finite-difference pair.
    open_loop=segment([x[0] for x in first]);snap.restore()
    open_diffs={}
    for a,b in zip(first,open_loop):
        for k in a[1]:
            d=float((a[1][k].float()-b[1][k].float()).abs().max())
            open_diffs[k]=max(open_diffs.get(k,0),d)
    tolerances={k:(5e-4 if k in ('v','w','qd') else 5e-5) for k in open_diffs}
    replay_ok=restored_exact and all(open_diffs[k]<=tolerances[k] for k in open_diffs)
    diffs={};exact=True;per_step=[]
    for a,b in zip(first,second):
        step_diff={}
        pairs=[('action',a[0],b[0]),('reward',a[2],b[2]),('context',a[3],b[3]),
               ('frame',a[4],b[4]),('episode',a[5],b[5])]
        pairs += [(k,a[1][k],b[1][k]) for k in a[1]]
        for k,x,y in pairs:
            d=float((x.float()-y.float()).abs().max())
            diffs[k]=max(diffs.get(k,0),d)
            step_diff[k]=d
            exact=exact and torch.equal(x,y)
        per_step.append(step_diff)
    restoration_ok=restored_exact and per_step[0]['action']==0 and all(x['frame']==0 and x['episode']==0 for x in per_step)
    return dict(passed=restoration_ok,trajectory_reproducible=replay_ok,exact=exact,
                restoration_only=True,restored_warp_arrays_bit_exact=restored_exact,
                open_loop_max_differences=open_diffs,open_loop_tolerances=tolerances,max_differences=diffs,
                all_warp_model_fields=len(snap.arrays['model']),all_warp_data_fields=len(snap.arrays['data']),
                environment_attributes=len(snap.attributes),steps=steps,per_step=per_step)
