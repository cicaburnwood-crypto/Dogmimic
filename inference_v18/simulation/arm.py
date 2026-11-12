"""Exogenous six-joint arm disturbance with swept-path safety projection.

XI extension: phase6, random amplitude6, last position target6, target velocity6,
requested planar speed1, requested |yaw|1. The Actor still controls legs only.
"""
import json
import torch
from pathlib import Path
from contract import DT
from supermimic.rotations import quat_to_mat
HERE=Path(__file__).resolve().parent
ARM_XI_DIM=26
BASE_XI_DIM=81

def segment_distance(a,b,c,d):
    u=b-a;v=d-c;w=a-c
    uu=(u*u).sum(-1);vv=(v*v).sum(-1);uv=(u*v).sum(-1)
    uw=(u*w).sum(-1);vw=(v*w).sum(-1)
    den=uu*vv-uv.square()
    si=(uv*vw-vv*uw)/den.clamp_min(1e-12)
    ti=(uu*vw-uv*uw)/den.clamp_min(1e-12)
    interior=(den>1e-12)&(si>=0)&(si<=1)&(ti>=0)&(ti<=1)
    inner=(w+si[...,None]*u-ti[...,None]*v).square().sum(-1)
    def point(p,x,y):
        e=y-x;t=((p-x)*e).sum(-1)/e.square().sum(-1).clamp_min(1e-12)
        return (p-x-t.clamp(0,1)[...,None]*e).square().sum(-1)
    ds=torch.stack([point(a,c,d),point(b,c,d),point(c,a,b),point(d,a,b),torch.where(interior,inner,torch.full_like(inner,1e6))],-1)
    return (ds.amin(-1)+1e-12).sqrt()

def box_separation(pa,ra,ha,pb,rb,hb):
    """Lower bound on Euclidean separation from the 15 OBB SAT axes."""
    rel=ra.transpose(-1,-2)@rb
    t=(ra.transpose(-1,-2)@(pb-pa)[...,None]).squeeze(-1)
    ab=rel.abs()+1e-7
    da=t.abs()-ha-(ab@hb[...,None]).squeeze(-1)
    db=((rel.transpose(-1,-2)@t[...,None]).squeeze(-1)).abs()-hb-(ab.transpose(-1,-2)@ha[...,None]).squeeze(-1)
    best=torch.maximum(da.amax(-1),db.amax(-1))
    for i in range(3):
        k,l=(i+1)%3,(i+2)%3
        for j in range(3):
            u,v=(j+1)%3,(j+2)%3
            length=(1-rel[...,i,j].square()).clamp_min(1e-12).sqrt()
            gap=(t[...,l]*rel[...,k,j]-t[...,k]*rel[...,l,j]).abs()
            gap-=ha[...,k]*ab[...,l,j]+ha[...,l]*ab[...,k,j]+hb[...,u]*ab[...,i,v]+hb[...,v]*ab[...,i,u]
            gap=torch.where(length>1e-4,gap/length,torch.full_like(gap,-1e4))
            best=torch.maximum(best,gap)
    return best

class ArmController:
    def __init__(self,task,device):
        self.task=task;self.device=device
        self.cfg=json.loads((HERE/'arm_contract.json').read_text());meta=json.loads((HERE/'model.json').read_text())
        def t(x,dtype=torch.float32):return torch.tensor(x,dtype=dtype,device=device)
        self.home=t(self.cfg['home']);self.half=t(self.cfg['half_range_rad'])
        import mujoco
        model=mujoco.MjModel.from_xml_path(str(HERE/'b2_a3.xml'))
        ids=[i for i in range(model.njnt) if 19<=model.jnt_qposadr[i]<25]
        assert len(ids)==6 and all(model.jnt_limited[i] for i in ids)
        self.lower=t(model.jnt_range[ids,0]);self.upper=t(model.jnt_range[ids,1])
        self.half=(self.upper-self.lower)/2
        self.lo=t(self.cfg['workspace_base_min_m']);self.hi=t(self.cfg['workspace_base_max_m'])
        self.offset=t(meta['arm_offsets']);self.R0=quat_to_mat(t(meta['arm_quaternions']))
        caps=self.cfg['capsules'];self.body=t([c['body']-1 for c in caps],torch.long)
        self.ca=t([c['a'] for c in caps]);self.cb=t([c['b'] for c in caps]);self.rad=t([c['r'] for c in caps])
        self.bc=t([c['box_center'] for c in caps]);self.br=t([c['box_rotation'] for c in caps]);self.bh=t([c['box_half'] for c in caps])
        self.pairs=t(self.cfg['pairs'],torch.long);self.armcaps=t(self.cfg['arm_capsules'],torch.long)
        self.pair_margins=t(self.cfg['pair_margins_m'])
        self.freq=t([.65,.8,.73,.85,.9,.7]);self.margin=self.cfg['margin_m']
        import mujoco
        model=mujoco.MjModel.from_xml_path(str(HERE/'b2_a3.xml'))
        self.mass=t(model.body_mass[14:20]);self.com=t(model.body_ipos[14:20])
        self._safety=torch.compile(self.safety,dynamic=False)
        self._advance=torch.compile(self.advance,dynamic=False)
        self.compiled=False

    def advance_fast(self,*args):return self._advance(*args) if self.compiled else self.advance(*args)
    def safety_fast(self,*args):return self._safety(*args) if self.compiled else self.safety(*args)

    def envelope(self,speed):
        # Piecewise linear knots: 0/.5/1/2 m/s -> 1/.75/.5/.2.
        v=speed.clamp(0,2)
        scale=torch.where(v<=1,1-.5*v,.5-.3*(v-1)).clamp(.2,1)
        return scale,.6*scale

    def fk(self,q):
        shape=q.shape[:-1];P=q.new_zeros(shape+(3,));R=torch.eye(3,device=q.device).expand(shape+(3,3))
        ps=[];rs=[]
        for i in range(6):
            P=P+(R@self.offset[i,:,None]).squeeze(-1)
            c=q[...,i].cos();s=q[...,i].sin();z=torch.zeros_like(c);o=torch.ones_like(c)
            rz=torch.stack([c,-s,z,s,c,z,z,z,o],-1).reshape(shape+(3,3))
            R=R@self.R0[i]@rz;ps.append(P);rs.append(R)
        return torch.stack(ps,-2),torch.stack(rs,-3)

    def gravity_torque(self,s,mass_scale):
        p,r=self.fk(s['aq'])
        pc=p+(r@self.com[...,None]).squeeze(-1)
        gravity=(s['rot'].transpose(-1,-2)@s['pos'].new_tensor([0.,0.,-9.81])[...,None]).squeeze(-1)
        force=gravity[...,None,:]*self.mass[...,None]*mass_scale[...,None,None]
        torques=[]
        for i in range(6):
            moment=torch.cross(pc[...,i:,:]-p[...,i,None,:],force[...,i:,:],dim=-1).sum(-2)
            torques.append(-(moment*r[...,i,:,2]).sum(-1))
        return torch.stack(torques,-1)

    def safety_details(self,s,aq):
        P,R=self.task.kin.forward(s['pos'],s['rot'],s['q'])
        ap,ar=self.fk(aq)
        awp=s['pos'][...,None,:]+(s['rot'][...,None,:,:]@ap[...,None]).squeeze(-1)
        awr=s['rot'][...,None,:,:]@ar
        P=torch.cat([P,awp],-2);R=torch.cat([R,awr],-3)
        bp=P[...,self.body,:]+(R[...,self.body,:,:]@self.bc[...,None]).squeeze(-1)
        br=R[...,self.body,:,:]@self.br
        i,j=self.pairs.unbind(-1)
        dist=box_separation(bp[...,i,:],br[...,i,:,:],self.bh[i],bp[...,j,:],br[...,j,:,:],self.bh[j])
        minimum=dist.amin(-1)
        ground=(bp[...,self.armcaps,2]-(br[...,self.armcaps,2,:].abs()*self.bh[self.armcaps]).sum(-1)).amin(-1)
        tcp=ap[...,-1,:]+ar[...,-1,:,2]*.125
        workspace=torch.ones_like(tcp[...,0],dtype=torch.bool)
        bounds=((aq>=self.lower)&(aq<=self.upper)).all(-1)
        valid=(dist>=self.pair_margins).all(-1)&(ground>=self.margin)&workspace&bounds
        return valid,minimum,ground,tcp,torch.relu(self.pair_margins-dist).amax(-1),dist-self.pair_margins

    def safety(self,s,aq):
        return self.safety_details(s,aq)[:4]

    def propose(self,s,xi):
        # Motion is an external disturbance, so its safety decisions do not
        # provide a reward gradient that the leg Actor can exploit.
        ss={k:v.detach() for k,v in s.items()};x=xi[...,81:].detach()
        phase,amplitude,last,velocity=x[...,:6],x[...,6:12],x[...,12:18],x[...,18:24]
        speed=torch.maximum(ss['v'][...,:2].norm(dim=-1),x[...,24])
        speed=torch.maximum(speed,.4*torch.maximum(ss['w'][...,2].abs(),x[...,25]))
        scale,limit=self.envelope(speed)
        phase=torch.remainder(phase+self.freq*DT,2*torch.pi)
        desired=self.home+self.half*scale[...,None]*amplitude*phase.sin()
        want=((desired-last)/DT).clamp(-limit[...,None],limit[...,None])
        vel=velocity+(want-velocity).clamp(-1.2*DT,1.2*DT)
        proposal=last+vel*DT
        # Cheap exogenous proposal for learned-model imagination only. The
        # physical simulator retains advance() and its complete safety guard.
        extension=torch.cat([phase,amplitude,proposal,vel,x[...,24:26]],-1)
        return proposal,extension

    def advance(self,s,xi):
        # Motion is an external disturbance, so its safety decisions do not
        # provide a reward gradient that the leg Actor can exploit.
        ss={k:v.detach() for k,v in s.items()};x=xi[...,81:].detach()
        phase,amplitude,last,velocity=x[...,:6],x[...,6:12],x[...,12:18],x[...,18:24]
        speed=torch.maximum(ss['v'][...,:2].norm(dim=-1),x[...,24])
        speed=torch.maximum(speed,.4*torch.maximum(ss['w'][...,2].abs(),x[...,25]))
        scale,limit=self.envelope(speed)
        phase=torch.remainder(phase+self.freq*DT,2*torch.pi)
        desired=self.home+self.half*scale[...,None]*amplitude*phase.sin()
        want=((desired-last)/DT).clamp(-limit[...,None],limit[...,None])
        vel=velocity+(want-velocity).clamp(-1.2*DT,1.2*DT)
        proposal=last+vel*DT
        # All probes on the short target sweep plus measured->target corridor.
        valid=torch.ones_like(speed,dtype=torch.bool)
        for blend in (0.,.5,1.):
            q=ss['aq']+blend*(proposal-ss['aq'])
            valid=valid & self.safety(ss,q)[0]
        target=torch.where(valid[...,None],proposal,last)
        vel=torch.where(valid[...,None],vel,torch.zeros_like(vel))
        extension=torch.cat([phase,amplitude,target,vel,x[...,24:26]],-1)
        return target,extension,~valid

    def reset(self,n,randomize=True):
        phase=torch.rand(n,6,device=self.device)*2*torch.pi
        amp=.5+.5*torch.rand(n,6,device=self.device)
        if randomize:amp[torch.rand(n,device=self.device)<.2]=0
        else:amp[:]=1
        return torch.cat([phase,amp,self.home.expand(n,-1),torch.zeros(n,8,device=self.device)],-1)
