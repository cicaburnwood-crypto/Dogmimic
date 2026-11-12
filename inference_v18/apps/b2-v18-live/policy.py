"""V18 24500 manager export and exact command contract. No hardware I/O."""
import sys,json,math,hashlib
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from teleop import ControlFault
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'b2-arm-reach'))
from geometry import Geometry,NOMINAL,tilt_limit
DT=.02
SHA='d5540cc2f1a8a283c38c9e644c9693d2b6317158d9beb2172f6d67c0594da703'
class Manager:
 def __init__(self):
  self.manifest=json.loads((ROOT/'models/manifest.json').read_text())
  if self.manifest['checkpoint_sha256']!=SHA or hashlib.sha256((ROOT/'models/manager.npz').read_bytes()).hexdigest()!=self.manifest['manager_sha256']:raise ValueError('联合策略权重校验失败')
  self.weights=dict(np.load(ROOT/'models/manager.npz'));assert self.weights['net.0.weight'].shape==(256,99)
 def __call__(self,obs):
  x=np.asarray(obs,np.float32)
  if x.shape!=(99,) or not np.isfinite(x).all():raise ValueError('V18 input must be 99 finite values')
  for i in [0,2,4]:
   x=self.weights[f'net.{i}.weight']@x+self.weights[f'net.{i}.bias']
   x=np.tanh(x) if i==4 else x/(1+np.exp(-np.clip(x,-80,80)))
  if not np.isfinite(x).all():raise ValueError('nonfinite V18 action')
  return x

def observe(s,ctx,goal_pos,goal_rot,last,target,velocity):
 rot=s['rot'];rel=rot.T@(goal_pos-s['ee'])/.5;rr=(s['eerot'].T@goal_rot).ravel()
 return np.concatenate([rel,rr,rot.T@s['v']/2,s['gyro'],-rot[2],s['q'],s['qd']/10,s['aq'],s['aqd'],ctx,last,[s['pos'][2]],target,velocity]).astype(np.float32)
def commands(u):
 xy=u[:2]*[2,1];xy=xy*min(1,2/max(np.linalg.norm(xy),1e-8));h=(.05 if u[5]>=0 else .2)*u[5];lim=tilt_limit(h)
 return np.array([*xy,.5*u[2],*(u[3:5]*lim),h],np.float32)
def arm_action(aq,aqd,v,w,requested,u,last,oldvel,lower,upper):
 speed=max(np.linalg.norm(v[:2]),np.linalg.norm(requested[:2]),.4*max(abs(w[2]),abs(requested[2])))
 speed=np.clip(speed,0,2);scale=np.clip(1-.5*speed if speed<=1 else .5-.3*(speed-1),.2,1);limit=.6*scale
 wanted=u[6:]*limit;lo=lower+np.array([.030,.030,.025,.020,.025,.025]);hi=upper-np.array([.030,.030,.025,.020,.025,.025])
 up=np.maximum(hi-np.maximum(last,aq),0);down=np.maximum(np.minimum(last,aq)-lo,0)
 vup=np.sqrt(2*1.2*up+(1.2*DT)**2)-1.2*DT;vdown=np.sqrt(2*1.2*down+(1.2*DT)**2)-1.2*DT
 wanted=np.clip(wanted,-vdown,vup)
 floor=(lo+np.maximum(last-aq,0)+.2*np.maximum(-aqd,0)-last)/.05
 ceiling=(hi-np.maximum(aq-last,0)-.2*np.maximum(aqd,0)-last)/.05
 wanted=np.minimum(np.maximum(wanted,np.clip(floor,-limit,limit)),np.clip(ceiling,-limit,limit))
 vel=oldvel+np.clip(wanted-oldvel,-1.2*DT,1.2*DT);proposed=last+DT*vel
 # The speed-dependent envelope limits the desired velocity above. Re-clipping
 # after acceleration limiting creates a velocity discontinuity when the dog
 # accelerates. Decelerate to that envelope at 1.2rad/s²; hard speed stays .6.
 low=np.maximum(np.maximum(oldvel-1.2*DT,-.6),(lo-last)/DT)
 high=np.minimum(np.minimum(oldvel+1.2*DT,.6),(hi-last)/DT)
 if np.any(low>high+1e-10):
  j=int(np.argmax(low-high))
  raise ValueError(f'J{j+1} 关节限位与加速度约束无法同时满足，需要停止并重新准备')
 vel=np.clip(vel,low,high);target=last+DT*vel
 return target,(target-last)/DT

def box_separation(pa,ra,ha,pb,rb,hb):
 rel=np.swapaxes(ra,-1,-2)@rb;t=(np.swapaxes(ra,-1,-2)@(pb-pa)[...,None])[...,0];ab=abs(rel)+1e-7
 da=abs(t)-ha-(ab@hb[...,None])[...,0];db=abs((np.swapaxes(rel,-1,-2)@t[...,None])[...,0])-hb-(np.swapaxes(ab,-1,-2)@ha[...,None])[...,0]
 best=np.maximum(da.max(-1),db.max(-1))
 for i in range(3):
  k,l=(i+1)%3,(i+2)%3
  for j in range(3):
   u,v=(j+1)%3,(j+2)%3;length=np.sqrt(np.maximum(1-rel[...,i,j]**2,1e-12))
   gap=abs(t[...,l]*rel[...,k,j]-t[...,k]*rel[...,l,j])-ha[...,k]*ab[...,l,j]-ha[...,l]*ab[...,k,j]-hb[...,u]*ab[...,i,v]-hb[...,v]*ab[...,i,u]
   best=np.maximum(best,np.where(length>1e-4,gap/length,-1e4))
 return best

class PhysicalGeometry(Geometry):
 def __init__(self):
  super().__init__();cfg=json.loads((ROOT/'models/arm_contract.json').read_text());caps=cfg['capsules']
  from geometry import MODEL
  meta=json.loads((ROOT/'models/manifest.json').read_text())
  for file,key in [(MODEL/'b2_a3.xml','model_sha256'),(ROOT/'models/arm_contract.json','arm_contract_sha256')]:
   if hashlib.sha256(file.read_bytes()).hexdigest()!=meta[key]:raise ValueError('安装几何版本与训练不一致')
  self.ids=np.array([c['body'] for c in caps]);self.bc=np.array([c['box_center'] for c in caps]);self.br=np.array([c['box_rotation'] for c in caps]);self.bh=np.array([c['box_half'] for c in caps]);self.pairs=np.array(cfg['pairs']);self.margins=np.array(cfg['pair_margins_m']);self.armcaps=np.array(cfg['arm_capsules']);self.margin=cfg['margin_m']
  self.cap_names=[c['name'] for c in caps];self.last_projection_issues=[]
  legpair=((self.ids[self.pairs]>=2)&(self.ids[self.pairs]<=13)).any(-1);self.reserve=np.where(self.margins>.0015,np.where(legpair,.005,.002),0)
 def state(self,dog,arm,reference):
  rs=dog['robot_state'];l=rs['locomotion_inputs'];o=rs['base_odometry'];a=arm['state'];p=self.measured(dog)
  yaw=p['rpy'][2]-reference['yaw'];rot=Rotation.from_euler('xyz',[*p['rpy'][:2],yaw]).as_matrix()
  pos=Rotation.from_euler('z',-reference['yaw']).apply(np.array(p['position'])-reference['position']);pos[2]=p['height']
  if o.get('velocity_world_m_s') is None:raise ValueError('missing independent world velocity vector')
  v=Rotation.from_euler('z',-reference['yaw']).apply(o['velocity_world_m_s']);gyro=np.array(l['gyro_body_rad_s']);w=rot@gyro
  aq=np.array(a['reality']['pose']);aqd=np.array(a['reality']['vel']);q=np.array(p['legs']);qd=np.array(l['dq_rad_s']).reshape(4,3)[[1,0,3,2]].ravel()
  body=[*p['rpy'][:2],yaw,p['height']-NOMINAL];self.set(body,aq,q,pos[:2]);ee=self.d.site_xpos[self.site].copy();eerot=self.d.site_xmat[self.site].reshape(3,3).copy()
  return dict(pos=pos,rot=rot,v=v,w=w,gyro=gyro,q=q,qd=qd,aq=aq,aqd=aqd,ee=ee,eerot=eerot,body=body)
 def details(self,s,aq):
  self.set(s['body'],aq,s['q'],s['pos'][:2]);P=self.d.xpos[self.ids];R=self.d.xmat[self.ids].reshape(-1,3,3);bp=P+(R@self.bc[...,None])[...,0];br=R@self.br;i,j=self.pairs.T
  sep=box_separation(bp[i],br[i],self.bh[i],bp[j],br[j],self.bh[j]);ground=np.min(bp[self.armcaps,2]-(abs(br[self.armcaps,2,:])*self.bh[self.armcaps]).sum(-1))
  valid=np.all(sep>=self.margins) and ground>=self.margin and np.all(aq>=self.arm_lo) and np.all(aq<=self.arm_hi)
  return bool(valid),sep-self.margins,float(ground)
 def violations(self,s,aq,clear=None,ground=None):
  if clear is None:_,clear,ground=self.details(s,aq)
  issues=[]
  for j,(q,lo,hi) in enumerate(zip(aq,self.arm_lo,self.arm_hi)):
   if q<lo or q>hi:
    issues.append({'code':'JOINT_LIMIT','joint':f'J{j+1}','value_deg':float(np.rad2deg(q)),'lower_deg':float(np.rad2deg(lo)),'upper_deg':float(np.rad2deg(hi)),'message':f'J{j+1} 实际/候选角度 {np.rad2deg(q):.2f}° 超出 [{np.rad2deg(lo):.2f}°, {np.rad2deg(hi):.2f}°]'})
  for idx in np.argsort(clear):
   if clear[idx]>=0:break
   a,b=self.pairs[idx];names=[self.cap_names[a],self.cap_names[b]]
   issues.append({'code':'SELF_COLLISION_CLEARANCE','links':names,'clearance_deficit_mm':float(-clear[idx]*1000),'message':f'{names[0]} ↔ {names[1]}：碰撞间距不足 {-clear[idx]*1000:.2f} mm（连杆约束，不是单个关节角度越限）'})
   if len(issues)>=10:break
  if ground<self.margin:issues.append({'code':'GROUND_CLEARANCE','message':f'机械臂最低点离地 {ground*1000:.1f} mm，要求至少 {self.margin*1000:.1f} mm'})
  return issues
 def saturation(self,s,u,target):
  issues=[];margin=np.array([.030,.030,.025,.020,.025,.025])
  for j,(q,lo,hi,demand,t) in enumerate(zip(s['aq'],self.arm_lo,self.arm_hi,u[6:],target)):
   lower=demand<-.02 and min(q,t)<=lo+margin[j]+.002
   upper=demand>.02 and max(q,t)>=hi-margin[j]-.002
   if lower or upper:issues.append({'code':'JOINT_COMMAND_MARGIN','joint':f'J{j+1}','value_deg':float(np.rad2deg(q)),'lower_deg':float(np.rad2deg(lo)),'upper_deg':float(np.rad2deg(hi)),'message':f'J{j+1} 请求继续向'+('下限' if lower else '上限')+f'运动，已进入指令保护余量；实际 {np.rad2deg(q):.2f}°，物理范围 [{np.rad2deg(lo):.2f}°, {np.rad2deg(hi):.2f}°]。联合策略仍可调整机体。'})
  return issues
 def project(self,s,u,last,oldvel,requested):
  self.last_projection_issues=[]
  target,vel=arm_action(s['aq'],s['aqd'],s['v'],s['w'],requested,u,last,oldvel,self.arm_lo,self.arm_hi)
  valid,clear,ground=self.details(s,s['aq']);reserve=np.minimum(clear,self.reserve);gr=min(ground-self.margin,.002)
  if not valid:
   issues=self.violations(s,s['aq'],clear,ground);raise ControlFault('PHYSICAL_CONSTRAINT',issues[0]['message'] if issues else '实际几何无效',issues)
  for blend in [.5,1.]:
   ok,dist,g=self.details(s,s['aq']+blend*(target-s['aq']))
   valid=valid and ok and np.all(dist>=reserve-1e-7) and g-self.margin>=gr-1e-7
   if not valid and not self.last_projection_issues:
    self.last_projection_issues=self.violations(s,s['aq']+blend*(target-s['aq']),dist,g)
    for idx in np.where(dist<reserve-1e-7)[0][:4]:
     a,b=self.pairs[idx];self.last_projection_issues.append({'code':'SWEPT_CLEARANCE','links':[self.cap_names[a],self.cap_names[b]],'message':f'候选运动缩小了 {self.cap_names[a]} ↔ {self.cap_names[b]} 的预留间距，机械臂该帧保持原目标'})
    if g-self.margin<gr-1e-7:self.last_projection_issues.append({'code':'SWEPT_GROUND','message':'候选运动缩小离地预留间距，机械臂该帧保持原目标'})
  return (target,vel,True) if valid else (last.copy(),np.zeros(6),False)
