import os,sys,threading,time,json,io,math,argparse,hashlib
from pathlib import Path
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
os.environ['MUJOCO_GL']='egl'
ROOT=Path(__file__).resolve().parent.parent;sys.path.insert(0,str(Path(__file__).resolve().parent))
from bootstrap import HERE
from joint import *
from arm import ArmController
from supermimic import networks
import mujoco,numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
CHECKPOINT=ROOT/'checkpoints/v18_best24500.pt'
CHECKPOINT_SHA='d5540cc2f1a8a283c38c9e644c9693d2b6317158d9beb2172f6d67c0594da703'
P4=ROOT/'checkpoints/p4_20000.pt'
P4_SHA='e0314388419456d6597e5b44502508dbac00da0019592da4e79bf3a6c992457b'
CRITERIA=dict(position_m=.01,orientation_deg=10,simultaneous_hold_s=1)
lock=threading.Lock();pending=[];frame=b'';state={'phase':'loading','checkpoint':24500,'model_version':'v18'}
def worker():
 global frame,state
 try:
  assert hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()==CHECKPOINT_SHA,'Selected checkpoint hash mismatch'
  assert hashlib.sha256(P4.read_bytes()).hexdigest()==P4_SHA,'Frozen P4 hash mismatch'
  # Exactly the envelope used by v18 formal training and physical evaluation.
  old_envelope=ArmController.envelope
  ArmController.envelope=lambda self,speed:(old_envelope(self,speed)[0],old_envelope(self,speed)[1]*1.25)
  torch.set_num_threads(2);networks.AMP['enabled']=False;torch.manual_seed(24500)
  e=JointEnv('cuda',1,randomize=False,compile=False,autoreset=False);e.reset();prev=torch.zeros(1,12,device='cuda')
  ck=torch.load(CHECKPOINT,map_location='cuda',weights_only=False)
  assert ck['it']==24000 and ck['lower_sha256']==P4_SHA and ck['wm_arm_normalization_version']==1
  ep,er=ee(e.arm,e.state);goal=torch.cat([ep,er.flatten(1)],1)
  manager=Manager(observe(e.state,e.context,goal,prev,e.arm,e.xi).shape[-1],e.arm).cuda();manager.load_state_dict(ck['manager']);manager.eval()
  lower=make_policy().cuda();lower.load_state_dict(torch.load(P4,map_location='cuda',weights_only=False)['policy']);lower.eval();lower.requires_grad_(False)
  m=mujoco.MjModel.from_xml_path(str(HERE/'b2_a3.xml'));m.vis.global_.offwidth=1000;m.vis.global_.offheight=700;d=mujoco.MjData(m);renderer=mujoco.Renderer(m,height=700,width=1000);cam=mujoco.MjvCamera();cam.distance=4.7;cam.azimuth=135;cam.elevation=-30;cam.lookat[:]=[0,0,.4]
  active=False;steps=0;hold=0.;phase='ready'
  with torch.no_grad():
   while True:
    started=time.monotonic()
    with lock: commands_pending=pending[:];pending.clear()
    for command in commands_pending:
     if command['type']=='reset':
      e.reset();e.upper.zero_();prev.zero_();ep,er=ee(e.arm,e.state);goal=torch.cat([ep,er.flatten(1)],1);active=False;steps=0;hold=0.;phase='ready'
     else:
      xyz=command['xyz'];rot=Rotation.from_euler('xyz',command['rpy'],degrees=True).as_matrix();goal=torch.tensor([xyz+rot.flatten().tolist()],device='cuda',dtype=torch.float32);active=True;hold=0.;steps=0;phase='running'
    if active:
     for _ in range(5):
      u=manager(e.state,e.context,goal,prev,e.xi);e.upper=u;e.set_target(commands(u));s,info=e.step(lower(*e.policy_inputs()));prev=u;steps+=1
      dist,ang,_=errors(e.arm,s,goal);hit=float(dist[0])<=CRITERIA['position_m'] and float(ang[0])<=math.radians(CRITERIA['orientation_deg']);hold=hold+DT if hit else 0.
      if bool(info['terminated'][0]):active=False;phase='terminated';break
      if hold>=1-1e-6:active=False;phase='success';break
      if steps>=600:active=False;phase='timeout';break
    s=e.state;ep,er=ee(e.arm,s);dist,ang,_=errors(e.arm,s,goal)
    pos=s['pos'][0].cpu().numpy();d.qpos[:3]=pos;quat=np.empty(4);mujoco.mju_mat2Quat(quat,s['rot'][0].cpu().numpy().astype(float).flatten());d.qpos[3:7]=quat;d.qpos[7:19]=s['q'][0].cpu().numpy();d.qpos[19:25]=s['aq'][0].cpu().numpy();mujoco.mj_forward(m,d)
    g=goal[0].cpu().numpy();cam.lookat[:]=[(pos[0]+g[0])/2,(pos[1]+g[1])/2,.5];renderer.update_scene(d,camera=cam);scene=renderer.scene;geom=scene.geoms[scene.ngeom];mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_SPHERE,np.array([.04]*3),g[:3],np.eye(3).flatten(),np.array([1.,.25,.1,1.]));scene.ngeom+=1
    image=Image.fromarray(renderer.render());buf=io.BytesIO();image.save(buf,format='JPEG',quality=85)
    new=dict(phase=phase,elapsed=steps*DT,hold=hold,position_error=float(dist[0]),orientation_error=float(torch.rad2deg(ang[0])),base=pos.tolist(),ee=ep[0].cpu().tolist(),goal=g[:3].tolist(),rpy=Rotation.from_matrix(g[3:].reshape(3,3)).as_euler('xyz',degrees=True).tolist(),checkpoint=24500,model_version='v18',checkpoint_path=str(CHECKPOINT),checkpoint_sha256=CHECKPOINT_SHA,environment=str(HERE),criteria=CRITERIA,simulation_only=True)
    with lock:frame=buf.getvalue();state=new
    time.sleep(max(0,.1-(time.monotonic()-started)))
 except Exception as exc:
  import traceback;traceback.print_exc()
  with lock:state={'phase':'error','message':str(exc)}
class Handler(BaseHTTPRequestHandler):
 def log_message(self,*args):pass
 def send(self,data,content='application/json',status=200):
  self.send_response(status);self.send_header('Content-Type',content);self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(data)
 def do_GET(self):
  if self.path.startswith('/frame'):
   with lock:data=frame
   self.send(data,'image/jpeg' if data else 'text/plain',200 if data else 503)
  elif self.path=='/status':
   with lock:data=json.dumps(state).encode()
   self.send(data)
  elif self.path=='/' or self.path=='/index.html':self.send(Path(__file__).with_name('index.html').read_bytes(),'text/html; charset=utf-8')
  else:self.send(b'{}',status=404)
 def do_POST(self):
  try:
   size=int(self.headers.get('Content-Length','0'))
   if size<0 or size>4096:raise ValueError('request too large')
   body=json.loads(self.rfile.read(size) or b'{}')
   if self.path=='/reset':command={'type':'reset'}
   elif self.path=='/target':
    xyz=body['xyz'];rpy=body['rpy']
    if len(xyz)!=3 or len(rpy)!=3 or not all(isinstance(v,(float,int)) and math.isfinite(v) for v in xyz+rpy):raise ValueError('invalid coordinates')
    if math.hypot(*xyz[:2])>3 or not .1<=xyz[2]<=1.6 or any(abs(v)>180 for v in rpy):raise ValueError('目标须在起点水平3米内，高度0.1–1.6米，角度±180°')
    command={'type':'target','xyz':xyz,'rpy':rpy}
   else:raise ValueError('unknown endpoint')
   with lock:
    if state['phase'] in ['loading','error']:raise ValueError('仿真尚未就绪')
    if command['type']=='target' and state['phase']=='terminated':raise ValueError('请先重置仿真，再执行新目标')
    pending.append(command)
   self.send(b'{"ok":true}')
  except (ValueError,KeyError,TypeError) as exc:self.send(json.dumps({'error':str(exc)}).encode(),status=400)
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=8878);args=parser.parse_args()
 server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
 threading.Thread(target=worker,daemon=True).start();server.serve_forever()
