#!/usr/bin/env python3
"""Local-only A3 console. Protocol taken from CVTE carm 0.1.20261022.
Startup/reconnect performs reads only. No queued/retried motion commands.
"""
import asyncio
import contextlib
import json
import math
import os
import secrets
import time
import uuid
from pathlib import Path
from aiohttp import web, ClientSession, ClientTimeout, WSMsgType

ROOT = Path(__file__).resolve().parent

def vector(value, n):
    if not isinstance(value, list) or len(value) != n:
        raise ValueError(f'需要 {n} 个数值')
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in value):
        raise ValueError('数值必须有限，不能包含 NaN / Infinity')
    return list(value)

def quat_from_rpy(v):
    r, p, y = [math.radians(x) / 2 for x in vector(v, 3)]
    cr,sr,cp,sp,cy,sy = math.cos(r),math.sin(r),math.cos(p),math.sin(p),math.cos(y),math.sin(y)
    return [sr*cp*cy-cr*sp*sy, cr*sp*cy+sr*cp*sy, cr*cp*sy-sr*sp*cy, cr*cp*cy+sr*sp*sy]

def rpy_from_quat(q):
    x,y,z,w = vector(q,4)
    return [math.degrees(math.atan2(2*(w*x+y*z),1-2*(x*x+y*y))), math.degrees(math.asin(max(-1,min(1,2*(w*y-z*x))))), math.degrees(math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))) ]

def pose_error(a,b):
    a,b=vector(a,7),vector(b,7)
    na,nb=math.sqrt(sum(x*x for x in a[3:])),math.sqrt(sum(x*x for x in b[3:]))
    if min(na,nb)<.99 or max(na,nb)>1.01: raise ValueError('控制器返回无效四元数')
    dot=abs(sum(x*y for x,y in zip(a[3:],b[3:]))/(na*nb))
    return math.dist(a[:3],b[:3]), math.degrees(2*math.acos(min(1,dot)))

class Arm:
    def __init__(self, endpoint='ws://10.42.0.101:8090', remote_url=None):
        self.endpoint=endpoint
        self.ws=None; self.raw={}; self.stamp=0.; self.pending={}; self.limits=None
        self.tool_coordinate=None; self.connected=False; self.problem='等待连接'; self.events=[]
        self.lock=asyncio.Lock(); self.epoch=0; self.heartbeats={}; self.active=None; self.frame=None
        self.token=secrets.token_urlsafe(32)
        self.last_auth_log=0
        from stream_control import StreamControl
        self.stream=StreamControl(self)
        from gripper_control import GripperControl
        self.gripper=GripperControl(self)
        from remote_monitor import RemoteMonitor
        self.remote_priority=RemoteMonitor(self,remote_url) if remote_url else None
    def require_remote_idle(self):
        if self.remote_priority: self.remote_priority.require()
    def log(self, kind, detail):
        item={'time':time.strftime('%H:%M:%S'), 'kind':kind, 'detail':detail}
        self.events=(self.events+[item])[-40:]
        (ROOT/'logs').mkdir(exist_ok=True)
        with (ROOT/'logs'/'events.jsonl').open('a') as f: f.write(json.dumps(item,ensure_ascii=False)+'\n')
    @property
    def state(self): return (self.raw.get('arm') or [{}])[0]
    def fresh(self):
        if not self.connected or time.monotonic()-self.stamp>.5: raise ValueError('实时状态中断，请等待重新连接')
        if self.state.get('arm_dof')!=6 or not self.state.get('arm_name','').startswith('A3'): raise ValueError('机械臂型号不匹配')
    def idle(self):
        self.fresh()
        if self.state.get('state') != 0 or self.state.get('task',{}).get('exe_flag') or self.active:
            raise ValueError('机械臂正在执行任务，请先停止或等待完成')
    def healthy(self):
        if self.raw.get('error') or self.state.get('state')==-1: raise ValueError(self.raw.get('errMsg') or '控制器报错，请检查后复位')
    def joints(self,q):
        q=vector(q,6)
        if not self.limits: raise ValueError('关节限位尚未读取')
        for i,(x,lo,hi) in enumerate(zip(q,self.limits['limit_lower'],self.limits['limit_upper'])):
            if x<lo-1e-7 or x>hi+1e-7: raise ValueError(f'J{i+1} 超出限位 {math.degrees(lo):.2f}° … {math.degrees(hi):.2f}°')
        return q
    async def request(self,command,**kw):
        priority_stop=kw.pop('_priority_stop',False)
        if not priority_stop and command in ('webRecieveTasks','setEffectorCtr','setServoEnable','setControlMode','setControllerErrorReset'): self.require_remote_idle()
        if not self.connected or not self.ws or self.ws.closed: raise ValueError('机械臂未连接')
        key=str(uuid.uuid4()); future=asyncio.get_running_loop().create_future()
        self.pending[key]=future
        packet={'command':command,'arm_index':0,'task_key':key,**kw}
        self.log('request',packet)
        try:
            await self.ws.send_json(packet)
            reply=await asyncio.wait_for(future,3)
            self.log('response',reply)
            if reply.get('recv')!='Task_Recieve': raise ValueError('控制器拒绝：'+json.dumps(reply,ensure_ascii=False))
            return reply
        finally: self.pending.pop(key,None)
    async def reader(self,ws):
        async for msg in ws:
            if msg.type!=WSMsgType.TEXT: continue
            data=json.loads(msg.data)
            if 'arm' in data and isinstance(data['arm'],list) and data['arm']:
                frame=data.get('Unix_time_stamp')
                if frame!=self.frame:
                    self.frame=frame; self.stamp=time.monotonic(); self.raw=data
                if self.active:
                    if self.state.get('task',{}).get('exe_flag'): self.active['seen']=True
                    elif not self.active.get('stream') and time.monotonic()-self.active['start']>1 and (self.active['seen'] or (max([abs(x) for x in self.state.get('reality',{}).get('vel',[1])])<.05 and max(abs(a-b) for a,b in zip(self.state.get('reality',{}).get('pose',[99]*6),self.active['target']))<.003)):
                        self.log('motion','任务已静止；请以实际角度 / 位姿核对到达结果'); self.active=None
            key=data.get('task_key'); future=self.pending.get(key)
            if future is not None and not future.done() and 'recv' in data: future.set_result(data)
            if data.get('command')=='onCarmError': self.log('controller_error',data)
        raise ConnectionError('机械臂连接关闭')
    async def run(self):
        async with ClientSession(timeout=ClientTimeout(total=5)) as session:
            while True:
                reader=None
                try:
                    async with session.ws_connect(self.endpoint,heartbeat=5) as ws:
                        self.ws=ws; self.connected=True; self.stamp=0; self.frame=None; self.limits=None
                        reader=asyncio.create_task(self.reader(ws))
                        p=await self.request('getJointParams')
                        limits=p['params']; vector(limits['limit_lower'],6); vector(limits['limit_upper'],6)
                        if any(a>=b for a,b in zip(limits['limit_lower'],limits['limit_upper'])): raise ValueError('无效限位')
                        self.limits=limits
                        for _ in range(30):
                            if self.stamp: break
                            await asyncio.sleep(.1)
                        self.fresh()
                        self.tool_coordinate=(await self.request('getCoordinate',type='tool',index=self.state.get('tool',0))).get('coordinate')
                        try: await self.gripper.read_params()
                        except Exception as e: self.log('gripper',str(e))
                        self.problem=''; self.log('connection','已连接；仅查询参数，不自动使能或恢复运动')
                        await reader
                except asyncio.CancelledError: raise
                except Exception as e:
                    self.problem=str(e); self.log('connection',str(e))
                finally:
                    self.connected=False; self.epoch+=1
                    self.gripper.params=None; self.gripper.last=None
                    if self.active: self.log('error','连接丢失，无法确认机械臂已停止；请使用硬件停止'); self.active=None
                    for f in list(self.pending.values()):
                        if not f.done(): f.set_exception(ConnectionError('连接丢失'))
                    if reader:
                        reader.cancel()
                        with contextlib.suppress(asyncio.CancelledError,Exception): await reader
                await asyncio.sleep(2)
    async def stop(self, emergency=False):
        self.epoch+=1
        result=await self.request('stopSignals',stop_id='SIG_EMERGENCY_STOP' if emergency else 'SIG_TASK_STOP',**({'step_cnt':5} if emergency else {'stop_at_once':True}))
        self.active=None
        return result
    async def watchdog(self):
        while True:
            await asyncio.sleep(.1)
            if self.active and (time.monotonic()-self.heartbeats.get(self.active['client'],0)>2 or time.monotonic()-self.stamp>.5):
                try:
                    await self.stop(); self.log('watchdog','页面心跳或状态超时，已请求停止')
                except Exception as e:
                    self.active=None; self.log('error','停止未确认：'+str(e))
    async def fk(self,q,tool):
        r=await self.request('getKinematics',task_id='forward',data={'tool':tool,'point_cnt':1,'joint1':self.joints(q)})
        return vector(r.get('data',{}).get('point1'),7)
    async def solve(self,body):
        self.fresh(); tool=self.state.get('tool',0)
        q0=vector(self.state.get('reality',{}).get('pose'),6)
        kind=body.get('kind')
        if kind=='joint':
            q=[math.radians(x) for x in vector(body.get('joints_deg'),6)]
            if 'joint_index' in body:
                i=body['joint_index']
                if type(i)!=int or not 0<=i<6: raise ValueError('无效关节编号')
                target=q[i]; q=q0.copy(); q[i]=target
            q=self.joints(q); pose=await self.fk(q,tool)
        elif kind=='ee':
            xyz=[x/1000 for x in vector(body.get('xyz_mm'),3)]
            pose=xyz+quat_from_rpy(body.get('rpy_deg'))
            r=await self.request('getKinematics',task_id='inverse',data={'tool':tool,'point_cnt':1,'chained':False,'point1':pose,'refer1':q0})
            candidate=r.get('data',{}).get('joint1')
            if not candidate:
                current_pose=await self.fk(q0,tool)
                pe,ae=pose_error(pose,current_pose)
                if pe>1e-6 or ae>.001:
                    raise ValueError('控制器未找到有效逆解：目标可能不可达或接近奇异位姿；未发送运动指令')
                candidate=q0  # Exact current FK is already a valid solution, including singular poses.
            q=self.joints(candidate)
            check=await self.fk(q,tool); pe,ae=pose_error(pose,check)
            if pe>.001 or ae>.1: raise ValueError(f'IK 未满足目标：位置残差 {pe*1000:.2f} mm，角度 {ae:.2f}°')
        else: raise ValueError('未知操作类型')
        self.fresh()
        if tool!=self.state.get('tool',0): raise ValueError('工具号变化，请重新求解')
        return {'joints_deg':[math.degrees(x) for x in q], 'joints_rad':q,'pose':pose,'xyz_mm':[x*1000 for x in pose[:3]],'rpy_deg':rpy_from_quat(pose[3:]),'tool':tool}
    async def action(self,body):
        op=body.get('op'); client=body.get('client','')
        if not isinstance(client,str) or not 8<=len(client)<=100: raise ValueError('无效页面会话')
        if op not in ('heartbeat','stop','emergency','preview','stream_hold'): self.require_remote_idle()
        if op=='gripper_set': return await self.gripper.action(body)
        if op.startswith('stream_'):return await self.stream.action(body)
        if op=='heartbeat': self.heartbeats[client]=time.monotonic(); return {}
        # Stops bypass the operation lock and invalidate any outstanding solve/enable.
        if op in ('stop','emergency'): await self.stop(op=='emergency'); return {'message':'停止请求已接收，请查看实际状态'}
        if op=='disable':
            self.epoch+=1
            await self.request('setServoEnable',enable=False); self.active=None
            return {'message':'解除使能请求已接收'}
        if self.lock.locked(): raise ValueError('另一条操作正在处理，请等待；停止按钮始终可用')
        async with self.lock:
            epoch=self.epoch
            self.fresh()
            if op=='preview': return await self.solve(body)
            if op=='reset':
                if self.active or self.state.get('task',{}).get('exe_flag'): raise ValueError('请先停止运动')
                await self.request('setControllerErrorReset')
                if epoch!=self.epoch: raise ValueError('操作被停止取消')
                await self.request('stopSignals',stop_id='SIG_ARM_RECOVER',step_cnt=5)
                return {'message':'已请求故障复位，不会自动使能或运动'}
            self.idle(); self.healthy()
            if op=='enable':
                # Match the teach pendant shipped on controller 1.0.260819:
                # reset controller -> servo enable -> position mode with params.level.
                # The IDLE controller may reject mode changes even when error == 0.
                await self.request('setControllerErrorReset',sig_name='SIG_C_ERROR',task_id='TASK_SIGNAL',task_level='Task_General')
                await asyncio.sleep(.1)
                if epoch!=self.epoch: raise ValueError('操作被停止取消')
                self.fresh(); self.healthy()
                if self.state.get('servo')!=1:
                    await self.request('setServoEnable',enable=True)
                    for _ in range(20):
                        if epoch!=self.epoch: raise ValueError('操作被停止取消')
                        self.fresh(); self.healthy()
                        if self.state.get('servo')==1: break
                        await asyncio.sleep(.05)
                    else: raise ValueError('伺服使能未确认，未发送模式切换')
                if epoch!=self.epoch: raise ValueError('操作被停止取消')
                await self.request('setControlMode',mode=1,params={'level':100})
                for _ in range(20):
                    if epoch!=self.epoch: raise ValueError('操作被停止取消')
                    self.fresh()
                    if self.state.get('servo')==1 and self.state.get('fsm_state')=='POSITION': return {'message':'已使能，位置控制模式'}
                    await asyncio.sleep(.1)
                raise ValueError('使能 / 位置模式尚未确认，请查看控制器状态')
            if op=='move':
                if self.state.get('servo')!=1 or self.state.get('fsm_state')!='POSITION': raise ValueError('请先使能并进入位置模式')
                speed=body.get('speed',5)
                if isinstance(speed,bool) or not isinstance(speed,(int,float)) or not math.isfinite(speed) or not 1<=speed<=100: raise ValueError('速度比例必须为 1–100%')
                solution=await self.solve(body)
                if epoch!=self.epoch: raise ValueError('操作被停止取消')
                self.idle(); self.healthy()
                if self.state.get('servo')!=1 or self.state.get('fsm_state')!='POSITION': raise ValueError('伺服状态变化，请重新操作')
                if time.monotonic()-self.heartbeats.get(client,0)>2: raise ValueError('页面心跳过期')
                self.active={'client':client,'start':time.monotonic(),'seen':False,'target':solution['joints_rad']}
                try:
                    await self.request('webRecieveTasks',task_id='TASK_MOVJ',task_level='Task_General',point_type={'space':0},data={'tool':solution['tool'],'target_pos':solution['joints_rad'],'speed':speed})
                except Exception:
                    with contextlib.suppress(Exception): await self.stop()
                    raise
                return {**solution,'message':'运动请求已接收（关节空间点到点），请观察实际反馈'}
            raise ValueError('未知操作')
    def snapshot(self):
        s=self.state; age=time.monotonic()-self.stamp if self.stamp else None
        pose=s.get('pose_tool') or s.get('pose')
        return {'connected':self.connected,'fresh':bool(age is not None and age<.5 and self.connected),'age_ms':round(age*1000) if age is not None else None,'state':s,'error':self.raw.get('error'),'errMsg':self.raw.get('errMsg'),'debug':self.raw.get('debug'),'limits':self.limits,'tool_coordinate':self.tool_coordinate,'rpy_deg':rpy_from_quat(pose[3:]) if pose and len(pose)==7 else None,'problem':self.problem,'events':[e for e in self.events if e['kind'] not in ('request','response')][-10:],'active':bool(self.active),'stream':self.stream.status(),'gripper_control':self.gripper.snapshot(),'remote_priority':self.remote_priority.status() if self.remote_priority else None}

def make_app(arm):
    @web.middleware
    async def security(request,handler):
        if request.host not in ('127.0.0.1:8879','localhost:8879'): return web.json_response({'error':'Host rejected'},status=403)
        if request.method=='POST':
            if request.headers.get('Origin') not in (None,'http://127.0.0.1:8879','http://localhost:8879'):
                return web.json_response({'error':'请求来源不匹配','code':'ORIGIN_REJECTED'},status=403)
            if not secrets.compare_digest(request.headers.get('X-Arm-Token',''),arm.token):
                if time.monotonic()-arm.last_auth_log>5:
                    arm.log('session','页面使用旧会话，未向机械臂发送命令；请刷新页面')
                    arm.last_auth_log=time.monotonic()
                return web.json_response({'error':'页面会话已过期，请刷新页面；操作没有执行','code':'SESSION_EXPIRED'},status=403)
        try: return await handler(request)
        except web.HTTPException:
            raise
        except (ValueError,ConnectionError,TimeoutError,KeyError) as e:
            message=str(e) or '控制器响应超时，未重发命令'
            arm.log('rejected',message)
            return web.json_response({'error':message},status=400)
        except Exception as e:
            arm.log('error',str(e)); return web.json_response({'error':str(e)},status=500)
    app=web.Application(middlewares=[security],client_max_size=16384)
    async def index(r):
        return web.Response(text=(ROOT/'web'/'index.html').read_text().replace('__TOKEN__',arm.token),content_type='text/html',headers={'Cache-Control':'no-store','Content-Security-Policy':"default-src 'self'; script-src 'self'; style-src 'self'; frame-ancestors 'none'"})
    async def session(r): return web.json_response({'token':arm.token},headers={'Cache-Control':'no-store'})
    async def status(r): return web.json_response(arm.snapshot(),headers={'Cache-Control':'no-store'})
    async def action(r):
        body=await r.json()
        if not isinstance(body,dict): raise ValueError('需要 JSON 对象')
        return web.json_response({'ok':True,**await arm.action(body)})
    app.router.add_get('/',index); app.router.add_get('/api/status',status); app.router.add_get('/api/session',session); app.router.add_post('/api/action',action)
    app.router.add_static('/static/',ROOT/'web')
    async def lifecycle(app):
        tasks=[asyncio.create_task(arm.run()),asyncio.create_task(arm.watchdog()),asyncio.create_task(arm.stream.watchdog())]
        if arm.remote_priority: tasks.append(asyncio.create_task(arm.remote_priority.run()))
        yield
        if arm.active:
            with contextlib.suppress(Exception): await arm.stop()
        for t in tasks: t.cancel()
        await asyncio.gather(*tasks,return_exceptions=True)
    app.cleanup_ctx.append(lifecycle)
    return app

def load_token(path):
    try:
        token=path.read_text().strip()
        if len(token)>=32: return token
    except FileNotFoundError: pass
    token=secrets.token_urlsafe(32)
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o600)
    with os.fdopen(fd,'w') as f: f.write(token)
    return token

if __name__=='__main__':
    arm=Arm(remote_url='http://127.0.0.1:18865/api/status'); arm.token=load_token(ROOT/'.session-token')
    web.run_app(make_app(arm),host='127.0.0.1',port=8879,access_log=None)
