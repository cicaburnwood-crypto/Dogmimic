"""Bounded, expiring pose-only overlay for existing P4. Never generates vx/vy."""
import math
import time

class StationaryReach:
    def __init__(self,engine):
        self.e=engine; self.engaged=False;self.client=None;self.deadline=0;self.anchor=None
        self.target=None;self.hold=None;self.fault=None;self.last=0
    def pose(self):
        t=self.e.telemetry;n=self.e.clock()
        if not t.fresh(n) or not t.position_fresh(n) or t.position_source not in ('odom','lf_odom'):raise ValueError('联合操作需要实时 IMU 与独立里程计')
        from sensor_feedback import rpy_degrees
        rpy=rpy_degrees(t.quaternion);oq=rpy_degrees(t.position_quaternion)
        if rpy is None or oq is None:raise ValueError('姿态反馈无效')
        return [math.radians(rpy[0]),math.radians(rpy[1]),math.radians(oq[2])],list(t.position)
    def limit(self,h):
        knots=self.e.height_config['tilt_knots']
        for (a,lo),(b,hi) in zip(knots,knots[1:]):
            if h<=b:return math.radians(lo+(hi-lo)*max(0,min(1,(h-a)/(b-a))))
        return math.radians(knots[-1][1])
    def validate(self,value):
        target=value.get('target')
        if not isinstance(target,list) or len(target)!=4 or not all(type(x) in (int,float) and math.isfinite(x) for x in target):raise ValueError('目标必须为 roll,pitch,yaw 弧度和高度增量米')
        r,p,y,h=target
        if not -.2<=h<=.05 or max(abs(r),abs(p))>self.limit(h)+1e-8:raise ValueError('目标超过 P4 高度 / 倾角包络')
        return target
    def request(self,value):
        if not isinstance(value,dict):raise ValueError('无效联合请求')
        if set(value)-{'op','client','target'}:raise ValueError('禁止平移或其他未支持字段')
        op=value.get('op');client=value.get('client')
        if not isinstance(client,str) or not 8<=len(client)<=100:raise ValueError('无效执行会话')
        if op=='begin':
            if self.engaged and self.client!=client:raise ValueError('已有联合任务')
            if not self.e.height_config:raise ValueError('当前策略未启用高度控制')
            target=self.validate(value);rpy,pos=self.pose()
            if not self.e._tilt_stable(allow_tilt=self.e.active=='trained'):raise ValueError('联合到点仅允许在稳定站立时开始')
            if any(abs(v)>.05 for v in self.e.telemetry.remote['axes'].values()) or self.e.telemetry.remote['buttons']:raise ValueError('请将遥控器回中并松开按键')
            if self.e.active=='native':
                self.e.selected='native'
                self.e.toggle_locomotion()
            if self.e.active!='trained' or self.e.tilt_returning or self.e.tilt_handover:raise ValueError('P4 尚未就绪')
            if not self.engaged:
                self.anchor=pos[:2];self.hold=[self.e.tilt_request[0],self.e.tilt_request[1],rpy[2],self.e.height_request]
            self.engaged=True;self.client=client;self.target=target;self.fault=None;self.deadline=self.e.clock()+.8;self.last=self.e.clock()
        elif op in ('update','hold','release'):
            if not self.engaged or client!=self.client:raise ValueError('联合会话不匹配或已被遥控器接管')
            if op=='update':
                if self.fault:raise ValueError(self.fault)
                self.target=self.validate(value);self.deadline=self.e.clock()+.8
            elif op=='hold':self.freeze('已停止选点执行，保持当前姿态')
            else:self.release()
        else:raise ValueError('无效联合操作')
        return self.status()
    def freeze(self,reason):
        if not self.engaged:return
        self.fault=reason;self.deadline=0
        self.target=self.hold[:]
    def release(self):
        if self.engaged:
            self.e.tilt_request=self.hold[:2];self.e.height_request=self.hold[3]
        self.engaged=False;self.client=None;self.target=None
    def command(self,now):
        if not self.engaged:return None
        dt=min(.04,max(0,now-self.last));self.last=now
        try:
            rpy,pos=self.pose()
            if math.hypot(pos[0]-self.anchor[0],pos[1]-self.anchor[1])>.05:self.freeze('水平漂移超过 5 cm，已停止转向和姿态调整')
            if now>self.deadline and not self.fault:self.freeze('控制心跳中断，保持姿态并停止转向')
            if self.fault:return [0.,0.,0.,self.hold[0],self.hold[1],self.hold[3]]
            def slew(v,w,d):return v+max(-d,min(d,w-v))
            limit=self.limit(self.hold[3])
            for i in (0,1):self.hold[i]=slew(self.hold[i],max(-limit,min(limit,self.target[i])),math.radians(2)*dt)
            height=slew(self.hold[3],self.target[3],.02*dt)
            # Reduce tilt first when lowering narrows the admissible envelope.
            if max(abs(self.hold[0]),abs(self.hold[1]))<=self.limit(height)+1e-9:self.hold[3]=height
            self.hold[2]=rpy[2]
            error=(self.target[2]-rpy[2]+math.pi)%(2*math.pi)-math.pi
            yaw=0. if abs(error)<math.radians(1) else max(-.15,min(.15,error*.7))
            # The first two elements are constants even on timeout/fault.
            return [0.,0.,yaw,self.hold[0],self.hold[1],self.hold[3]]
        except ValueError as exc:
            self.freeze(str(exc));return [0.,0.,0.,self.hold[0],self.hold[1],self.hold[3]]
    def status(self):
        return dict(engaged=self.engaged,client=self.client,target=self.target,hold=self.hold,fault=self.fault,translation_allowed=False,max_drift_m=.05)
