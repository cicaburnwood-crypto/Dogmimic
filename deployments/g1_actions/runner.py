"""Action lifecycle independent of DDS; no commands until execute() is called."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import json,threading,time
from pathlib import Path
import numpy as np
from runtime import Controller,rotation

class Backend:
    def __init__(self, transport, models=None, log_dir=None):
        self.transport=transport
        base=Path(models or Path(__file__).parent/'models')
        self.controllers={s:Controller(base/s) for s in ('jump_60cm','funky_chicken')}
        self.lock=threading.RLock();self.action=None;self.phase=None;self.progress=None
        self.last_error=None;self.errors=[];self.worker=None
        self.log_dir=Path(log_dir or Path(__file__).parent/'results');self.log_dir.mkdir(parents=True,exist_ok=True)

    def model_info(self):
        return [{key:c.meta.get(key) for key in ('skill','label','iteration','duration_s',
                'checkpoint_sha256','weights_sha256','reference_alignment','physical_eval_passed','hardware_validated')}
                for c in self.controllers.values()]

    @staticmethod
    def stable(s):
        # These define the user's stable-standing condition; no dwell/unlock step.
        if 'native_fsm' in s and (s['native_fsm'] is None or s['native_fsm'] in (0,1,3)):
            return False  # Zero torque, damping or sitting is not stable standing.
        R=rotation(s['quaternion']);q=np.asarray(s['q']);dq=np.asarray(s['dq'])
        # Both legs extended, knees not folded, near-zero body/joint motion.
        return bool(R[2,2]>np.cos(np.deg2rad(15)) and
                    max(abs(q[[0,6]]))<.65 and max(abs(q[[3,9]]))<1.1 and
                    max(abs(q[[1,7]]))<.35 and np.max(abs(dq[:12]))<.5 and
                    np.linalg.norm(s['gyro'])<.25 and np.linalg.norm(s['velocity'])<.12)

    def status(self):
        with self.lock:
            try:
                s=self.transport.snapshot()
                connected=True;stable=self.stable(s)
            except Exception:
                s={};connected=False;stable=None
            return dict(connected=connected,stable_standing=stable,action=self.action,phase=self.phase,
                        models=self.model_info(),models_loaded=True,
                        progress=self.progress,last_error=self.last_error,errors=list(self.errors),
                        standing_reason={0:'原生零力矩模式',1:'原生阻尼模式',3:'原生坐姿模式'}.get(s.get('native_fsm')),
                        native_fsm=s.get('native_fsm'),
                        height=s.get('height'),velocity=float(np.linalg.norm(s['velocity'])) if s else None,
                        tilt=float(np.rad2deg(np.arccos(np.clip(rotation(s['quaternion'])[2,2],-1,1)))) if s else None,
                        battery=s.get('battery'))

    def execute(self, action):
        with self.lock:
            if action not in self.controllers:raise ValueError('未知动作')
            if self.worker and self.worker.is_alive():
                # Idempotent during execution: don't restart or overlap a trajectory.
                return {'action':self.action,'already_running':True}
            if not self.stable(self.transport.snapshot()):raise ValueError('请在稳定站立时执行动作')
            self.action=action;self.phase='接管控制';self.progress=0.;self.last_error=None;self.errors=[]
            self.worker=threading.Thread(target=self.run,args=(action,),daemon=True)
            self.worker.start()
            return {'action':action,'accepted':True}

    def run(self,action):
        c=self.controllers[action];c.reset();receipt=None
        path=self.log_dir/f'{action}_{time.time_ns()}.jsonl'
        try:
            receipt=self.transport.acquire(c.meta)
            # Sensor read happens after the potentially blocking ownership RPC.
            with path.open('w',buffering=1) as log:
                deadline=time.monotonic()
                for _ in range(c.meta['reference_frames']-1):
                    s=self.transport.snapshot()
                    start=time.monotonic()
                    target,info=c.step(s['q'],s['dq'],s['gyro'],s['quaternion'],s['position'])
                    inference_ms=(time.monotonic()-start)*1000
                    send_start=time.monotonic()
                    self.transport.command(target,c.meta,s)
                    command_ms=(time.monotonic()-send_start)*1000
                    self.phase='执行动作';self.progress=info['progress']
                    log.write(json.dumps(dict(time=time.time(),state=s,target_q=target.tolist(),
                                              inference_ms=inference_ms,command_ms=command_ms,**info))+'\n')
                    deadline+=c.meta['policy_dt'];delay=deadline-time.monotonic()
                    if delay>0:time.sleep(delay)
                    else:deadline=time.monotonic() # Never fast-forward missed reference frames.
            self.phase='轨迹结束，交回原生'
        except Exception as exc:
            self.last_error=str(exc);self.errors.append({'stage':'action','error':str(exc),'time':time.time()});self.phase='动作接口异常'
            with path.open('a') as f:f.write(json.dumps({'error':str(exc),'time':time.time()})+'\n')
        finally:
            if receipt is not None:
                try:
                    self.transport.release(receipt)
                    if not self.last_error:self.phase='已结束，原生控制'
                except Exception as exc:
                    self.errors.append({'stage':'restore','error':str(exc),'time':time.time()})
                    self.last_error='；'.join(e['error'] for e in self.errors);self.phase='控制交接异常'
                finally:
                    with path.open('a') as f:f.write(json.dumps({'event':'handover_result','time':time.time(),'errors':self.errors,'handover':getattr(self.transport,'handover_events',[])})+'\n')


def create_backend():
    from sdk_transport import SDKTransport
    return Backend(SDKTransport.from_environment())
