"""Independent, bounded effector position commands; never send arm joint targets."""
import asyncio
import math
import time


def scalar(value):
    if not isinstance(value, list) or len(value) != 1:
        raise ValueError('夹爪单自由度参数或反馈无效')
    x = value[0]
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        raise ValueError('夹爪参数或反馈不是有限数值')
    return x


class GripperControl:
    OPEN = .040  # Metres, verified in the gripper-only hardware check.
    CLOSED = 0.
    FORCE = 5.

    def __init__(self, arm):
        self.arm = arm
        self.params = None
        self.last = None
        self.lock = asyncio.Lock()
        self.requests = {}

    async def read_params(self):
        self.params = None
        r = await self.arm.request('getEeffParams')
        if r.get('ret') is False:
            raise ValueError('控制器拒绝读取夹爪参数')
        p = r.get('params', {})
        if p.get('eeff_type') != 'gripper' or p.get('dof') != 1:
            raise ValueError('末端不是单自由度夹爪')
        lo, hi, force = (scalar(p.get(k)) for k in ('eeff_lower', 'eeff_upper', 'eeff_tau'))
        if not lo <= self.CLOSED < self.OPEN <= hi or force < self.FORCE:
            raise ValueError('夹爪限位不支持 0–40 mm / 5 N，未发送动作')
        self.params = p

    def ready(self, client=None):
        a = self.arm
        a.fresh()
        a.healthy()
        if a.raw.get('debug') is not False:
            raise ValueError('夹爪需要真实硬件反馈')
        e = a.state.get('eeff', {})
        if e.get('eeff_type') != 'gripper' or e.get('eeff_dof') != 1 or e.get('is_connect') is not True:
            raise ValueError('夹爪未连接或类型不匹配')
        if e.get('eeff_state') != 1:
            raise ValueError('夹爪未使能或状态异常，请在机械臂控制台检查')
        if self.params is None or e.get('eeff_name') != self.params.get('eeff_name'):
            raise ValueError('夹爪限位参数尚未就绪')
        pos = scalar(e.get('eeff_pos'))
        if not scalar(self.params['eeff_lower']) - .005 <= pos <= scalar(self.params['eeff_upper']) + .005:
            raise ValueError('夹爪位置反馈超出物理范围')
        if client is not None:
            active = a.active
            if active and (not active.get('stream') or active.get('client') != client):
                raise ValueError('机械臂准备或其他控制会话执行中，请等待完成')
            if not active and a.state.get('task', {}).get('exe_flag'):
                raise ValueError('控制器正在执行其他机械臂任务')
        return e, pos

    def snapshot(self):
        error = None
        e = self.arm.state.get('eeff', {})
        pos = None
        try:
            e, pos = self.ready()
        except ValueError as exc:
            error = str(exc)
        planned = pos
        try:
            planned = scalar(e.get('eeff_plan_pos'))
        except ValueError:
            pass
        # Use commanded opening: a closed gripper holding an object must next open.
        if self.last and time.monotonic() - self.last['time'] < .5:
            planned = self.last['target_m']
        phase = 'idle'
        target = None
        if self.last:
            target = self.last['target_m']
            phase = self.last['phase']
            if phase == 'accepted' and pos is not None:
                phase = 'at_target' if abs(pos - target) <= .005 else ('not_reached' if time.monotonic() - self.last['time'] > 5 else 'moving')
        return {'ready': error is None, 'error': error, 'position_mm': None if pos is None else pos * 1000,
                'target_mm': None if target is None else target * 1000, 'phase': phase,
                'next_open': planned is None or planned < (self.OPEN + self.CLOSED) / 2,
                'open_mm': self.OPEN * 1000, 'closed_mm': 0, 'force_n': self.FORCE,
                'pending': self.lock.locked()}

    async def interrupt(self):
        if not self.last or self.last['phase'] not in ('pending','accepted','unconfirmed'):
            return
        # Stop an in-flight opening/closing at measured gap; never release an object.
        _, pos = self.ready()
        if abs(pos - self.last['target_m']) <= .001 and self.last['phase'] == 'accepted':
            return
        self.last = {'target_m': pos, 'time': time.monotonic(), 'phase': 'paused'}
        await self.arm.request('setEffectorCtr', pos=[pos], vel=[0.], tau=[self.FORCE], control_motor=False, _priority_stop=True)

    async def action(self, body):
        client, ident, target = body.get('client'), body.get('id'), body.get('target')
        if not isinstance(ident, str) or not 8 <= len(ident) <= 100 or target not in ('open', 'closed'):
            raise ValueError('需要明确的夹爪开合目标与请求编号')
        key = (client, ident)
        if key in self.requests:
            old = self.requests[key]
            if old['target'] != target:
                raise ValueError('夹爪请求编号已用于另一个目标')
            if old.get('error'):
                raise ValueError(old['error'])
            return old['result']
        if self.lock.locked():
            raise ValueError('夹爪指令等待应答，请松开空格后再操作')
        async with self.lock:
            self.ready(client)
            pos = self.OPEN if target == 'open' else self.CLOSED
            item = {'target': target, 'error': '夹爪指令应答尚未确认，未自动重发'}
            self.requests[key] = item
            if len(self.requests) > 128:
                self.requests.pop(next(iter(self.requests)))
            command_state = {'target_m': pos, 'time': time.monotonic(), 'phase': 'pending'}
            self.last = command_state
            try:
                r = await self.arm.request('setEffectorCtr', pos=[pos], vel=[0.], tau=[self.FORCE], control_motor=False)
                if r.get('recv') != 'Task_Recieve' or r.get('ret') is False:
                    raise ValueError('控制器拒绝夹爪开合指令')
            except Exception as exc:
                command_state['phase'] = 'unconfirmed'
                item['error'] = str(exc) or '夹爪应答超时，执行结果未知；请查看实际位置，未自动重发'
                raise ValueError(item['error']) from exc
            command_state['phase'] = 'accepted'
            result = {'message': f'已请求夹爪{"张开 40 mm" if target == "open" else "闭合"}，请以位置反馈核对', 'gripper': self.snapshot()}
            item.update(error=None, result=result)
            return result
