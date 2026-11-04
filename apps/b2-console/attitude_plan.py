"""Confirmed sequence of single-axis commands. Editing a browser draft is inert."""
import math


def validate_plan(value):
    if not isinstance(value, dict) or set(value)-{'mode', 'actions'}:
        raise ValueError('编排格式无效')
    mode = value.get('mode', 'return_each')
    if mode not in ('return_each', 'continuous'):
        raise ValueError('编排模式无效')
    actions = value.get('actions')
    if not isinstance(actions, list) or not actions:
        raise ValueError('请先添加至少一个动作')
    result = []
    for action in actions:
        if not isinstance(action, dict) or set(action) != {'axis', 'degrees'}:
            raise ValueError('动作需要方向和角度')
        axis, degrees = action['axis'], action['degrees']
        if axis not in ('roll', 'pitch'):
            raise ValueError('只支持 roll / pitch 方向')
        if isinstance(degrees, bool) or not isinstance(degrees, (int, float)) or not math.isfinite(degrees) or abs(degrees)>30:
            raise ValueError('角度须为 −30° 至 30° 的有限数值')
        result.append(dict(axis=axis, degrees=float(degrees)))
    return dict(mode=mode, actions=result)


class AttitudePlan:
    """Advance only after an inference target was successfully submitted."""
    def __init__(self, value):
        self.plan = validate_plan(value)
        self.index = 0
        self.phase = 'prepare'
        self.phase_steps = 0
        self.steps = 0

    def target(self):
        target = [0., 0.]
        if self.phase == 'execute':
            a = self.plan['actions'][self.index]
            target[0 if a['axis']=='roll' else 1] = math.radians(a['degrees'])
        return target

    def advance(self, stable, context):
        if self.phase == 'complete':
            return True
        self.steps += 1
        self.phase_steps += 1
        if self.phase == 'prepare' and self.phase_steps >= 100:
            self.phase = 'execute';self.phase_steps = 0
        elif self.phase == 'execute' and self.phase_steps >= 200:
            if self.plan['mode'] == 'continuous' and self.index+1 < len(self.plan['actions']):
                self.index += 1
            else:
                self.phase = 'recover'
            self.phase_steps = 0
        elif self.phase == 'recover':
            neutral = max(abs(float(x)) for x in context[:2]) < math.radians(.5)
            slow = max(abs(float(x)) for x in context[2:4]) < math.radians(1.)
            if stable and neutral and slow:
                if self.index+1 < len(self.plan['actions']):
                    self.index += 1;self.phase = 'execute';self.phase_steps = 0
                else:
                    self.phase = 'complete'
        return self.phase == 'complete'

    def status(self):
        return dict(plan=self.plan, plan_phase=self.phase, step_index=self.index,
                    step_number=self.index+1, step_count=len(self.plan['actions']),
                    policy_time_s=round(self.steps*.02, 2), phase_time_s=round(self.phase_steps*.02, 2),
                    complete=self.phase=='complete')
