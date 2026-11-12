"""Physical remote arbitration, independent of button mappings or gait policy."""
import math


class RemotePriority:
    QUIET_SECONDS = 2.

    def __init__(self):
        self.active = False
        self.fresh = False
        self.quiet_since = None
        self.seen_input = False
        self.generation = 0

    def update(self, remote, fresh, now):
        axes = remote.get('axes', {})
        valid = isinstance(axes, dict) and all(isinstance(v, (int, float)) and math.isfinite(v) for v in axes.values())
        fresh = bool(fresh and valid)
        active = bool(remote.get('buttons') or any(v != 0 for v in axes.values())) if fresh else False
        if (active and not self.active) or (not fresh and self.fresh):
            self.generation += 1
        if active or not fresh:
            self.seen_input = True
            self.quiet_since = None
        elif self.seen_input and self.quiet_since is None:
            self.quiet_since = now
        self.active, self.fresh = active, fresh
        return self.status(now)

    def status(self, now):
        remaining = self.QUIET_SECONDS if self.quiet_since is None else max(0., self.QUIET_SECONDS - (now - self.quiet_since))
        if not self.seen_input and self.fresh:
            remaining = 0.
        blocked = not self.fresh or self.active or remaining > 0
        reason = '遥控器反馈过期' if not self.fresh else '遥控器输入优先，其他控制暂时暂停' if self.active else f'等待遥控器连续静默 2 秒（剩余 {remaining:.1f} 秒）' if blocked else ''
        return dict(blocked=blocked, active=self.active, fresh=self.fresh, quiet_remaining_s=remaining,
                    generation=self.generation, quiet_seconds=self.QUIET_SECONDS, axis_threshold=0., reason=reason)
